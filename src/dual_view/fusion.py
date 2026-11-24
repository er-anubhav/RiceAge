"""
Dual-View Fusion and Aggregation Engine.
Merges matched and unmatched ViewObservation instances into canonical PhysicalGrain records,
enforces agronomic defect precedence, and executes non-duplicating sample-level aggregation.

Supports:
  - 'bottom_primary' (Authoritative Delivery Architecture): Bottom scanner is the single authoritative
    source of truth for counts, dimensions, whole/broken, defects, and denominators. Top scanner provides
    secondary verification and discrepancy flags only. Top detections are NEVER added to primary count.
  - 'experimental_canonical_fusion' (Phase 4/4.1 Prototype): Preserved behind feature flag for experimental
    symmetric research.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import cv2

from src.dual_view.models import (
    ViewObservation,
    PhysicalGrain,
    CorrespondenceStatus,
    DualViewFusionResult
)
from src.dual_view.correspondence import DualViewCorrespondenceEngine



def assess_image_quality(
    image_bgr: Optional[np.ndarray],
    min_laplacian_var: float = 8.0,
    min_contrast: float = 12.0,
    min_brightness: float = 15.0,
    max_brightness: float = 240.0
) -> Dict[str, Any]:
    """
    Evaluates optical sharpness, contrast, and illumination suitability of an input scan.
    Prevents degraded/blurry secondary top images from generating false defect discrepancies.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return {
            "status": "NOT_VERIFIABLE",
            "quality_tier": "UNAVAILABLE",
            "reasons": ["Image array missing, None, or empty"],
            "laplacian_variance": 0.0,
            "rms_contrast": 0.0,
            "mean_brightness": 0.0,
            "is_suitable": False
        }

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    
    # Foreground segmentation (> 50 intensity to filter black scanner glass)
    fg_mask = gray > 50
    if np.sum(fg_mask) > 100:
        eval_pixels = gray[fg_mask]
        lap = cv2.Laplacian(gray, cv2.CV_64F)[fg_mask]
        lap_var = float(np.var(lap))
    else:
        eval_pixels = gray
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    rms_contrast = float(np.std(gray))
    mean_brightness = float(np.mean(eval_pixels))

    reasons = []
    if lap_var < min_laplacian_var:
        reasons.append(f"Image blur / out-of-focus (Laplacian variance {lap_var:.2f} < {min_laplacian_var:.1f})")
    if mean_brightness < min_brightness:
        reasons.append(f"Severe underexposure / platen dark (Mean brightness {mean_brightness:.1f} < {min_brightness:.1f})")
    if mean_brightness > max_brightness:
        reasons.append(f"Severe overexposure / saturation (Mean brightness {mean_brightness:.1f} > {max_brightness:.1f})")
    if rms_contrast < min_contrast:
        reasons.append(f"Low optical contrast (RMS contrast {rms_contrast:.2f} < {min_contrast:.1f})")

    is_suitable = (len(reasons) == 0)
    status = "SUITABLE" if is_suitable else "NOT_VERIFIABLE"

    return {
        "status": status,
        "quality_tier": "SUITABLE" if is_suitable else ("BLURRY" if lap_var < min_laplacian_var else "DEGRADED"),
        "reasons": reasons,
        "laplacian_variance": round(lap_var, 2),
        "rms_contrast": round(rms_contrast, 2),
        "mean_brightness": round(mean_brightness, 2),
        "is_suitable": is_suitable
    }


def dataframe_to_view_observations(
    df: Optional[pd.DataFrame],
    scanner_id: str,
    pixel_scale_mm: float = 0.08467
) -> List[ViewObservation]:
    """
    Converts a pipeline measurements DataFrame into a list of ViewObservation dataclass records.
    Safely parses missing or optional columns without raising KeyErrors.
    """
    if df is None or df.empty:
        return []

    observations: List[ViewObservation] = []
    prefix = scanner_id[0].upper()

    for idx, row in df.iterrows():
        det_id = str(row.get("grain_id", f"{scanner_id}_{idx}"))
        obs_id = f"OBS_{prefix}_{idx:05d}"

        cx = float(row.get("centroid_x", row.get("x", 0.0) + row.get("width", 0.0) / 2.0))
        cy = float(row.get("centroid_y", row.get("y", 0.0) + row.get("height", 0.0) / 2.0))
        bbox = (
            int(row.get("x", 0)),
            int(row.get("y", 0)),
            int(row.get("width", 0)),
            int(row.get("height", 0))
        )
        area_px = float(row.get("area_px", 0.0))
        len_mm = float(row.get("length_mm", row.get("length_px", 0.0) * pixel_scale_mm))
        wid_mm = float(row.get("width_mm", row.get("width_px", 0.0) * pixel_scale_mm))
        ar = float(row.get("aspect_ratio", len_mm / max(wid_mm, 0.01)))

        # Quality and defect labels
        dclass = str(row.get("classification", row.get("quality_class", "Good / Head Rice")))
        conf = float(row.get("confidence", 1.0))
        is_broken = bool(row.get("is_broken", False) or "Broken" in dclass or ar < 1.6)

        obs = ViewObservation(
            observation_id=obs_id,
            scanner_id=scanner_id,
            detection_id=det_id,
            centroid_x=cx,
            centroid_y=cy,
            bounding_box=bbox,
            area_px=area_px,
            length_mm=len_mm,
            width_mm=wid_mm,
            aspect_ratio=ar,
            defect_class=dclass,
            defect_confidence=conf,
            is_broken=is_broken,
            detection_confidence=conf,
            raw_features=row.to_dict()
        )
        observations.append(obs)

    return observations


class DualViewFusionEngine:
    """
    Fuses cross-view observations into canonical physical grains.
    Supports both Bottom-Scanner-Primary architecture and legacy Canonical Symmetric fusion.
    """

    DEFAULT_DEFECT_SEVERITY_ORDER = [
        "Foreign Matter",
        "Damaged Grains",
        "Discoloured Grains",
        "Small Brokens (<25%)",
        "Big Brokens (25-75%)",
        "Chalky Grains",
        "Good / Head Rice"
    ]

    def __init__(
        self,
        correspondence_engine: DualViewCorrespondenceEngine,
        pixel_scale_mm: float = 0.08467,  # Default for 300 DPI (25.4 / 300)
        mass_conversion_factor: Optional[float] = None,
        defect_fusion_rule: str = "surface_defect_precedence",
        defect_fusion_version: str = "1.0.0-experimental",
        defect_severity_order: Optional[List[str]] = None,
        defect_precedence_is_standard_approved: bool = False,
        architecture_mode: Optional[str] = None
    ):
        self.correspondence_engine = correspondence_engine
        self.pixel_scale_mm = pixel_scale_mm
        self.mass_conv_factor = mass_conversion_factor
        self.defect_fusion_rule = defect_fusion_rule
        self.defect_fusion_version = defect_fusion_version
        self.defect_severity_order = list(defect_severity_order) if defect_severity_order is not None else self.DEFAULT_DEFECT_SEVERITY_ORDER
        self.defect_precedence_is_standard_approved = defect_precedence_is_standard_approved

        env_mode = os.environ.get("RICEAGE_DUAL_VIEW_MODE", "").strip().lower()
        if env_mode in ("canonical_fusion", "symmetric", "experimental", "mode_b"):
            raise ValueError(
                "Symmetric dual-view fusion (Mode B) is strictly prohibited from production "
                "to prevent +78% grain count inflation and statutory non-compliance. "
                "Supported modes: 'single_view' or 'bottom_primary_verification'."
            )

        if architecture_mode is not None:
            clean = architecture_mode.strip().lower()
            if clean in ("bottom_primary_verification", "mode_c", "verification"):
                self.architecture_mode = "bottom_primary_verification"
            elif clean in ("bottom_primary", "single_view", "single"):
                self.architecture_mode = "bottom_primary"
            elif clean in ("experimental_canonical_fusion", "canonical_fusion", "symmetric"):
                # Preserved for unit test regression only; prohibited in production deployment
                self.architecture_mode = "experimental_canonical_fusion"
            else:
                self.architecture_mode = "bottom_primary"
        elif env_mode in ("bottom_primary_verification", "mode_c", "verification"):
            self.architecture_mode = "bottom_primary_verification"
        else:
            self.architecture_mode = "bottom_primary"

    def fuse_views(
        self,
        bottom_obs: List[ViewObservation],
        top_obs: List[ViewObservation],
        top_quality_status: str = "SUITABLE",
        top_quality_metrics: Optional[Dict[str, Any]] = None
    ) -> DualViewFusionResult:
        """
        Executes correspondence matching and routes to the active architectural mode.
        If top_quality_status is NOT_VERIFIABLE, top view observations cannot be used to verify bottom predictions.
        """
        if self.architecture_mode in ("bottom_primary", "bottom_primary_verification", "single_view"):
            return self._fuse_views_bottom_primary(
                bottom_obs,
                top_obs,
                top_quality_status=top_quality_status,
                top_quality_metrics=top_quality_metrics
            )
        elif self.architecture_mode == "experimental_canonical_fusion":
            return self._fuse_views_canonical_symmetric(bottom_obs, top_obs)
        else:
            raise ValueError(
                f"Architecture mode '{self.architecture_mode}' is prohibited from production. "
                "Supported modes: 'single_view' or 'bottom_primary_verification'."
            )

    def _fuse_views_bottom_primary(
        self,
        bottom_obs: List[ViewObservation],
        top_obs: List[ViewObservation],
        top_quality_status: str = "SUITABLE",
        top_quality_metrics: Optional[Dict[str, Any]] = None
    ) -> DualViewFusionResult:
        """
        Executes Bottom-Scanner-Primary Analysis.
        The bottom scanner is the single authoritative source of truth for:
          - Grain count
          - Whole/broken classification
          - Defect classification
          - Physical dimensions (length, width, area, aspect ratio)
          - Quality percentages and denominators
        Top scanner observations are used strictly for secondary verification and discrepancy flags.
        Top detections are NEVER concatenated or added to the primary grain count.
        Unresolved bottom clusters are preserved and segregated.
        If bottom scanner observations are missing or empty, returns an explicit analysis failure.
        """
        # 1. Fail-closed if bottom scanner observations are missing or empty
        if not bottom_obs:
            return DualViewFusionResult(
                physical_grains=[],
                total_physical_grains=0,
                total_single_physical_grains=0,
                total_inventory_records=0,
                matched_count=0,
                unmatched_bottom_count=0,
                unmatched_top_count=len(top_obs) if top_obs else 0,
                ambiguous_count=0,
                registration_failed_count=0,
                unresolved_cluster_count=0,
                discrepancy_count=0,
                top_image_quality_status=top_quality_status,
                top_image_quality_metrics=top_quality_metrics or {},
                architecture_mode="BOTTOM_SCANNER_PRIMARY",
                primary_scanner="bottom",
                secondary_scanner="top",
                estimated_mass_g=0.0,
                estimated_optical_mass_unvalidated_g=0.0,
                is_experimental=False,
                registration_metadata={
                    "status": "FAILED_BOTTOM_PRIMARY_MISSING",
                    "message": "Bottom primary scanner observations missing. Bottom-primary architecture strictly prohibits uncertified top-only substitution."
                },
                aggregation_summary={
                    "status": "ANALYSIS_FAILED",
                    "failure_reason": "BOTTOM_PRIMARY_SCANNER_FAILED",
                    "primary_scanner": "bottom",
                    "primary_scanner_available": False,
                    "count_basis": {"total_physical_grains": 0},
                    "mass_basis": {"total_mass_g": 0.0},
                    "stats": [],
                    "rows": [],
                    "avg_length_mm": 0.0,
                    "avg_breadth_mm": 0.0,
                    "length_breadth_ratio": 0.0
                }
            )

        # 2. Run correspondence matching to associate bottom grains with top counterparts
        match_records = self.correspondence_engine.match_observations(bottom_obs, top_obs)

        physical_grains: List[PhysicalGrain] = []
        grain_counter = 1

        matched_cnt = 0
        unmatched_b_cnt = 0
        unmatched_t_cnt = 0
        ambiguous_cnt = 0
        reg_failed_cnt = 0
        unresolved_cluster_cnt = 0
        discrepancy_cnt = 0

        # Extract unmatched top observations to record in diagnostics without adding to primary count
        unmatched_top_obs: List[ViewObservation] = []

        for m in match_records:
            status: CorrespondenceStatus = m["status"]
            b_obs: Optional[ViewObservation] = m.get("bottom_obs")
            t_obs: Optional[ViewObservation] = m.get("top_obs")
            conf: float = m.get("confidence", 0.0)
            cost_comps: dict = m.get("cost_components", {})

            # A. If record has NO bottom observation (e.g. UNMATCHED_TOP, top ambiguous, top cluster):
            # STRICT REQUIREMENT: Do NOT add unmatched top observations to primary grain count!
            if b_obs is None:
                if status == CorrespondenceStatus.UNMATCHED_TOP:
                    unmatched_t_cnt += 1
                    if t_obs:
                        unmatched_top_obs.append(t_obs)
                elif status == CorrespondenceStatus.AMBIGUOUS:
                    pass
                elif status == CorrespondenceStatus.SEGMENTATION_UNRESOLVED:
                    pass
                continue

            # B. Record corresponds to a BOTTOM observation (authoritative physical grain / cluster)
            phys_id = f"PHYS_{grain_counter:05d}"
            grain_counter += 1

            if status == CorrespondenceStatus.MATCHED and t_obs is not None:
                matched_cnt += 1
                # Bottom-Primary: Quality and dimensions come authoritatively from bottom
                b_class = b_obs.defect_class
                t_class = t_obs.defect_class
                
                is_top_verifiable = (top_quality_status == "SUITABLE")
                if is_top_verifiable:
                    is_agree = (b_class == t_class)
                    is_discrepancy = not is_agree
                    if is_discrepancy:
                        discrepancy_cnt += 1
                    verif_str = "VERIFIED_AGREEMENT" if is_agree else "DISCREPANCY_FLAG"
                    disc_detail = f"Bottom: {b_class} vs Top: {t_class}" if is_discrepancy else None
                    top_status_str = "MATCHED"
                else:
                    # Top image quality gate triggered: NOT_VERIFIABLE
                    # Task 1 STRICT REQUIREMENT: Do not classify poor image quality as a grain-classification discrepancy!
                    is_agree = False
                    is_discrepancy = False
                    verif_str = "NOT_VERIFIABLE"
                    disc_detail = "Top image quality unsuitable for optical verification (NOT_VERIFIABLE)"
                    top_status_str = "NOT_VERIFIABLE"

                mass_g = self._calculate_single_grain_mass(b_obs.area_px, b_obs.length_mm, b_obs.width_mm)

                q_attrs = {
                    "primary_scanner": "bottom",
                    "bottom_predicted_class": b_class,
                    "bottom_confidence": round(b_obs.defect_confidence, 4),
                    "top_predicted_class": t_class,
                    "top_confidence": round(t_obs.defect_confidence, 4),
                    "fused_class": b_class,  # Bottom is authoritative!
                    "defect_class": b_class, # Bottom is authoritative!
                    "defect_confidence": round(b_obs.defect_confidence, 4),
                    "defect_precedence_source": "bottom_primary_authoritative",
                    "top_match_status": top_status_str,
                    "is_verified": is_agree,
                    "discrepancy_flag": is_discrepancy,
                    "top_verification_status": verif_str,
                    "discrepancy_details": disc_detail,
                    "fusion_rule": "bottom_primary_with_top_verification",
                    "fusion_rule_version": self.defect_fusion_version,
                    "fusion_applied": True,
                    "fusion_status": "BOTTOM_PRIMARY",
                    "is_broken": b_obs.is_broken,
                    "length_mm": round(b_obs.length_mm, 2),
                    "width_mm": round(b_obs.width_mm, 2),
                    "area_px": round(b_obs.area_px, 1),
                    "aspect_ratio": round(b_obs.aspect_ratio, 2),
                    "estimated_mass_g": round(mass_g, 5),
                    "is_cluster": False,
                    "is_fused": True
                }
                prov = {
                    "matching_algorithm": "Hungarian_Bipartite",
                    "reprojection_distance_px": m.get("distance_px"),
                    "cost_components": cost_comps,
                    "architecture": "BOTTOM_SCANNER_PRIMARY",
                    "verification": verif_str
                }

            elif status == CorrespondenceStatus.UNMATCHED_BOTTOM:
                unmatched_b_cnt += 1
                mass_g = self._calculate_single_grain_mass(b_obs.area_px, b_obs.length_mm, b_obs.width_mm)
                q_attrs = {
                    "primary_scanner": "bottom",
                    "bottom_predicted_class": b_obs.defect_class,
                    "bottom_confidence": round(b_obs.defect_confidence, 4),
                    "top_predicted_class": None,
                    "top_confidence": None,
                    "fused_class": b_obs.defect_class,
                    "defect_class": b_obs.defect_class,
                    "defect_confidence": round(b_obs.defect_confidence, 4),
                    "defect_precedence_source": "bottom_primary_unmatched_top",
                    "top_match_status": "UNMATCHED_IN_TOP",
                    "top_verification_status": "UNVERIFIED_SINGLE_VIEW" if top_quality_status == "SUITABLE" else "NOT_VERIFIABLE",
                    "is_verified": False,
                    "discrepancy_flag": False,
                    "discrepancy_details": "Observed on bottom platen only (no top match)",
                    "fusion_rule": "bottom_primary_single_view",
                    "fusion_rule_version": self.defect_fusion_version,
                    "fusion_applied": False,
                    "fusion_status": "BOTTOM_PRIMARY_ONLY",
                    "is_broken": b_obs.is_broken,
                    "length_mm": round(b_obs.length_mm, 2),
                    "width_mm": round(b_obs.width_mm, 2),
                    "area_px": round(b_obs.area_px, 1),
                    "aspect_ratio": round(b_obs.aspect_ratio, 2),
                    "estimated_mass_g": round(mass_g, 5),
                    "is_cluster": False,
                    "is_fused": False
                }
                prov = {"matching_algorithm": "None", "reason": cost_comps.get("reason"), "architecture": "BOTTOM_SCANNER_PRIMARY"}

            elif status == CorrespondenceStatus.AMBIGUOUS:
                ambiguous_cnt += 1
                mass_g = self._calculate_single_grain_mass(b_obs.area_px, b_obs.length_mm, b_obs.width_mm)
                q_attrs = {
                    "primary_scanner": "bottom",
                    "bottom_predicted_class": b_obs.defect_class,
                    "bottom_confidence": round(b_obs.defect_confidence, 4),
                    "top_predicted_class": None,
                    "top_confidence": None,
                    "fused_class": b_obs.defect_class,
                    "defect_class": b_obs.defect_class,
                    "defect_confidence": round(b_obs.defect_confidence, 4),
                    "defect_precedence_source": "bottom_primary_ambiguous_top",
                    "top_match_status": "AMBIGUOUS",
                    "top_verification_status": "AMBIGUOUS",
                    "is_verified": False,
                    "discrepancy_flag": True,
                    "discrepancy_details": "Ambiguous geometric correspondence in top view",
                    "fusion_rule": "bottom_primary_ambiguous",
                    "fusion_rule_version": self.defect_fusion_version,
                    "fusion_applied": False,
                    "fusion_status": "AMBIGUOUS_TOP_VERIFICATION",
                    "is_broken": b_obs.is_broken,
                    "length_mm": round(b_obs.length_mm, 2),
                    "width_mm": round(b_obs.width_mm, 2),
                    "area_px": round(b_obs.area_px, 1),
                    "aspect_ratio": round(b_obs.aspect_ratio, 2),
                    "estimated_mass_g": round(mass_g, 5),
                    "is_cluster": False,
                    "is_fused": False
                }
                prov = {"matching_algorithm": "Ambiguity_Detection", "reason": cost_comps.get("reason"), "architecture": "BOTTOM_SCANNER_PRIMARY"}

            elif status == CorrespondenceStatus.SEGMENTATION_UNRESOLVED:
                # Bottom unresolved multi-grain cluster!
                unresolved_cluster_cnt += 1
                mass_g = self._calculate_single_grain_mass(b_obs.area_px, b_obs.length_mm, b_obs.width_mm)
                q_attrs = {
                    "primary_scanner": "bottom",
                    "bottom_predicted_class": "Unresolved Multi-Grain Cluster",
                    "bottom_confidence": 0.0,
                    "top_predicted_class": None,
                    "top_confidence": None,
                    "fused_class": "Unresolved / Pending Review",
                    "defect_class": "Unresolved / Pending Review",
                    "defect_confidence": 0.0,
                    "defect_precedence_source": "bottom_unresolved_cluster",
                    "top_match_status": "CLUSTER_SEGREGATED",
                    "top_verification_status": "CLUSTER_SEGREGATED",
                    "is_verified": False,
                    "discrepancy_flag": False,
                    "discrepancy_details": "Under-segmented touching cluster segregated from single grains",
                    "fusion_rule": "unresolved_cluster_segregation",
                    "fusion_rule_version": self.defect_fusion_version,
                    "fusion_applied": False,
                    "fusion_status": "UNRESOLVED_CLUSTER",
                    "is_broken": False,
                    "length_mm": round(b_obs.length_mm, 2),
                    "width_mm": round(b_obs.width_mm, 2),
                    "area_px": round(b_obs.area_px, 1),
                    "aspect_ratio": round(b_obs.aspect_ratio, 2),
                    "estimated_mass_g": round(mass_g, 5),
                    "is_cluster": True,
                    "separation_status": "under_segmented_cluster",
                    "is_fused": False
                }
                prov = {"matching_algorithm": "Cluster_Segregation", "reason": "Cluster area exceeds single grain threshold", "architecture": "BOTTOM_SCANNER_PRIMARY"}

            elif status == CorrespondenceStatus.REGISTRATION_FAILED:
                reg_failed_cnt += 1
                mass_g = self._calculate_single_grain_mass(b_obs.area_px, b_obs.length_mm, b_obs.width_mm)
                q_attrs = {
                    "primary_scanner": "bottom",
                    "bottom_predicted_class": b_obs.defect_class,
                    "bottom_confidence": round(b_obs.defect_confidence, 4),
                    "top_predicted_class": None,
                    "top_confidence": None,
                    "fused_class": b_obs.defect_class,
                    "defect_class": b_obs.defect_class,
                    "defect_confidence": round(b_obs.defect_confidence, 4),
                    "defect_precedence_source": "bottom_primary_registration_failed",
                    "top_match_status": "REGISTRATION_FAILED",
                    "top_verification_status": "REGISTRATION_FAILED",
                    "is_verified": False,
                    "discrepancy_flag": False,
                    "discrepancy_details": "Registration health check failed; secondary verification withheld",
                    "fusion_rule": "fail_closed_registration",
                    "fusion_rule_version": self.defect_fusion_version,
                    "fusion_applied": False,
                    "fusion_status": "FAIL_CLOSED",
                    "is_broken": b_obs.is_broken,
                    "length_mm": round(b_obs.length_mm, 2),
                    "width_mm": round(b_obs.width_mm, 2),
                    "area_px": round(b_obs.area_px, 1),
                    "aspect_ratio": round(b_obs.aspect_ratio, 2),
                    "estimated_mass_g": round(mass_g, 5),
                    "is_cluster": False,
                    "is_fused": False
                }
                prov = {"matching_algorithm": "Fail_Closed", "reason": cost_comps.get("reason"), "architecture": "BOTTOM_SCANNER_PRIMARY"}

            else:
                mass_g = self._calculate_single_grain_mass(b_obs.area_px, b_obs.length_mm, b_obs.width_mm)
                q_attrs = self._extract_single_view_quality(b_obs, view="bottom")
                prov = {"matching_algorithm": "Unknown"}

            phys_grain = PhysicalGrain(
                physical_grain_id=phys_id,
                correspondence_status=status,
                correspondence_confidence=conf,
                bottom_observation_id=b_obs.observation_id if b_obs else None,
                top_observation_id=t_obs.observation_id if t_obs else None,
                bottom_obs=b_obs,
                top_obs=t_obs,
                quality_attributes=q_attrs,
                provenance=prov
            )
            physical_grains.append(phys_grain)

        # 3. Aggregate Sample Metrics strictly across canonical bottom single grains
        # Separate single grains from clusters
        single_grains = [g for g in physical_grains if not g.quality_attributes.get("is_cluster", False)]
        cluster_grains = [g for g in physical_grains if g.quality_attributes.get("is_cluster", False)]

        # Sample aggregation on single grains only
        agg_summary, total_mass_g = self._aggregate_physical_sample(single_grains)

        total_single_physical_grains = len(single_grains)
        total_inventory_records = len(physical_grains)

        # Calculate cross-view count agreement metrics
        n_bottom = len(bottom_obs)
        n_top = len(top_obs)
        diff = abs(n_bottom - n_top)
        max_n = max(n_bottom, n_top, 1)
        count_agreement_pct = round(100.0 * (1.0 - (diff / max_n)), 2) if (n_bottom > 0 and n_top > 0) else 0.0
        if (diff / max_n) <= 0.03:
            count_tolerance_status = "VERIFIED_EXCELLENT"
        elif (diff / max_n) <= 0.07:
            count_tolerance_status = "ADVISORY_DISCREPANCY"
        else:
            count_tolerance_status = "HIGH_DISCREPANCY"

        agg_summary["count_agreement_pct"] = count_agreement_pct
        agg_summary["count_tolerance_status"] = count_tolerance_status
        agg_summary["bottom_grain_count"] = n_bottom
        agg_summary["top_grain_count"] = n_top

        return DualViewFusionResult(
            physical_grains=physical_grains,
            total_physical_grains=total_single_physical_grains,
            total_single_physical_grains=total_single_physical_grains,
            total_inventory_records=total_inventory_records,
            matched_count=matched_cnt,
            unmatched_bottom_count=unmatched_b_cnt,
            unmatched_top_count=unmatched_t_cnt,
            ambiguous_count=ambiguous_cnt,
            registration_failed_count=reg_failed_cnt,
            unresolved_cluster_count=unresolved_cluster_cnt,
            discrepancy_count=discrepancy_cnt,
            count_agreement_pct=count_agreement_pct,
            count_tolerance_status=count_tolerance_status,
            bottom_grain_count=n_bottom,
            top_grain_count=n_top,
            top_image_quality_status=top_quality_status,
            top_image_quality_metrics=top_quality_metrics or {},
            architecture_mode="BOTTOM_SCANNER_PRIMARY",
            primary_scanner="bottom",
            secondary_scanner="top",
            estimated_mass_g=total_mass_g,
            estimated_optical_mass_unvalidated_g=total_mass_g,
            mass_measurement_status="NOT_PHYSICALLY_MEASURED",
            is_physically_measured=False,
            statutory_acceptance_usable=False,
            is_experimental=False,  # Bottom-primary is the official production delivery mode
            registration_metadata=self.correspondence_engine.registration.get_health_report(),
            aggregation_summary=agg_summary
        )

    def _fuse_views_canonical_symmetric(
        self,
        bottom_obs: List[ViewObservation],
        top_obs: List[ViewObservation]
    ) -> DualViewFusionResult:
        """
        Executes Experimental Symmetric Canonical Fusion (Phase 4 / Phase 4.1 Prototype).
        Preserved behind feature flag for research and experimental benchmarking.
        """
        match_records = self.correspondence_engine.match_observations(bottom_obs, top_obs)

        physical_grains: List[PhysicalGrain] = []
        grain_counter = 1

        matched_cnt = 0
        unmatched_b_cnt = 0
        unmatched_t_cnt = 0
        ambiguous_cnt = 0
        reg_failed_cnt = 0
        unresolved_cluster_cnt = 0

        for m in match_records:
            status: CorrespondenceStatus = m["status"]
            b_obs: Optional[ViewObservation] = m.get("bottom_obs")
            t_obs: Optional[ViewObservation] = m.get("top_obs")
            conf: float = m.get("confidence", 0.0)
            cost_comps: dict = m.get("cost_components", {})

            phys_id = f"PHYS_{grain_counter:05d}"
            grain_counter += 1

            if status == CorrespondenceStatus.MATCHED:
                matched_cnt += 1
                q_attrs = self._fuse_matched_quality(b_obs, t_obs)
                prov = {
                    "matching_algorithm": "Hungarian_Bipartite",
                    "reprojection_distance_px": m.get("distance_px"),
                    "cost_components": cost_comps,
                    "fusion_policy": self.defect_fusion_rule
                }
            elif status == CorrespondenceStatus.UNMATCHED_BOTTOM:
                unmatched_b_cnt += 1
                q_attrs = self._extract_single_view_quality(b_obs, view="bottom")
                prov = {"matching_algorithm": "None", "reason": cost_comps.get("reason")}
            elif status == CorrespondenceStatus.UNMATCHED_TOP:
                unmatched_t_cnt += 1
                q_attrs = self._extract_single_view_quality(t_obs, view="top")
                prov = {"matching_algorithm": "None", "reason": cost_comps.get("reason")}
            elif status == CorrespondenceStatus.AMBIGUOUS:
                ambiguous_cnt += 1
                obs = b_obs if b_obs is not None else t_obs
                q_attrs = self._handle_unfused_record(obs, status=status, reason=cost_comps.get("reason"))
                prov = {"matching_algorithm": "Ambiguity_Detection", "reason": cost_comps.get("reason")}
            elif status == CorrespondenceStatus.REGISTRATION_FAILED:
                reg_failed_cnt += 1
                obs = b_obs if b_obs is not None else t_obs
                q_attrs = self._handle_unfused_record(obs, status=status, reason=cost_comps.get("reason"))
                prov = {"matching_algorithm": "Fail_Closed", "reason": cost_comps.get("reason")}
            elif status == CorrespondenceStatus.SEGMENTATION_UNRESOLVED:
                unresolved_cluster_cnt += 1
                obs = b_obs if b_obs is not None else t_obs
                q_attrs = self._handle_unfused_record(obs, status=status, reason=cost_comps.get("reason"))
                prov = {"matching_algorithm": "Cluster_Segregation", "reason": cost_comps.get("reason")}
            else:
                obs = b_obs if b_obs is not None else t_obs
                q_attrs = self._handle_unfused_record(obs, status=status, reason="Unknown")
                prov = {"matching_algorithm": "Unknown"}

            phys_grain = PhysicalGrain(
                physical_grain_id=phys_id,
                correspondence_status=status,
                correspondence_confidence=conf,
                bottom_observation_id=b_obs.observation_id if b_obs else None,
                top_observation_id=t_obs.observation_id if t_obs else None,
                bottom_obs=b_obs,
                top_obs=t_obs,
                quality_attributes=q_attrs,
                provenance=prov
            )
            physical_grains.append(phys_grain)

        # Aggregate sample-level metrics across canonical physical grains
        agg_summary, total_mass_g = self._aggregate_physical_sample(physical_grains)

        # Reconcile separate counts: single grains vs clusters vs total inventory records
        total_single_physical_grains = matched_cnt + unmatched_b_cnt + unmatched_t_cnt + ambiguous_cnt
        total_inventory_records = len(physical_grains)

        return DualViewFusionResult(
            physical_grains=physical_grains,
            total_physical_grains=total_single_physical_grains,
            total_single_physical_grains=total_single_physical_grains,
            total_inventory_records=total_inventory_records,
            matched_count=matched_cnt,
            unmatched_bottom_count=unmatched_b_cnt,
            unmatched_top_count=unmatched_t_cnt,
            ambiguous_count=ambiguous_cnt,
            registration_failed_count=reg_failed_cnt,
            unresolved_cluster_count=unresolved_cluster_cnt,
            discrepancy_count=0,
            architecture_mode="EXPERIMENTAL_CANONICAL_FUSION",
            primary_scanner="dual_symmetric",
            secondary_scanner="none",
            estimated_mass_g=total_mass_g,
            estimated_optical_mass_unvalidated_g=total_mass_g,
            mass_measurement_status="NOT_PHYSICALLY_MEASURED",
            is_physically_measured=False,
            statutory_acceptance_usable=False,
            is_experimental=True,
            registration_metadata=self.correspondence_engine.registration.get_health_report(),
            aggregation_summary=agg_summary
        )

    def _fuse_matched_quality(self, b_obs: ViewObservation, t_obs: ViewObservation) -> Dict[str, Any]:
        """
        Fuses quality attributes for a confirmed one-to-one matched grain pair.
        Implements configurable defect precedence with full provenance and traceability.
        """
        b_class = b_obs.defect_class
        t_class = t_obs.defect_class

        if b_class == t_class:
            fused_class = b_class
            fused_conf = float(np.mean([b_obs.defect_confidence, t_obs.defect_confidence]))
            defect_source = "both_views_agree"
        else:
            b_rank = self.defect_severity_order.index(b_class) if b_class in self.defect_severity_order else 99
            t_rank = self.defect_severity_order.index(t_class) if t_class in self.defect_severity_order else 99
            if b_rank <= t_rank:
                fused_class = b_class
                fused_conf = b_obs.defect_confidence
                defect_source = "bottom_view_defect_precedence"
            else:
                fused_class = t_class
                fused_conf = t_obs.defect_confidence
                defect_source = "top_view_defect_precedence"

        # Geometric attributes
        fused_len_mm = float(np.mean([b_obs.length_mm, t_obs.length_mm]))
        fused_wid_mm = float(np.mean([b_obs.width_mm, t_obs.width_mm]))
        fused_area_px = float(np.mean([b_obs.area_px, t_obs.area_px]))
        fused_aspect_ratio = round(fused_len_mm / max(fused_wid_mm, 0.01), 2)
        is_broken = bool(b_obs.is_broken or t_obs.is_broken or "Broken" in fused_class)

        # Single mass calculation (uncalibrated optical proxy)
        mass_g = self._calculate_single_grain_mass(fused_area_px, fused_len_mm, fused_wid_mm)

        return {
            "bottom_predicted_class": b_class,
            "bottom_confidence": round(b_obs.defect_confidence, 4),
            "top_predicted_class": t_class,
            "top_confidence": round(t_obs.defect_confidence, 4),
            "fused_class": fused_class,
            "defect_class": fused_class,
            "defect_confidence": round(fused_conf, 4),
            "defect_precedence_source": defect_source,
            "fusion_rule": self.defect_fusion_rule,
            "fusion_rule_version": self.defect_fusion_version,
            "fusion_applied": True,
            "fusion_status": "AUTOMATIC",
            "defect_precedence_is_standard_approved": self.defect_precedence_is_standard_approved,
            "precedence_standard_notice": (
                "Approved statutory standard" if self.defect_precedence_is_standard_approved
                else "Experimental engineering heuristic; not an approved statutory APSCSCL standard"
            ),
            "is_broken": is_broken,
            "length_mm": round(fused_len_mm, 2),
            "width_mm": round(fused_wid_mm, 2),
            "area_px": round(fused_area_px, 1),
            "aspect_ratio": fused_aspect_ratio,
            "estimated_mass_g": round(mass_g, 5),
            "is_fused": True
        }

    def _extract_single_view_quality(self, obs: ViewObservation, view: str) -> Dict[str, Any]:
        """Extracts quality attributes for an unmatched grain observed on one platen only."""
        mass_g = self._calculate_single_grain_mass(obs.area_px, obs.length_mm, obs.width_mm)
        return {
            "bottom_predicted_class": obs.defect_class if view == "bottom" else None,
            "bottom_confidence": round(obs.defect_confidence, 4) if view == "bottom" else None,
            "top_predicted_class": obs.defect_class if view == "top" else None,
            "top_confidence": round(obs.defect_confidence, 4) if view == "top" else None,
            "fused_class": obs.defect_class,
            "defect_class": obs.defect_class,
            "defect_confidence": round(obs.defect_confidence, 4),
            "defect_precedence_source": f"single_view_{view}",
            "fusion_rule": "none_single_view",
            "fusion_rule_version": self.defect_fusion_version,
            "fusion_applied": False,
            "fusion_status": "WITHHELD",
            "is_broken": obs.is_broken,
            "length_mm": round(obs.length_mm, 2),
            "width_mm": round(obs.width_mm, 2),
            "area_px": round(obs.area_px, 1),
            "aspect_ratio": round(obs.aspect_ratio, 2),
            "estimated_mass_g": round(mass_g, 5),
            "is_fused": False,
            "view_present": view
        }

    def _handle_unfused_record(
        self,
        obs: Optional[ViewObservation],
        status: CorrespondenceStatus,
        reason: Optional[str]
    ) -> Dict[str, Any]:
        """Preserves record with automatic fusion strictly disabled for ambiguous/unresolved/failed cases."""
        area = obs.area_px if obs else 0.0
        len_mm = obs.length_mm if obs else 0.0
        wid_mm = obs.width_mm if obs else 0.0
        mass_g = self._calculate_single_grain_mass(area, len_mm, wid_mm) if obs else 0.0

        is_bottom = bool(obs and obs.scanner_id == "bottom")
        is_top = bool(obs and obs.scanner_id == "top")

        return {
            "bottom_predicted_class": obs.defect_class if is_bottom else None,
            "bottom_confidence": round(obs.defect_confidence, 4) if is_bottom else None,
            "top_predicted_class": obs.defect_class if is_top else None,
            "top_confidence": round(obs.defect_confidence, 4) if is_top else None,
            "fused_class": "Unresolved / Pending Review",
            "defect_class": "Unresolved / Pending Review",
            "defect_confidence": 0.0,
            "defect_precedence_source": "none_unfused_safe_fallback",
            "fusion_rule": "safe_segregation",
            "fusion_rule_version": self.defect_fusion_version,
            "fusion_applied": False,
            "fusion_status": "WITHHELD",
            "unfused_reason": reason or status.value,
            "is_broken": False,
            "length_mm": round(len_mm, 2),
            "width_mm": round(wid_mm, 2),
            "area_px": round(area, 1),
            "aspect_ratio": round(len_mm / max(wid_mm, 0.01), 2) if wid_mm > 0 else 0.0,
            "estimated_mass_g": round(mass_g, 5),
            "is_fused": False,
            "is_cluster": status == CorrespondenceStatus.SEGMENTATION_UNRESOLVED,
            "separation_status": obs.raw_features.get("separation_status", "under_segmented_cluster" if status == CorrespondenceStatus.SEGMENTATION_UNRESOLVED else "single_grain") if obs else "unknown"
        }

    def _calculate_single_grain_mass(self, area_px: float, length_mm: float, width_mm: float) -> float:
        """Estimates single grain mass. Guaranteed never to double-count."""
        if self.mass_conv_factor is not None:
            return float(area_px * self.mass_conv_factor)
        if length_mm > 0.0 and width_mm > 0.0:
            thickness_mm = 0.70 * width_mm
            vol_mm3 = (4.0 / 3.0) * np.pi * (length_mm / 2.0) * (width_mm / 2.0) * (thickness_mm / 2.0)
            density_g_mm3 = 0.00142  # ~1.42 g/cm3 standard rice endosperm density
            return float(vol_mm3 * density_g_mm3)
        else:
            return float(area_px * (self.pixel_scale_mm ** 2) * 0.00025)

    def _aggregate_physical_sample(self, grains: List[PhysicalGrain]) -> Tuple[Dict[str, Any], float]:
        """
        Executes sample-level aggregation strictly on canonical PhysicalGrain records.
        Each physical grain contributes exactly once to count and mass totals.
        """
        total_physical_count = len(grains)
        total_sample_mass_g = 0.0

        category_counts = {
            "Good / Head Rice": 0,
            "Big Brokens (25-75%)": 0,
            "Small Brokens (<25%)": 0,
            "Foreign Matter": 0,
            "Damaged Grains": 0,
            "Discoloured Grains": 0,
            "Chalky Grains": 0,
            "Unresolved / Pending Review": 0
        }
        category_masses = {k: 0.0 for k in category_counts}

        lengths, breadths = [], []

        CLASS_MAP = {
            "NORMAL_RICE": "Good / Head Rice",
            "WHOLE_RICE": "Good / Head Rice",
            "WHOLE": "Good / Head Rice",
            "WHOLE_GRAIN": "Good / Head Rice",
            "GOOD_GRAIN": "Good / Head Rice",
            "BROKEN_RICE": "Big Brokens (25-75%)",
            "BROKEN": "Big Brokens (25-75%)",
            "BROKEN_GRAIN": "Big Brokens (25-75%)",
            "SMALL_BROKEN": "Small Brokens (<25%)",
            "SMALL_BROKEN_RICE": "Small Brokens (<25%)",
            "DAMAGED_RICE": "Damaged Grains",
            "DAMAGED": "Damaged Grains",
            "DAMAGED_GRAIN": "Damaged Grains",
            "DISCOLORED_RICE": "Discoloured Grains",
            "DISCOLOURED_RICE": "Discoloured Grains",
            "DISCOLORED": "Discoloured Grains",
            "DISCOLOURED": "Discoloured Grains",
            "DISCOLORED_GRAIN": "Discoloured Grains",
            "CHALKY_RICE": "Chalky Grains",
            "CHALKY": "Chalky Grains",
            "CHALKY_GRAIN": "Chalky Grains",
            "FOREIGN_MATTER": "Foreign Matter",
            "FOREIGN_OBJECT": "Foreign Matter",
            "BOUNDARY_TRUNCATED": "Good / Head Rice",
        }

        for g in grains:
            qa = g.quality_attributes
            mass = qa.get("estimated_mass_g", 0.0)
            dclass = qa.get("defect_class", "Unresolved / Pending Review")
            mapped_class = CLASS_MAP.get(str(dclass).upper().strip(), dclass)

            total_sample_mass_g += mass

            if mapped_class in category_counts:
                category_counts[mapped_class] += 1
                category_masses[mapped_class] += mass
            else:
                category_counts["Unresolved / Pending Review"] += 1
                category_masses["Unresolved / Pending Review"] += mass

            if qa.get("length_mm", 0.0) > 0.0:
                lengths.append(qa["length_mm"])
            if qa.get("width_mm", 0.0) > 0.0:
                breadths.append(qa["width_mm"])

        avg_len = float(np.mean(lengths)) if lengths else 0.0
        avg_brd = float(np.mean(breadths)) if breadths else 0.0

        denom_cnt = max(total_physical_count, 1)
        denom_mass = max(total_sample_mass_g, 1e-6)

        count_basis = {
            "total_physical_grains": total_physical_count,
            "whole_count": category_counts["Good / Head Rice"],
            "whole_pct": round(category_counts["Good / Head Rice"] / denom_cnt * 100, 2),
            "broken_count": category_counts["Big Brokens (25-75%)"],
            "broken_pct": round(category_counts["Big Brokens (25-75%)"] / denom_cnt * 100, 2),
            "small_broken_count": category_counts["Small Brokens (<25%)"],
            "small_broken_pct": round(category_counts["Small Brokens (<25%)"] / denom_cnt * 100, 2),
            "foreign_matter_count": category_counts["Foreign Matter"],
            "foreign_matter_pct": round(category_counts["Foreign Matter"] / denom_cnt * 100, 2),
            "damaged_count": category_counts["Damaged Grains"],
            "damaged_pct": round(category_counts["Damaged Grains"] / denom_cnt * 100, 2),
            "discoloured_count": category_counts["Discoloured Grains"],
            "discoloured_pct": round(category_counts["Discoloured Grains"] / denom_cnt * 100, 2),
            "chalky_count": category_counts["Chalky Grains"],
            "chalky_pct": round(category_counts["Chalky Grains"] / denom_cnt * 100, 2),
            "unresolved_count": category_counts["Unresolved / Pending Review"],
            "unresolved_pct": round(category_counts["Unresolved / Pending Review"] / denom_cnt * 100, 2),
            "total_broken_count": category_counts["Big Brokens (25-75%)"] + category_counts["Small Brokens (<25%)"],
            "total_broken_pct": round((category_counts["Big Brokens (25-75%)"] + category_counts["Small Brokens (<25%)"]) / denom_cnt * 100, 2),
        }

        mass_basis = {
            "total_sample_mass_g": round(total_sample_mass_g, 4),
            "whole_mass_g": round(category_masses["Good / Head Rice"], 4),
            "whole_pct": round(category_masses["Good / Head Rice"] / denom_mass * 100, 2),
            "broken_mass_g": round(category_masses["Big Brokens (25-75%)"], 4),
            "broken_pct": round(category_masses["Big Brokens (25-75%)"] / denom_mass * 100, 2),
            "small_broken_mass_g": round(category_masses["Small Brokens (<25%)"], 4),
            "small_broken_pct": round(category_masses["Small Brokens (<25%)"] / denom_mass * 100, 2),
            "foreign_matter_mass_g": round(category_masses["Foreign Matter"], 4),
            "foreign_matter_pct": round(category_masses["Foreign Matter"] / denom_mass * 100, 2),
            "damaged_mass_g": round(category_masses["Damaged Grains"], 4),
            "damaged_pct": round(category_masses["Damaged Grains"] / denom_mass * 100, 2),
            "discoloured_mass_g": round(category_masses["Discoloured Grains"], 4),
            "discoloured_pct": round(category_masses["Discoloured Grains"] / denom_mass * 100, 2),
            "chalky_mass_g": round(category_masses["Chalky Grains"], 4),
            "chalky_pct": round(category_masses["Chalky Grains"] / denom_mass * 100, 2),
            "unresolved_mass_g": round(category_masses["Unresolved / Pending Review"], 4),
            "unresolved_pct": round(category_masses["Unresolved / Pending Review"] / denom_mass * 100, 2),
            "total_broken_mass_g": round(category_masses["Big Brokens (25-75%)"] + category_masses["Small Brokens (<25%)"], 4),
            "total_broken_pct": round((category_masses["Big Brokens (25-75%)"] + category_masses["Small Brokens (<25%)"]) / denom_mass * 100, 2),
            "mass_measurement_status": "NOT_PHYSICALLY_MEASURED",
            "is_physically_measured": False,
            "statutory_acceptance_usable": False,
            "statutory_mass_disclaimer": (
                "Optical mass is an uncalibrated geometric proxy. Prohibited from statutory "
                "procurement, tender acceptance, or legal trade decisions without physical balance verification."
            )
        }

        stats = [
            {"param": "Good / Head Rice", "count": count_basis["whole_count"], "pct": count_basis["whole_pct"], "mass_g": mass_basis["whole_mass_g"], "mass_pct": mass_basis["whole_pct"], "limit": "Min 75.00%"},
            {"param": "Big Brokens (25-75%)", "count": count_basis["broken_count"], "pct": count_basis["broken_pct"], "mass_g": mass_basis["broken_mass_g"], "mass_pct": mass_basis["broken_pct"], "limit": "Max 15.00%"},
            {"param": "Small Brokens (<25%)", "count": count_basis["small_broken_count"], "pct": count_basis["small_broken_pct"], "mass_g": mass_basis["small_broken_mass_g"], "mass_pct": mass_basis["small_broken_pct"], "limit": "Max 1.00%"},
            {"param": "Foreign Matter", "count": count_basis["foreign_matter_count"], "pct": count_basis["foreign_matter_pct"], "mass_g": mass_basis["foreign_matter_mass_g"], "mass_pct": mass_basis["foreign_matter_pct"], "limit": "Max 0.50%"},
            {"param": "Damaged Grains", "count": count_basis["damaged_count"], "pct": count_basis["damaged_pct"], "mass_g": mass_basis["damaged_mass_g"], "mass_pct": mass_basis["damaged_pct"], "limit": "Max 2.00%"},
            {"param": "Discoloured Grains", "count": count_basis["discoloured_count"], "pct": count_basis["discoloured_pct"], "mass_g": mass_basis["discoloured_mass_g"], "mass_pct": mass_basis["discoloured_pct"], "limit": "Max 1.00%"},
            {"param": "Chalky Grains", "count": count_basis["chalky_count"], "pct": count_basis["chalky_pct"], "mass_g": mass_basis["chalky_mass_g"], "mass_pct": mass_basis["chalky_pct"], "limit": "Max 3.00%"},
        ]

        if category_counts["Unresolved / Pending Review"] > 0:
            stats.append({
                "param": "Unresolved Clusters / Review Pending",
                "count": count_basis["unresolved_count"],
                "pct": count_basis["unresolved_pct"],
                "mass_g": mass_basis["unresolved_mass_g"],
                "mass_pct": mass_basis["unresolved_pct"],
                "limit": "Max 0.00% (Informational)"
            })

        summary = {
            "count_basis": count_basis,
            "mass_basis": mass_basis,
            "stats": stats,
            "avg_length_mm": round(avg_len, 2),
            "avg_breadth_mm": round(avg_brd, 2),
            "length_breadth_ratio": round(avg_len / max(avg_brd, 0.01), 2),
            "total_physical_count": total_physical_count,
            "total_mass_g": round(total_sample_mass_g, 4)
        }

        return summary, total_sample_mass_g
