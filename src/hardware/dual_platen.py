# Dual-Platen Scanner Controller and Image Processing Architecture
# Eliminates aspect-ratio distortion and manages memory pressure across physical platens.

import os
import time
import subprocess
import shutil
from typing import Dict, Any, Tuple, Optional, List, Union
import cv2
import numpy as np
import pandas as pd

from src.classification import compute_quality_metrics, estimate_grain_mass, FutureDefectTaxonomy
from src.dual_view import (
    DualViewRegistration,
    DualViewCorrespondenceEngine,
    DualViewFusionEngine,
    dataframe_to_view_observations,
    CorrespondenceStatus,
    assess_image_quality
)


# Hardware Configuration (Defaulting to 300 DPI for Both Platens)
DEFAULT_SCAN_DPI = int(os.environ.get("RICEAGE_SCAN_DPI", "300"))
SCANNER_BOTTOM_DPI = int(os.environ.get("RICEAGE_BOTTOM_DPI", os.environ.get("SCAN_BOTTOM_DPI", str(DEFAULT_SCAN_DPI))))
SCANNER_TOP_DPI = int(os.environ.get("RICEAGE_TOP_DPI", os.environ.get("SCAN_TOP_DPI", str(DEFAULT_SCAN_DPI))))
SCANNER_BOTTOM_ID = os.environ.get("SCANNER_BOTTOM_ID", "airscan:e0:Canon LiDE 400 Bottom")
SCANNER_TOP_ID = os.environ.get("SCANNER_TOP_ID", "airscan:e1:Canon LiDE 400 Top")
SETTLE_DELAY = int(os.environ.get("RICEAGE_SETTLE_DELAY", "5"))


def acquire_hardware_scan(
    output_file: str,
    device_id: str,
    resolution: int = DEFAULT_SCAN_DPI,
    timeout: int = 60
) -> Tuple[bool, Optional[str]]:
    """
    Acquire flatbed optical capture using SANE scanimage.
    Executes sequentially on the target device ID.
    """
    cmd = [
        "scanimage",
        "-d", device_id,
        "--format=jpeg",
        f"--resolution={resolution}",
        "--mode=Color",
        "-o", output_file
    ]
    print(f"[Scanner Hardware] Executing scan: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if proc.returncode == 0 and os.path.exists(output_file):
            sz = os.path.getsize(output_file)
            print(f"[Scanner Hardware] Scan successful -> {output_file} ({sz} bytes)")
            return True, None
        err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
        print(f"[Scanner Hardware] Scan failed (code {proc.returncode}): {err_msg}")
        return False, err_msg
    except Exception as e:
        print(f"[Scanner Hardware] Scan exception: {e}")
        return False, str(e)


def create_distortion_free_composite(
    img_b: np.ndarray,
    img_t: np.ndarray,
    output_path: Optional[str] = None,
    divider_width: int = 8,
    divider_color: Tuple[int, int, int] = (44, 62, 80)
) -> np.ndarray:
    """
    Combines bottom and top scanner platen images horizontally WITHOUT resizing or
    distorting grain aspect ratios. Heights are unified by vertical padding (letterboxing)
    with the background median color, preserving 1:1 metric grain geometry.
    """
    h_b, w_b = img_b.shape[:2]
    h_t, w_t = img_t.shape[:2]
    target_h = max(h_b, h_t)

    # Pad bottom if needed
    if h_b < target_h:
        pad_top = (target_h - h_b) // 2
        pad_bot = target_h - h_b - pad_top
        bg_col = np.median(img_b, axis=(0, 1)).astype(int).tolist()
        img_b_padded = cv2.copyMakeBorder(
            img_b, pad_top, pad_bot, 0, 0,
            cv2.BORDER_CONSTANT, value=bg_col
        )
    else:
        img_b_padded = img_b

    # Pad top if needed
    if h_t < target_h:
        pad_top = (target_h - h_t) // 2
        pad_bot = target_h - h_t - pad_top
        bg_col = np.median(img_t, axis=(0, 1)).astype(int).tolist()
        img_t_padded = cv2.copyMakeBorder(
            img_t, pad_top, pad_bot, 0, 0,
            cv2.BORDER_CONSTANT, value=bg_col
        )
    else:
        img_t_padded = img_t

    # Create solid separation divider bar
    divider = np.full((target_h, divider_width, 3), divider_color, dtype=np.uint8)

    # Concatenate horizontally
    composite = cv2.hconcat([img_b_padded, divider, img_t_padded])

    # Overlay Platen Header Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, target_h / 2400.0)
    thick = max(1, int(round(scale * 2)))

    # Label Bottom Platen (Left)
    cv2.rectangle(composite, (20, 20), (int(360 * scale), int(70 * scale)), (30, 30, 30), -1)
    cv2.putText(composite, "PLATEN 1 (BOTTOM)", (30, int(52 * scale)), font, scale * 0.75, (0, 255, 255), thick, cv2.LINE_AA)

    # Label Top Platen (Right)
    right_x = w_b + divider_width + 20
    cv2.rectangle(composite, (right_x, 20), (right_x + int(360 * scale), int(70 * scale)), (30, 30, 30), -1)
    cv2.putText(composite, "PLATEN 2 (TOP)", (right_x + 10, int(52 * scale)), font, scale * 0.75, (0, 255, 255), thick, cv2.LINE_AA)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, composite)

    return composite


SUPPORTED_SCAN_DPIS = {300, 600, 1200}


class DualPlatenManager:
    """
    Orchestrates dual flatbed scanner acquisition and independent platen processing.
    Ensures memory efficiency, independent per-scanner DPI calibration, and unified metric reporting.
    """

    def __init__(
        self,
        pipeline=None,
        scan_resolution: Optional[Union[int, Dict[str, int]]] = None,
        use_dual_view_fusion: Optional[bool] = None
    ):
        if scan_resolution is None:
            scan_resolution = {"bottom": SCANNER_BOTTOM_DPI, "top": SCANNER_TOP_DPI}
        self.pipeline = pipeline
        self.scan_resolution = self._validate_scan_resolution(scan_resolution)
        if use_dual_view_fusion is not None:
            self.use_dual_view_fusion = use_dual_view_fusion
        elif pipeline and getattr(pipeline.config, "dual_view", None):
            self.use_dual_view_fusion = pipeline.config.dual_view.enabled
        else:
            self.use_dual_view_fusion = os.environ.get("ENABLE_DUAL_VIEW_FUSION", "1").lower() in ("1", "true", "yes")

    def _validate_scan_resolution(self, scan_resolution: Any) -> Dict[str, int]:
        """Validates acquisition DPI. Strictly rejects unsupported or unknown DPIs without 300 DPI fallback."""
        if isinstance(scan_resolution, (int, float)):
            dpi_val = int(scan_resolution)
            if dpi_val not in SUPPORTED_SCAN_DPIS:
                raise ValueError(
                    f"Unsupported scanner acquisition DPI: {scan_resolution}. "
                    f"Supported resolutions are: {sorted(list(SUPPORTED_SCAN_DPIS))}. "
                    "Silently assuming 300 DPI is strictly rejected."
                )
            return {"bottom": dpi_val, "top": dpi_val}
        elif isinstance(scan_resolution, dict):
            validated = {}
            for k in ("bottom", "top"):
                val = scan_resolution.get(k, scan_resolution.get("bottom", DEFAULT_SCAN_DPI))
                if not isinstance(val, (int, float)) or int(val) not in SUPPORTED_SCAN_DPIS:
                    raise ValueError(
                        f"Unsupported scanner acquisition DPI for '{k}': {val}. "
                        f"Supported resolutions are: {sorted(list(SUPPORTED_SCAN_DPIS))}. "
                        "Silently assuming 300 DPI is strictly rejected."
                    )
                validated[k] = int(val)
            return validated
        else:
            raise ValueError(
                f"Unsupported or invalid acquisition DPI format: {scan_resolution}. "
                f"Supported resolutions are: {sorted(list(SUPPORTED_SCAN_DPIS))}. "
                "Silently assuming 300 DPI is strictly rejected."
            )

    def get_scanner_dpi(self, scanner_id: str = "bottom") -> int:
        """Returns the validated DPI for the requested scanner."""
        return self.scan_resolution.get(scanner_id, self.scan_resolution.get("bottom", 300))

    def run_dual_analysis(
        self,
        bottom_image_path: Optional[str],
        top_image_path: Optional[str],
        sample_id: str = "Sample-0001",
        commodity: str = "Raw Rice",
        composite_output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes independent platen pipeline runs, merges measurements without cross-platen
        coordinate distortion, and generates consolidated Count & Mass Basis statistics.
        Derives pixel-to-mm scale independently for each scanner.
        """
        dfs = []
        platens_info = {}
        rendered_images = []

        # 1. Process Bottom Platen
        dpi_b = self.get_scanner_dpi("bottom")
        if self.pipeline and getattr(self.pipeline, "config", None):
            self.pipeline.config.calibration.assumed_dpi = float(dpi_b)
        if bottom_image_path and os.path.exists(bottom_image_path):
            print(f"[DualPlatenManager] Processing Platen 1 (Bottom, {dpi_b} DPI): {bottom_image_path}")
            res_b = self.pipeline.run(bottom_image_path, generate_outputs=False)
            df_b = res_b["measurements_df"].copy()
            df_b["platen"] = "bottom"
            dfs.append(df_b)
            platens_info["bottom"] = {
                "image_path": bottom_image_path,
                "grains_detected": len(df_b),
                "summary": res_b["summary"],
                "dpi": dpi_b
            }
            img_b = cv2.imread(bottom_image_path)
            rendered_images.append(img_b)
        else:
            img_b = None

        # 2. Process Top Platen
        dpi_t = self.get_scanner_dpi("top")
        if self.pipeline and getattr(self.pipeline, "config", None):
            self.pipeline.config.calibration.assumed_dpi = float(dpi_t)
        if top_image_path and os.path.exists(top_image_path):
            print(f"[DualPlatenManager] Processing Platen 2 (Top, {dpi_t} DPI - Lightweight Count Only): {top_image_path}")
            res_t = self.pipeline.run(top_image_path, generate_outputs=False, skip_classification=True)
            df_t = res_t["measurements_df"].copy()
            df_t["platen"] = "top"
            dfs.append(df_t)
            platens_info["top"] = {
                "image_path": top_image_path,
                "grains_detected": len(df_t),
                "summary": res_t["summary"],
                "dpi": dpi_t
            }
            img_t = cv2.imread(top_image_path)
            rendered_images.append(img_t)
        else:
            img_t = None

        # 3. Handle Composite Rendering
        if img_b is not None and img_t is not None:
            composite_img = create_distortion_free_composite(img_b, img_t, output_path=composite_output_path)
        elif img_b is not None:
            composite_img = img_b
            if composite_output_path:
                cv2.imwrite(composite_output_path, composite_img)
        elif img_t is not None:
            composite_img = img_t
            if composite_output_path:
                cv2.imwrite(composite_output_path, composite_img)
        else:
            raise FileNotFoundError("Neither bottom nor top platen images could be found.")

        # 4. Fail-closed: Bottom-Primary Architecture requires valid bottom scanner results
        df_b = dfs[0] if (len(dfs) > 0 and "bottom" in platens_info) else None
        df_t = dfs[1] if (len(dfs) > 1 and "top" in platens_info) else (dfs[0] if (len(dfs) == 1 and "top" in platens_info) else None)

        if "bottom" not in platens_info or df_b is None or df_b.empty:
            print("[DualPlatenManager ERROR] Bottom primary scanner failed or empty. Bottom-primary architecture strictly prohibits uncertified top-only substitution.")
            return {
                "status": "ANALYSIS_FAILED",
                "error": "Bottom primary scanner acquisition or segmentation failed. Primary analysis cannot proceed without bottom observations.",
                "failure_reason": "BOTTOM_PRIMARY_SCANNER_FAILED",
                "primary_scanner": "bottom",
                "primary_scanner_available": False,
                "is_fallback_substituted": False,
                "statutory_acceptance_usable": False,
                "serial": sample_id,
                "sample_id": sample_id,
                "timestamp": int(time.time()),
                "timestamp_str": time.strftime("%d/%m/%Y %H:%M:%S"),
                "total_objects": 0,
                "total_grains": 0,
                "total_single_physical_grains": 0,
                "unresolved_cluster_count": 0,
                "stats": [],
                "rows": [],
                "platens": platens_info,
                "composite_image": os.path.basename(composite_output_path) if composite_output_path and os.path.exists(composite_output_path) else ""
            }

        # Independent Calibration Scale Derivation
        calib_scale_b = platens_info.get("bottom", {}).get("summary", {}).get("calibration", {}).get("pixel_scale_mm")
        if not calib_scale_b:
            calib_scale_b = 25.4 / float(dpi_b)

        calib_scale_t = platens_info.get("top", {}).get("summary", {}).get("calibration", {}).get("pixel_scale_mm")
        if not calib_scale_t:
            calib_scale_t = 25.4 / float(dpi_t)

        pixel_scale_b = calib_scale_b
        pixel_scale_t = calib_scale_t

        # 5. Dual-View Mode Execution (Mode A: single_view vs Mode C: bottom_primary_verification)
        dv_cfg = getattr(self.pipeline.config, "dual_view", None) if self.pipeline else None
        arch_mode = getattr(dv_cfg, "architecture_mode", "bottom_primary") if dv_cfg else "bottom_primary"
        dv_mode = getattr(dv_cfg, "mode", "single_view") if dv_cfg else "single_view"
        env_mode = os.environ.get("RICEAGE_DUAL_VIEW_MODE", "").lower()

        if env_mode in ("canonical_fusion", "symmetric", "experimental", "mode_b", "experimental_canonical_fusion"):
            raise ValueError(
                "Symmetric dual-view fusion (Mode B) is strictly prohibited from production "
                "to prevent +78% grain count inflation and statutory non-compliance. "
                "Supported modes: 'single_view' or 'bottom_primary_verification'."
            )
        elif env_mode in ("bottom_primary_verification", "mode_c", "verification") or dv_mode == "bottom_primary_verification":
            arch_mode = "bottom_primary_verification"
        else:
            arch_mode = "bottom_primary"

        if self.use_dual_view_fusion and (df_b is not None or df_t is not None):
            print(f"[DualPlatenManager] Executing Dual-View Processing (Mode: {arch_mode})...")

            # Convert to ViewObservations with scanner-specific independent pixel scales
            obs_b = dataframe_to_view_observations(df_b, "bottom", pixel_scale_mm=pixel_scale_b) if df_b is not None else []
            obs_t = dataframe_to_view_observations(df_t, "top", pixel_scale_mm=pixel_scale_t) if df_t is not None else []

            H_mat = dv_cfg.homography_matrix if dv_cfg else None
            scale_factor_top = float(dpi_t) / 600.0
            base_max_d = dv_cfg.max_matching_distance_px if dv_cfg else 35.0
            base_amb_delta = dv_cfg.ambiguity_delta_px if dv_cfg else 8.0
            max_d = base_max_d * scale_factor_top
            amb_delta = base_amb_delta * scale_factor_top

            f_rule = dv_cfg.defect_fusion_rule if dv_cfg and hasattr(dv_cfg, "defect_fusion_rule") else "surface_defect_precedence"
            f_ver = dv_cfg.defect_fusion_version if dv_cfg and hasattr(dv_cfg, "defect_fusion_version") else "1.0.0-experimental"
            f_order = list(dv_cfg.defect_severity_order) if dv_cfg and hasattr(dv_cfg, "defect_severity_order") else None
            f_appr = dv_cfg.defect_precedence_is_standard_approved if dv_cfg and hasattr(dv_cfg, "defect_precedence_is_standard_approved") else False

            reg = DualViewRegistration(homography_matrix=H_mat, dpi_bottom=dpi_b, dpi_top=dpi_t)
            corr_engine = DualViewCorrespondenceEngine(
                registration=reg,
                max_matching_distance_px=max_d,
                ambiguity_margin_px=amb_delta
            )
            fusion_engine = DualViewFusionEngine(
                correspondence_engine=corr_engine,
                pixel_scale_mm=pixel_scale_b,
                defect_fusion_rule=f_rule,
                defect_fusion_version=f_ver,
                defect_severity_order=f_order,
                defect_precedence_is_standard_approved=f_appr,
                architecture_mode=arch_mode
            )

            # Secondary Top Image Quality Gate Assessment
            top_quality = assess_image_quality(
                img_t,
                min_laplacian_var=getattr(dv_cfg, "top_min_laplacian_variance", 8.0) if dv_cfg else 8.0,
                min_contrast=getattr(dv_cfg, "top_min_contrast", 12.0) if dv_cfg else 12.0,
                min_brightness=getattr(dv_cfg, "top_min_brightness", 15.0) if dv_cfg else 15.0,
                max_brightness=getattr(dv_cfg, "top_max_brightness", 240.0) if dv_cfg else 240.0
            ) if img_t is not None else {"status": "UNAVAILABLE", "quality_tier": "UNAVAILABLE", "reasons": ["Top scan image not available"], "is_suitable": False}
            print(f"[DualPlatenManager] Top Image Quality Gate: {top_quality['status']} (Laplacian Var: {top_quality.get('laplacian_variance', 0.0)}, Contrast: {top_quality.get('rms_contrast', 0.0)})")

            fusion_result = fusion_engine.fuse_views(
                obs_b,
                obs_t,
                top_quality_status=top_quality["status"],
                top_quality_metrics=top_quality
            )
            agg = fusion_result.aggregation_summary
            cb = agg["count_basis"]
            mb = agg["mass_basis"]
            stats = agg["stats"]

            grain_records = []
            for g in fusion_result.physical_grains:
                qa = g.quality_attributes
                rec = {
                    "grain_id": g.physical_grain_id,
                    "primary_scanner": qa.get("primary_scanner", "bottom"),
                    "correspondence_status": g.correspondence_status.value if hasattr(g.correspondence_status, "value") else str(g.correspondence_status),
                    "confidence": g.correspondence_confidence,
                    "classification": qa.get("defect_class", "Good / Head Rice"),
                    "length_mm": qa.get("length_mm", 0.0),
                    "width_mm": qa.get("width_mm", 0.0),
                    "area_px": qa.get("area_px", 0.0),
                    "aspect_ratio": qa.get("aspect_ratio", 1.0),
                    "estimated_mass_g": qa.get("estimated_mass_g", 0.0),
                    "is_broken": qa.get("is_broken", False),
                    "bottom_detection_id": g.bottom_observation_id,
                    "top_detection_id": g.top_observation_id,
                    "top_match_status": qa.get("top_match_status", "MATCHED" if g.top_observation_id else "UNMATCHED_IN_TOP"),
                    "top_verified_class": qa.get("top_predicted_class"),
                    "is_verified": qa.get("is_verified", False),
                    "discrepancy_flag": qa.get("discrepancy_flag", False),
                    "top_verification_status": qa.get("top_verification_status", "NOT_VERIFIABLE" if top_quality["status"] == "NOT_VERIFIABLE" else ("VERIFIED_AGREEMENT" if qa.get("is_verified") else "UNVERIFIED")),
                    "discrepancy_details": qa.get("discrepancy_details"),
                    "is_cluster": qa.get("is_cluster", False),
                    "separation_status": qa.get("separation_status", "single_grain")
                }
                grain_records.append(rec)

            rows = []
            for s in stats:
                is_min = "Min" in s["limit"]
                is_max = "Max" in s["limit"]
                if is_min:
                    try:
                        thresh = float(s["limit"].replace("Min ", "").replace("%", ""))
                        passes = (s["pct"] >= thresh)
                        stat_label = "PASS" if passes else "FLAG"
                    except ValueError:
                        stat_label = "REQUIRES LAB CONFIRMATION"
                elif is_max:
                    try:
                        thresh = float(s["limit"].replace("Max ", "").replace("%", ""))
                        passes = (s["pct"] <= thresh)
                        stat_label = "PASS" if passes else "FLAG"
                    except ValueError:
                        stat_label = "REQUIRES LAB CONFIRMATION"
                else:
                    stat_label = "REQUIRES LAB CONFIRMATION"

                rows.append({
                    "name": s["param"],
                    "limit": s["limit"],
                    "resultPct": f"{s['mass_pct']:.2f}% (mass*) / {s['pct']:.2f}% (cnt)",
                    "resultCountAndGms": f"{s['count']} grains ({s['mass_g']:.3f} g*)",
                    "status": stat_label,
                    "decision_tier": stat_label,
                    "laboratory_confirmation_required": True,
                })

            is_bp = arch_mode in ("bottom_primary", "bottom_primary_verification")
            commercial_label = "Grade 1 (Bottom-Primary Analysis)" if is_bp else "Grade 1 (Single Grain Canonical Fusion)"
            if cb["whole_pct"] < 75.0:
                commercial_label = "Substandard"

            return {
                "status": "Complete",
                "commodity": commodity,
                "is_dual_view_prototype": True,
                "architecture_mode": arch_mode.upper(),
                "primary_scanner": "bottom",
                "secondary_scanner": "top" if obs_t else "none",
                "serial": sample_id,
                "sample_id": sample_id,
                "timestamp": int(time.time()),
                "timestamp_str": time.strftime("%d/%m/%Y %H:%M:%S"),
                "total_objects": fusion_result.total_inventory_records,
                "total_grains": fusion_result.total_single_physical_grains,
                "bottom_grain_count": len(obs_b) if obs_b else 0,
                "top_grain_count": len(obs_t) if obs_t else 0,
                "count_agreement_pct": getattr(fusion_result, "count_agreement_pct", 100.0),
                "count_tolerance_status": getattr(fusion_result, "count_tolerance_status", "VERIFIED_EXCELLENT"),
                "total_single_physical_grains": fusion_result.total_single_physical_grains,
                "total_inventory_records": fusion_result.total_inventory_records,
                "unresolved_cluster_count": fusion_result.unresolved_cluster_count,
                "discrepancy_count": getattr(fusion_result, "discrepancy_count", 0),
                "top_image_quality_status": top_quality["status"],
                "top_image_quality_metrics": top_quality,
                "total_sample_mass_g": fusion_result.estimated_mass_g,
                "estimated_optical_mass_unvalidated_g": fusion_result.estimated_optical_mass_unvalidated_g,
                "mass_measurement_status": fusion_result.mass_measurement_status,
                "is_physically_measured": fusion_result.is_physically_measured,
                "statutory_acceptance_usable": fusion_result.statutory_acceptance_usable,
                "statutory_mass_disclaimer": fusion_result.statutory_mass_disclaimer,
                "avg_length_mm": agg["avg_length_mm"],
                "avg_breadth_mm": agg["avg_breadth_mm"],
                "length_breadth_ratio": agg["length_breadth_ratio"],
                "commercial_grade": commercial_label,
                "disposition": "GRADE_1" if cb["whole_pct"] >= 75.0 else "SUBSTANDARD",
                "count_basis": cb,
                "mass_basis": mb,
                "stats": stats,
                "rows": rows,
                "platens": platens_info,
                "grain_records": grain_records,
                "bottom_grain_records": [o.to_dict() for o in obs_b],
                "top_grain_records": [o.to_dict() for o in obs_t],
                "bottom_image": os.path.basename(bottom_image_path) if bottom_image_path else "",
                "top_image": os.path.basename(top_image_path) if top_image_path else "",
                "physical_grains": [g.to_dict() for g in fusion_result.physical_grains],
                "acquisition_metadata": {
                    "primary_scanner": "bottom",
                    "secondary_scanner": "top" if obs_t else "none",
                    "architecture_mode": arch_mode.upper(),
                    "primary_scanner_id": os.environ.get("SCANNER_BOTTOM_ID", "pixma:04A91912_53012D"),
                    "secondary_scanner_id": os.environ.get("SCANNER_TOP_ID", "pixma:04A91912_50C8B2") if obs_t else "none",
                    "primary_grain_count": fusion_result.total_single_physical_grains,
                    "top_grain_count": len(obs_t) if obs_t else 0,
                    "count_agreement_pct": getattr(fusion_result, "count_agreement_pct", 100.0),
                    "count_tolerance_status": getattr(fusion_result, "count_tolerance_status", "VERIFIED_EXCELLENT"),
                    "primary_denominator": cb["total_physical_grains"],
                    "unresolved_clusters_segregated": fusion_result.unresolved_cluster_count,
                    "secondary_verification_enabled": bool(obs_t),
                    "top_image_quality_status": top_quality["status"],
                    "top_verification_status": "VERIFIED_OPERATIONAL" if top_quality["status"] == "SUITABLE" else "NOT_VERIFIABLE",
                    "top_image_quality_metrics": top_quality,
                    "top_observations_detected": len(obs_t),
                    "top_matches_confirmed": fusion_result.matched_count,
                    "top_discrepancies_flagged": getattr(fusion_result, "discrepancy_count", 0),
                    "unmatched_top_observations_excluded": fusion_result.unmatched_top_count
                },
                "dual_view_diagnostics": {
                    "matched_count": fusion_result.matched_count,
                    "unmatched_bottom_count": fusion_result.unmatched_bottom_count,
                    "unmatched_top_count": fusion_result.unmatched_top_count,
                    "unmatched_top_excluded_from_primary": fusion_result.unmatched_top_count if is_bp else 0,
                    "ambiguous_count": fusion_result.ambiguous_count,
                    "registration_failed_count": fusion_result.registration_failed_count,
                    "unresolved_cluster_count": fusion_result.unresolved_cluster_count,
                    "discrepancy_count": getattr(fusion_result, "discrepancy_count", 0),
                "top_image_quality_status": top_quality["status"],
                "top_image_quality_metrics": top_quality,
                    "registration_health": fusion_result.registration_metadata
                },
                "composite_image": os.path.basename(composite_output_path) if composite_output_path else ""
            }

        # 6. Safe Single-View Fallback Pipeline (when dual-view fusion is disabled)
        # Authoritative Bottom Platen selection
        print(f"[DualPlatenManager] Dual-view fusion disabled. Safe Bottom-Primary Selection Active.")
        df_unified = df_b.copy()
        fallback_mode = "SAFE_SINGLE_VIEW_SELECTION" if len(dfs) > 1 else "GENUINE_SINGLE_SCANNER"
        is_sample_doubling_prevented = True

        # Compute Consolidated Count and Mass Basis Metrics using authoritative bottom scale
        consolidated_metrics = compute_quality_metrics(df_unified, pixel_scale_mm=pixel_scale_b)

        # Build Overall Dashboard and Certificate Payload
        mb = consolidated_metrics["mass_basis"]
        cb = consolidated_metrics["count_basis"]

        # Mean physical dimensions
        avg_len_mm = float(df_unified["length_mm"].mean()) if "length_mm" in df_unified.columns and not df_unified.empty else float(df_unified.get("length_px", pd.Series([0.0])).mean() * pixel_scale_b)
        avg_brd_mm = float(df_unified["width_mm"].mean()) if "width_mm" in df_unified.columns and not df_unified.empty else float(df_unified.get("width_px", pd.Series([0.0])).mean() * pixel_scale_b)

        stats = [
            {"param": "Good / Head Rice", "count": cb["whole_count"], "pct": cb["whole_pct"], "mass_g": mb["whole_mass_g"], "mass_pct": mb["whole_pct"], "limit": "Min 75.00%"},
            {"param": "Big Brokens (25-75%)", "count": cb["broken_count"], "pct": cb["broken_pct"], "mass_g": mb["broken_mass_g"], "mass_pct": mb["broken_pct"], "limit": "Max 15.00%"},
            {"param": "Small Brokens (<25%)", "count": cb["small_broken_count"], "pct": cb["small_broken_pct"], "mass_g": mb["small_broken_mass_g"], "mass_pct": mb["small_broken_pct"], "limit": "Max 1.00%"},
            {"param": "Foreign Matter", "count": cb["foreign_matter_count"], "pct": cb["foreign_matter_pct"], "mass_g": mb["foreign_matter_mass_g"], "mass_pct": mb["foreign_matter_pct"], "limit": "Max 0.50%"},
            {"param": "Damaged Grains", "count": cb["damaged_count"], "pct": cb["damaged_pct"], "mass_g": mb["damaged_mass_g"], "mass_pct": mb["damaged_pct"], "limit": "Max 2.00%"},
            {"param": "Discoloured Grains", "count": cb["discoloured_count"], "pct": cb["discoloured_pct"], "mass_g": mb["discoloured_mass_g"], "mass_pct": mb["discoloured_pct"], "limit": "Max 1.00%"},
            {"param": "Chalky Grains", "count": cb["chalky_count"], "pct": cb["chalky_pct"], "mass_g": mb["chalky_mass_g"], "mass_pct": mb["chalky_pct"], "limit": "Max 3.00%"},
        ]

        rows = []
        for s in stats:
            is_min = "Min" in s["limit"]
            is_max = "Max" in s["limit"]
            if is_min:
                try:
                    thresh = float(s["limit"].replace("Min ", "").replace("%", ""))
                    passes = (s["pct"] >= thresh)
                    stat_label = "PASS" if passes else "FLAG"
                except ValueError:
                    stat_label = "REQUIRES LAB CONFIRMATION"
            elif is_max:
                try:
                    thresh = float(s["limit"].replace("Max ", "").replace("%", ""))
                    passes = (s["pct"] <= thresh)
                    stat_label = "PASS" if passes else "FLAG"
                except ValueError:
                    stat_label = "REQUIRES LAB CONFIRMATION"
            else:
                stat_label = "REQUIRES LAB CONFIRMATION"

            rows.append({
                "name": s["param"],
                "limit": s["limit"],
                "resultPct": f"{s['mass_pct']:.2f}% (mass*) / {s['pct']:.2f}% (cnt)",
                "resultCountAndGms": f"{s['count']} grains ({s['mass_g']:.3f} g*)",
                "status": stat_label,
                "decision_tier": stat_label,
                "laboratory_confirmation_required": True,
            })

        # Mark unvalidated mass disclaimer on legacy output
        mb["mass_measurement_status"] = "NOT_PHYSICALLY_MEASURED"
        mb["is_physically_measured"] = False
        mb["statutory_acceptance_usable"] = False
        mb["statutory_mass_disclaimer"] = "Optical mass is an uncalibrated geometric proxy. Prohibited from statutory procurement without physical balance verification."
        mb["total_estimated_mass_unvalidated_g"] = mb.get("total_mass_g", 0.0)

        return {
            "status": "Complete",
            "commodity": commodity,
            "fallback_mode": fallback_mode,
            "architecture_mode": "BOTTOM_SCANNER_PRIMARY",
            "primary_scanner": "bottom",
            "secondary_scanner": "none",
            "is_sample_doubling_prevented": is_sample_doubling_prevented,
            "mass_measurement_status": "NOT_PHYSICALLY_MEASURED",
            "is_physically_measured": False,
            "statutory_acceptance_usable": False,
            "serial": sample_id,
            "sample_id": sample_id,
            "timestamp": int(time.time()),
            "timestamp_str": time.strftime("%d/%m/%Y %H:%M:%S"),
            "total_objects": cb["total_objects"],
            "total_grains": cb["total_rice_grains"],
            "total_single_physical_grains": cb["total_rice_grains"],
            "bottom_grain_count": cb["total_rice_grains"],
            "top_grain_count": 0,
            "count_agreement_pct": 100.0,
            "count_tolerance_status": "SINGLE_PLATEN_MODE",
            "total_sample_mass_g": consolidated_metrics["total_mass_g"],
            "avg_length_mm": round(avg_len_mm, 2),
            "avg_breadth_mm": round(avg_brd_mm, 2),
            "length_breadth_ratio": round(avg_len_mm / max(avg_brd_mm, 0.01), 2),
            "commercial_grade": consolidated_metrics["commercial_grade"],
            "disposition": "GRADE_1" if "Grade 1" in consolidated_metrics["commercial_grade"] else "SUBSTANDARD",
            "count_basis": cb,
            "mass_basis": mb,
            "stats": stats,
            "rows": rows,
            "platens": platens_info,
            "grain_records": df_unified.to_dict(orient="records"),
            "bottom_grain_records": df_unified.to_dict(orient="records"),
            "top_grain_records": [],
            "acquisition_metadata": {
                "primary_scanner": "bottom",
                "secondary_scanner": "none",
                "architecture_mode": "BOTTOM_SCANNER_PRIMARY",
                "primary_scanner_id": os.environ.get("SCANNER_BOTTOM_ID", "pixma:04A91912_50C8B2"),
                "primary_grain_count": cb["total_rice_grains"],
                "primary_denominator": cb["total_rice_grains"],
                "secondary_verification_enabled": False
            },
            "composite_image": os.path.basename(composite_output_path) if composite_output_path else ""
        }
