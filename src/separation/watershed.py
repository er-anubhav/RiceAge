"""Selective marker-controlled watershed segmentation for touching grain separation."""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .cluster_classifier import evaluate_component_geometry, ComponentCategory, GrainStatus


def _segment_cluster_watershed(
    roi_mask: np.ndarray,
    min_peak_distance_px: int,
    peak_threshold_rel: float
) -> Optional[np.ndarray]:
    """Segment a multi-grain cluster mask into distinct labels using distance watershed."""
    dist_local = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    max_d = float(dist_local.max())

    peaks = peak_local_max(
        dist_local,
        min_distance=min_peak_distance_px,
        threshold_abs=max(max_d * peak_threshold_rel, 3.0),
        exclude_border=False,
    )
    if len(peaks) <= 1:
        return None

    markers = np.zeros_like(roi_mask, dtype=np.int32)
    for seed_idx, (pr, pc) in enumerate(peaks, start=1):
        markers[pr, pc] = seed_idx

    markers = ndi.maximum_filter(markers, size=3)
    return watershed(-dist_local, markers, mask=roi_mask.astype(bool))


def separate_touching_grains(
    binary_mask: np.ndarray,
    min_peak_distance_px: int = 8,
    peak_threshold_rel: float = 0.38,
    cluster_area_thresh_px: int = 650,
    solidity_thresh: float = 0.92,
    min_valid_area_px: int = 40,
    min_single_area_px: Optional[int] = None
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Segment and separate touching grains using selective marker-controlled watershed."""
    h, w = binary_mask.shape[:2]
    instance_mask = np.zeros((h, w), dtype=np.int32)
    
    num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8), connectivity=8
    )

    current_grain_id = 1
    grain_diagnostics: Dict[int, Dict[str, Any]] = {}

    clusters_processed = 0
    clusters_split = 0
    clusters_unresolved = 0
    isolated_single_count = 0
    grains_from_clusters = 0
    fragments_count = 0

    min_single = min_single_area_px if min_single_area_px is not None else int(round(cluster_area_thresh_px * (180.0 / 650.0)))

    for comp_idx in range(1, num_cc):
        comp_area = cc_stats[comp_idx, cv2.CC_STAT_AREA]
        if comp_area < min_valid_area_px:
            continue

        cx = cc_stats[comp_idx, cv2.CC_STAT_LEFT]
        cy = cc_stats[comp_idx, cv2.CC_STAT_TOP]
        cw = cc_stats[comp_idx, cv2.CC_STAT_WIDTH]
        ch = cc_stats[comp_idx, cv2.CC_STAT_HEIGHT]

        roi_slice = (slice(cy, cy + ch), slice(cx, cx + cw))
        roi_mask = (cc_labels[roi_slice] == comp_idx).astype(np.uint8)

        geom = evaluate_component_geometry(
            roi_mask,
            solidity_thresh=solidity_thresh,
            single_area_range=(min_single, cluster_area_thresh_px),
            pair_area_range=(cluster_area_thresh_px, cluster_area_thresh_px * 2),
        )
        category = geom["category"]

        if category == ComponentCategory.SINGLE_GRAIN:
            instance_mask[roi_slice][roi_mask > 0] = current_grain_id
            grain_diagnostics[current_grain_id] = {
                "status": GrainStatus.ISOLATED_SINGLE.value,
                "parent_component_id": comp_idx,
                "solidity": geom["solidity"],
                "component_area": comp_area,
                "was_split": False,
            }
            isolated_single_count += 1
            current_grain_id += 1

        elif category == ComponentCategory.FRAGMENT_OR_NOISE:
            instance_mask[roi_slice][roi_mask > 0] = current_grain_id
            grain_diagnostics[current_grain_id] = {
                "status": GrainStatus.OVER_SEGMENTED_FRAGMENT.value,
                "parent_component_id": comp_idx,
                "solidity": geom["solidity"],
                "component_area": comp_area,
                "was_split": False,
            }
            fragments_count += 1
            current_grain_id += 1

        else:
            clusters_processed += 1
            local_ws = _segment_cluster_watershed(roi_mask, min_peak_distance_px, peak_threshold_rel)

            if local_ws is not None:
                sub_regions = [
                    (lbl, int(np.sum(local_ws == lbl)))
                    for lbl in np.unique(local_ws)
                    if lbl > 0 and np.sum(local_ws == lbl) >= min_valid_area_px
                ]

                if len(sub_regions) > 1:
                    for lbl, sub_area in sub_regions:
                        instance_mask[roi_slice][local_ws == lbl] = current_grain_id
                        grain_diagnostics[current_grain_id] = {
                            "status": GrainStatus.SEPARATED_FROM_CLUSTER.value,
                            "parent_component_id": comp_idx,
                            "solidity": geom["solidity"],
                            "component_area": comp_area,
                            "sub_area": sub_area,
                            "was_split": True,
                        }
                        current_grain_id += 1

                    clusters_split += 1
                    grains_from_clusters += len(sub_regions)
                    continue

            clusters_unresolved += 1
            instance_mask[roi_slice][roi_mask > 0] = current_grain_id
            grain_diagnostics[current_grain_id] = {
                "status": GrainStatus.UNDER_SEGMENTED_CLUSTER.value,
                "parent_component_id": comp_idx,
                "solidity": geom["solidity"],
                "component_area": comp_area,
                "was_split": False,
            }
            current_grain_id += 1

    summary_stats = {
        "raw_connected_components": num_cc - 1,
        "total_detected_grains": current_grain_id - 1,
        "isolated_single_grains": isolated_single_count,
        "clusters_detected": clusters_processed,
        "clusters_successfully_split": clusters_split,
        "clusters_unresolved": clusters_unresolved,
        "grains_derived_from_clusters": grains_from_clusters,
        "fragments_count": fragments_count,
    }

    return instance_mask, grain_diagnostics, summary_stats
