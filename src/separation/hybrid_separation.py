"""Hybrid touching-grain separation and multi-method benchmark module."""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import cv2
import pandas as pd

from .cluster_classifier import evaluate_component_geometry, ComponentCategory
from .watershed import separate_touching_grains
from .concave_neck import split_cluster_concave_neck


@dataclass
class ClusterComparison:
    cluster_id: int
    bbox: Tuple[int, int, int, int]
    original_area: int
    solidity: float
    estimated_grain_count: int
    watershed_count: int
    concave_split_count: int
    final_count: int
    status: str
    cuts_found: int
    labeled_concave: np.ndarray
    applied_cuts: List[Tuple[Tuple[int, int], Tuple[int, int]]]


def _evaluate_cluster(
    cluster_id: int,
    bbox: Tuple[int, int, int, int],
    roi_mask: np.ndarray,
    comp_area: int,
    solidity: float,
    mask_ws: np.ndarray,
    median_grain_area_px: float,
    min_valid_area_px: int
) -> ClusterComparison:
    cx, cy, cw, ch = bbox
    roi_slice = (slice(cy, cy + ch), slice(cx, cx + cw))
    
    roi_ws = mask_ws[roi_slice]
    ws_unique = np.unique(roi_ws[roi_mask > 0])
    ws_unique = ws_unique[ws_unique > 0]
    ws_count = len(ws_unique)
    ws_success = ws_count >= 2

    labeled_cnc, cnc_count, cnc_cuts = split_cluster_concave_neck(
        roi_mask,
        min_defect_depth_px=2.0,
        max_neck_width_px=60.0,
        min_piece_area_px=min_valid_area_px
    )
    cnc_success = cnc_count >= 2

    estimated_count = max(2, int(round(comp_area / median_grain_area_px)))

    if not ws_success and cnc_success:
        status = "repaired_by_concave"
        final_count = cnc_count
    elif ws_success:
        status = "split_by_watershed"
        final_count = ws_count
    else:
        status = "unresolved_both"
        final_count = 1

    return ClusterComparison(
        cluster_id=cluster_id,
        bbox=bbox,
        original_area=comp_area,
        solidity=round(solidity, 3),
        estimated_grain_count=estimated_count,
        watershed_count=ws_count,
        concave_split_count=cnc_count,
        final_count=final_count,
        status=status,
        cuts_found=len(cnc_cuts),
        labeled_concave=labeled_cnc,
        applied_cuts=cnc_cuts
    )


def benchmark_and_separate_hybrid(
    binary_mask: np.ndarray,
    min_peak_distance_px: int = 8,
    peak_threshold_rel: float = 0.38,
    cluster_area_thresh_px: int = 650,
    solidity_thresh: float = 0.92,
    min_valid_area_px: int = 40,
    median_grain_area_px: float = 380.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any], Dict[int, Dict[str, Any]]]:
    """Execute and benchmark Watershed, Concave-Neck, and Hybrid separation."""
    h, w = binary_mask.shape[:2]

    mask_ws, _, _ = separate_touching_grains(
        binary_mask,
        min_peak_distance_px=min_peak_distance_px,
        peak_threshold_rel=peak_threshold_rel,
        cluster_area_thresh_px=cluster_area_thresh_px,
        solidity_thresh=solidity_thresh,
        min_valid_area_px=min_valid_area_px
    )

    num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8), connectivity=8
    )

    mask_concave = np.zeros((h, w), dtype=np.int32)
    mask_hybrid = mask_ws.copy()

    concave_current_id = 1
    hybrid_current_id = int(mask_ws.max()) + 1

    cluster_records = []
    cluster_rois: Dict[int, Dict[str, Any]] = {}

    method_a_resolved = 0
    method_b_resolved = 0
    method_c_resolved = 0
    total_clusters = 0
    clusters_repaired_by_concave = 0
    remaining_unresolved = 0

    min_single_area = int(round(cluster_area_thresh_px * (180.0 / 650.0)))

    for comp_idx in range(1, num_cc):
        comp_area = cc_stats[comp_idx, cv2.CC_STAT_AREA]
        if comp_area < min_valid_area_px:
            continue

        cx = cc_stats[comp_idx, cv2.CC_STAT_LEFT]
        cy = cc_stats[comp_idx, cv2.CC_STAT_TOP]
        cw = cc_stats[comp_idx, cv2.CC_STAT_WIDTH]
        ch = cc_stats[comp_idx, cv2.CC_STAT_HEIGHT]

        roi_slice = (slice(cy, cy + ch), slice(cx, cx + cw))
        roi_mask = (cc_labels[roi_slice] == comp_idx).astype(np.uint8) * 255

        geom = evaluate_component_geometry(
            roi_mask,
            solidity_thresh=solidity_thresh,
            single_area_range=(min_single_area, cluster_area_thresh_px),
            pair_area_range=(cluster_area_thresh_px, cluster_area_thresh_px * 2)
        )

        if geom["category"] in [ComponentCategory.TOUCHING_PAIR, ComponentCategory.CLUSTER_COMPLEX]:
            total_clusters += 1
            cmp = _evaluate_cluster(
                cluster_id=comp_idx,
                bbox=(cx, cy, cw, ch),
                roi_mask=roi_mask,
                comp_area=comp_area,
                solidity=geom["solidity"],
                mask_ws=mask_ws,
                median_grain_area_px=median_grain_area_px,
                min_valid_area_px=min_valid_area_px
            )

            if cmp.watershed_count >= 2:
                method_a_resolved += 1
            if cmp.concave_split_count >= 2:
                method_b_resolved += 1

            if cmp.concave_split_count >= 2:
                for lbl in range(1, cmp.concave_split_count + 1):
                    mask_concave[roi_slice][cmp.labeled_concave == lbl] = concave_current_id
                    concave_current_id += 1
            else:
                mask_concave[roi_slice][roi_mask > 0] = concave_current_id
                concave_current_id += 1

            if cmp.status == "repaired_by_concave":
                clusters_repaired_by_concave += 1
                method_c_resolved += 1
                mask_hybrid[roi_slice][roi_mask > 0] = 0
                for lbl in range(1, cmp.concave_split_count + 1):
                    mask_hybrid[roi_slice][cmp.labeled_concave == lbl] = hybrid_current_id
                    hybrid_current_id += 1
            elif cmp.status == "split_by_watershed":
                method_c_resolved += 1
            else:
                remaining_unresolved += 1

            cluster_records.append({
                "cluster_id": cmp.cluster_id,
                "bbox_x": cx,
                "bbox_y": cy,
                "bbox_w": cw,
                "bbox_h": ch,
                "original_area": cmp.original_area,
                "solidity": cmp.solidity,
                "estimated_grain_count": cmp.estimated_grain_count,
                "watershed_count": cmp.watershed_count,
                "concave_split_count": cmp.concave_split_count,
                "final_count": cmp.final_count,
                "confidence_status": cmp.status,
                "cuts_found": cmp.cuts_found
            })

            cluster_rois[comp_idx] = {
                "bbox": (cx, cy, cw, ch),
                "original_mask": roi_mask,
                "watershed_roi": mask_ws[roi_slice].copy(),
                "concave_roi": cmp.labeled_concave,
                "hybrid_roi": mask_hybrid[roi_slice].copy(),
                "cnc_cuts": cmp.applied_cuts,
                "status": cmp.status
            }
        else:
            mask_concave[roi_slice][roi_mask > 0] = concave_current_id
            concave_current_id += 1

    benchmark_df = pd.DataFrame(cluster_records)
    summary = {
        "total_clusters_evaluated": total_clusters,
        "method_a_watershed_resolved": method_a_resolved,
        "method_a_resolution_rate": round(method_a_resolved / max(1, total_clusters) * 100, 1),
        "method_b_concave_resolved": method_b_resolved,
        "method_b_resolution_rate": round(method_b_resolved / max(1, total_clusters) * 100, 1),
        "method_c_hybrid_resolved": method_c_resolved,
        "method_c_resolution_rate": round(method_c_resolved / max(1, total_clusters) * 100, 1),
        "clusters_repaired_by_concave": clusters_repaired_by_concave,
        "remaining_unresolved_clusters": remaining_unresolved,
    }

    return mask_ws, mask_concave, mask_hybrid, benchmark_df, summary, cluster_rois

