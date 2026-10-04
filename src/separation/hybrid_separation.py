"""Hybrid touching-grain separation and multi-method benchmark module."""

from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np
import cv2
import pandas as pd

from .cluster_classifier import evaluate_component_geometry, ComponentCategory, GrainStatus
from .distance_transform import compute_distance_transform
from .watershed import separate_touching_grains
from .concave_neck import split_cluster_concave_neck, detect_concave_points, pair_concave_points


def benchmark_and_separate_hybrid(
    binary_mask: np.ndarray,
    min_peak_distance_px: int = 8,
    peak_threshold_rel: float = 0.38,
    cluster_area_thresh_px: int = 650,
    solidity_thresh: float = 0.92,
    min_valid_area_px: int = 40,
    median_grain_area_px: float = 380.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any], Dict[int, Dict[str, Any]]]:
    """Execute and benchmark:
    - Method A: Watershed alone
    - Method B: Concave-neck alone (on clusters)
    - Method C: Hybrid (Watershed + Concave repair)
    
    Returns:
        mask_watershed: int32 instance mask for Method A.
        mask_concave: int32 instance mask for Method B.
        mask_hybrid: int32 instance mask for Method C (Hybrid).
        cluster_benchmark_df: DataFrame with per-cluster comparison metrics.
        summary_benchmark: High-level comparative statistics.
        cluster_rois: Dict of cluster crop images/masks for visualization.
    """
    h, w = binary_mask.shape[:2]
    
    # 1. Run Method A: Standard selective watershed
    mask_ws, diags_ws, stats_ws = separate_touching_grains(
        binary_mask,
        min_peak_distance_px=min_peak_distance_px,
        peak_threshold_rel=peak_threshold_rel,
        cluster_area_thresh_px=cluster_area_thresh_px,
        solidity_thresh=solidity_thresh,
        min_valid_area_px=min_valid_area_px
    )

    # 2. Extract initial connected components
    num_cc, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8), connectivity=8
    )

    cluster_records = []
    cluster_rois: Dict[int, Dict[str, Any]] = {}

    # Initialize masks for Concave Alone and Hybrid
    mask_concave = np.zeros((h, w), dtype=np.int32)
    mask_hybrid = mask_ws.copy()

    concave_current_id = 1
    hybrid_current_id = int(mask_ws.max()) + 1

    method_a_resolved = 0
    method_b_resolved = 0
    method_c_resolved = 0
    total_clusters = 0

    clusters_repaired_by_concave = 0
    remaining_unresolved = 0

    for comp_idx in range(1, num_cc):
        comp_area = cc_stats[comp_idx, cv2.CC_STAT_AREA]
        if comp_area < min_valid_area_px:
            continue

        comp_x = cc_stats[comp_idx, cv2.CC_STAT_LEFT]
        comp_y = cc_stats[comp_idx, cv2.CC_STAT_TOP]
        comp_w = cc_stats[comp_idx, cv2.CC_STAT_WIDTH]
        comp_h = cc_stats[comp_idx, cv2.CC_STAT_HEIGHT]

        roi_mask = (cc_labels[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w] == comp_idx).astype(np.uint8) * 255

        geom = evaluate_component_geometry(
            roi_mask,
            solidity_thresh=solidity_thresh,
            single_area_range=(int(round(cluster_area_thresh_px * (180.0 / 650.0))), cluster_area_thresh_px),
            pair_area_range=(cluster_area_thresh_px, cluster_area_thresh_px * 2)
        )
        cat = geom["category"]

        # Only process clusters for comparison
        if cat in [ComponentCategory.TOUCHING_PAIR, ComponentCategory.CLUSTER_COMPLEX]:
            total_clusters += 1
            estimated_count = max(2, int(round(comp_area / median_grain_area_px)))

            # Inspect Method A result inside this cluster ROI
            roi_ws = mask_ws[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w]
            ws_unique = np.unique(roi_ws[roi_mask > 0])
            ws_unique = ws_unique[ws_unique > 0]
            ws_count = len(ws_unique)
            ws_success = (ws_count >= 2)
            if ws_success:
                method_a_resolved += 1

            # Method B: Concave-neck splitting alone on this cluster
            labeled_cnc, cnc_count, cnc_cuts = split_cluster_concave_neck(
                roi_mask,
                min_defect_depth_px=2.0,
                max_neck_width_px=60.0,
                min_piece_area_px=min_valid_area_px
            )
            cnc_success = (cnc_count >= 2)
            if cnc_success:
                method_b_resolved += 1
                for lbl in range(1, cnc_count + 1):
                    mask_concave[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w][labeled_cnc == lbl] = concave_current_id
                    concave_current_id += 1
            else:
                mask_concave[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w][roi_mask > 0] = concave_current_id
                concave_current_id += 1

            # Method C (Hybrid): If watershed failed (ws_count == 1), repair with concave neck
            hybrid_roi = roi_ws.copy()
            if not ws_success and cnc_success:
                # Concave neck successfully repaired watershed's failure!
                clusters_repaired_by_concave += 1
                method_c_resolved += 1
                final_status = "repaired_by_concave"
                final_count = cnc_count

                # Update mask_hybrid
                # Clear previous single watershed ID in this ROI
                mask_hybrid[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w][roi_mask > 0] = 0
                for lbl in range(1, cnc_count + 1):
                    mask_hybrid[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w][labeled_cnc == lbl] = hybrid_current_id
                    hybrid_current_id += 1
                hybrid_roi = mask_hybrid[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w]
            elif ws_success:
                method_c_resolved += 1
                final_status = "split_by_watershed"
                final_count = ws_count
            else:
                remaining_unresolved += 1
                final_status = "unresolved_both"
                final_count = 1

            cluster_records.append({
                "cluster_id": comp_idx,
                "bbox_x": comp_x,
                "bbox_y": comp_y,
                "bbox_w": comp_w,
                "bbox_h": comp_h,
                "original_area": comp_area,
                "solidity": round(geom["solidity"], 3),
                "estimated_grain_count": estimated_count,
                "watershed_count": ws_count,
                "concave_split_count": cnc_count,
                "final_count": final_count,
                "confidence_status": final_status,
                "cuts_found": len(cnc_cuts)
            })

            # Store crop masks for montage
            cluster_rois[comp_idx] = {
                "bbox": (comp_x, comp_y, comp_w, comp_h),
                "original_mask": roi_mask,
                "watershed_roi": roi_ws.copy(),
                "concave_roi": labeled_cnc,
                "hybrid_roi": hybrid_roi.copy(),
                "cnc_cuts": cnc_cuts,
                "status": final_status
            }

        else:
            # Single isolated grains or small fragments: carry over
            mask_concave[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w][roi_mask > 0] = concave_current_id
            concave_current_id += 1

    cluster_benchmark_df = pd.DataFrame(cluster_records)

    summary_benchmark = {
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

    return mask_ws, mask_concave, mask_hybrid, cluster_benchmark_df, summary_benchmark, cluster_rois
