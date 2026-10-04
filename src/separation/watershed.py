"""Selective marker-controlled watershed segmentation for touching grain separation."""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .cluster_classifier import evaluate_component_geometry, ComponentCategory, GrainStatus
from .distance_transform import compute_distance_transform


def separate_touching_grains(
    binary_mask: np.ndarray,
    min_peak_distance_px: int = 8,
    peak_threshold_rel: float = 0.38,
    cluster_area_thresh_px: int = 650,
    solidity_thresh: float = 0.92,
    min_valid_area_px: int = 40,
    min_single_area_px: Optional[int] = None
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Segment and separate touching grains using selective marker-controlled watershed.
    
    Workflow:
    1. Extract connected components from binary mask.
    2. For single grains (high solidity & normal area), preserve as single instances.
    3. For multi-grain clusters (large area or concave shape), compute local distance
       transform and identify local maxima seeds.
    4. Run watershed on the inverted distance map for clustered components.
    5. Assign contiguous global IDs (1..N) to all individual grains.
    6. Record diagnostic state (isolated, separated, under-segmented, fragment) for every grain.
    
    Returns:
        instance_mask: int32 array of shape (H, W), 0=background, 1..N=grain_id.
        grain_diagnostics: Dict[grain_id, dict] containing origin and separation metadata.
        summary_stats: High-level metrics on separation efficiency.
    """
    h, w = binary_mask.shape[:2]
    instance_mask = np.zeros((h, w), dtype=np.int32)
    
    num_cc, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(
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

    for comp_idx in range(1, num_cc):
        comp_area = cc_stats[comp_idx, cv2.CC_STAT_AREA]
        if comp_area < min_valid_area_px:
            continue

        comp_x = cc_stats[comp_idx, cv2.CC_STAT_LEFT]
        comp_y = cc_stats[comp_idx, cv2.CC_STAT_TOP]
        comp_w = cc_stats[comp_idx, cv2.CC_STAT_WIDTH]
        comp_h = cc_stats[comp_idx, cv2.CC_STAT_HEIGHT]

        # Extract local ROI for efficiency
        roi_mask = (cc_labels[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w] == comp_idx).astype(np.uint8)

        # Determine single grain minimum area threshold
        actual_min_single = min_single_area_px if min_single_area_px is not None else int(round(cluster_area_thresh_px * (180.0 / 650.0)))

        # Check geometry
        geom = evaluate_component_geometry(
            roi_mask,
            solidity_thresh=solidity_thresh,
            single_area_range=(actual_min_single, cluster_area_thresh_px),
            pair_area_range=(cluster_area_thresh_px, cluster_area_thresh_px * 2)
        )
        cat = geom["category"]

        if cat == ComponentCategory.SINGLE_GRAIN:
            # Clean isolated single grain: preserve directly
            instance_mask[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w][roi_mask > 0] = current_grain_id
            grain_diagnostics[current_grain_id] = {
                "status": GrainStatus.ISOLATED_SINGLE.value,
                "parent_component_id": comp_idx,
                "solidity": geom["solidity"],
                "component_area": comp_area,
                "was_split": False
            }
            isolated_single_count += 1
            current_grain_id += 1

        elif cat == ComponentCategory.FRAGMENT_OR_NOISE:
            # Small fragment
            instance_mask[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w][roi_mask > 0] = current_grain_id
            grain_diagnostics[current_grain_id] = {
                "status": GrainStatus.OVER_SEGMENTED_FRAGMENT.value,
                "parent_component_id": comp_idx,
                "solidity": geom["solidity"],
                "component_area": comp_area,
                "was_split": False
            }
            fragments_count += 1
            current_grain_id += 1

        else:
            # TOUCHING_PAIR or CLUSTER_COMPLEX: Apply local marker-controlled watershed
            clusters_processed += 1
            dist_local = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
            max_d = float(dist_local.max())

            peaks = peak_local_max(
                dist_local,
                min_distance=min_peak_distance_px,
                threshold_abs=max(max_d * peak_threshold_rel, 3.0),
                exclude_border=False
            )

            if len(peaks) > 1:
                # Multiple grain seeds detected -> build local markers
                local_markers = np.zeros_like(roi_mask, dtype=np.int32)
                for seed_idx, (pr, pc) in enumerate(peaks, start=1):
                    local_markers[pr, pc] = seed_idx

                # Dilate markers slightly using maximum filter so watershed has solid seeds
                local_markers = ndi.maximum_filter(local_markers, size=3)

                # Run watershed on inverted distance map
                local_ws = watershed(-dist_local, local_markers, mask=roi_mask.astype(bool))

                # Transfer each separated region into global instance mask
                num_sub_grains = 0
                for local_lbl in np.unique(local_ws):
                    if local_lbl == 0:
                        continue
                    sub_mask = (local_ws == local_lbl)
                    sub_area = int(np.sum(sub_mask))
                    if sub_area < min_valid_area_px:
                        continue

                    instance_mask[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w][sub_mask] = current_grain_id
                    grain_diagnostics[current_grain_id] = {
                        "status": GrainStatus.SEPARATED_FROM_CLUSTER.value,
                        "parent_component_id": comp_idx,
                        "solidity": geom["solidity"],
                        "component_area": comp_area,
                        "sub_area": sub_area,
                        "was_split": True
                    }
                    current_grain_id += 1
                    num_sub_grains += 1

                if num_sub_grains > 1:
                    clusters_split += 1
                    grains_from_clusters += num_sub_grains
                else:
                    # In case sub-regions filtered out
                    clusters_unresolved += 1
            else:
                # Could not resolve into multiple seeds: under-segmented cluster
                clusters_unresolved += 1
                instance_mask[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w][roi_mask > 0] = current_grain_id
                grain_diagnostics[current_grain_id] = {
                    "status": GrainStatus.UNDER_SEGMENTED_CLUSTER.value,
                    "parent_component_id": comp_idx,
                    "solidity": geom["solidity"],
                    "component_area": comp_area,
                    "was_split": False
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
