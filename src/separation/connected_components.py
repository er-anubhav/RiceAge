"""Connected component analysis baseline module."""

from typing import Tuple, Dict, Any, List
import numpy as np
import cv2


def extract_connected_components(
    binary_mask: np.ndarray,
    connectivity: int = 8
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Perform connected components labeling on a binary mask.
    
    Args:
        binary_mask: uint8 binary mask (0 or 255).
        connectivity: 4 or 8.
        
    Returns:
        num_labels: Total number of labels including background.
        labels: Label matrix (H, W).
        stats: Statistics array (N, 5) with [x, y, w, h, area].
        centroids: Centroid coordinates (N, 2).
    """
    return cv2.connectedComponentsWithStats(binary_mask, connectivity=connectivity)


def summarize_connected_components(
    stats: np.ndarray,
    min_area: int = 50
) -> Dict[str, Any]:
    """Generate summary statistics for connected components."""
    if len(stats) <= 1:
        return {
            "total_components": 0,
            "valid_components": 0,
            "min_area": 0,
            "max_area": 0,
            "median_area": 0.0,
            "mean_area": 0.0,
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_areas = areas[areas >= min_area]

    return {
        "total_components": int(len(areas)),
        "valid_components": int(len(valid_areas)),
        "min_area": int(areas.min()) if len(areas) else 0,
        "max_area": int(areas.max()) if len(areas) else 0,
        "median_area": float(np.median(valid_areas)) if len(valid_areas) else 0.0,
        "mean_area": float(np.mean(valid_areas)) if len(valid_areas) else 0.0,
    }
