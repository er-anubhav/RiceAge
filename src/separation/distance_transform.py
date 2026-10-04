"""Distance transform and local maxima peak detection module."""

from typing import Tuple, List, Optional
import numpy as np
import cv2
from skimage.feature import peak_local_max


def compute_distance_transform(
    binary_mask: np.ndarray,
    distance_type: int = cv2.DIST_L2,
    mask_size: int = 5
) -> np.ndarray:
    """Compute exact Euclidean distance transform of a binary mask.
    
    Args:
        binary_mask: uint8 array (0 or 255).
        distance_type: cv2.DIST_L2 for Euclidean.
        mask_size: 3 or 5.
        
    Returns:
        dist_map: float32 distance map.
    """
    binary_uint8 = (binary_mask > 0).astype(np.uint8)
    return cv2.distanceTransform(binary_uint8, distance_type, mask_size)


def detect_distance_peaks(
    distance_map: np.ndarray,
    min_distance: int = 7,
    threshold_rel: float = 0.35,
    threshold_abs: Optional[float] = None
) -> np.ndarray:
    """Detect local maxima peaks on distance map corresponding to grain centroids.
    
    Args:
        distance_map: float32 2D distance transform array.
        min_distance: minimum pixel distance between neighboring peaks.
        threshold_rel: minimum intensity of peaks relative to maximum distance.
        threshold_abs: absolute minimum distance threshold.
        
    Returns:
        coords: (N, 2) array of (row, col) coordinates of peaks.
    """
    if distance_map.max() == 0:
        return np.empty((0, 2), dtype=int)

    return peak_local_max(
        distance_map,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        threshold_abs=threshold_abs,
        exclude_border=False
    )
