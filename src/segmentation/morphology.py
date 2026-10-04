"""Morphological cleanup, hole filling, and small noise removal module."""

import numpy as np
import cv2
from scipy import ndimage as ndi


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal cavities and hollow specular holes inside rice grains.
    
    Args:
        mask: uint8 binary mask (0 or 255).
        
    Returns:
        Hole-filled uint8 binary mask.
    """
    filled = ndi.binary_fill_holes(mask > 0)
    return filled.astype(np.uint8) * 255


def remove_small_objects(mask: np.ndarray, min_area_px: int = 40) -> np.ndarray:
    """Filter out noise, dust, and minute scanner artifacts smaller than min_area_px."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            clean_mask[labels == label] = 255
    return clean_mask


def cleanup_mask(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
    min_area_px: int = 40,
    fill_holes: bool = True
) -> np.ndarray:
    """Complete morphological cleanup pipeline.
    
    1. Fill small specular reflection holes inside grains.
    2. Morphological opening (ellipse kernel) to break single-pixel bridges and smooth borders.
    3. Small noise removal based on area threshold.
    """
    # 1. Fill holes
    if fill_holes:
        processed = fill_binary_holes(binary_mask)
    else:
        processed = binary_mask.copy()

    # 2. Morphological opening with small elliptic structuring element
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. Filter small noise
    if min_area_px > 0:
        processed = remove_small_objects(processed, min_area_px=min_area_px)

    return processed
