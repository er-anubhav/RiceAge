"""Connected component geometry and cluster categorization module."""

from enum import Enum
from typing import Tuple, Dict, Any
import numpy as np
import cv2


class ComponentCategory(str, Enum):
    SINGLE_GRAIN = "single_grain"
    TOUCHING_PAIR = "touching_pair"
    CLUSTER_COMPLEX = "cluster_complex"
    FRAGMENT_OR_NOISE = "fragment_or_noise"


class GrainStatus(str, Enum):
    ISOLATED_SINGLE = "isolated_single"
    SEPARATED_FROM_CLUSTER = "separated_from_cluster"
    UNDER_SEGMENTED_CLUSTER = "under_segmented_cluster"
    OVER_SEGMENTED_FRAGMENT = "over_segmented_fragment"
    BOUNDARY_TRUNCATED = "boundary_truncated"


def evaluate_component_geometry(
    component_mask: np.ndarray,
    solidity_thresh: float = 0.92,
    single_area_range: Tuple[int, int] = (200, 650),
    pair_area_range: Tuple[int, int] = (650, 1200)
) -> Dict[str, Any]:
    """Analyze the geometry of a single binary component to detect touching grains.
    
    Returns:
        dict with area, perimeter, solidity, aspect_ratio, category.
    """
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "area": 0,
            "solidity": 0.0,
            "category": ComponentCategory.FRAGMENT_OR_NOISE
        }

    c = contours[0]
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-5) if hull_area > 0 else 0.0

    # Minimum area bounding box
    rect = cv2.minAreaRect(c)
    (_, (dim1, dim2), angle) = rect
    major = max(dim1, dim2)
    minor = min(dim1, dim2)
    aspect_ratio = major / (minor + 1e-5)

    if area < single_area_range[0]:
        category = ComponentCategory.FRAGMENT_OR_NOISE
    elif area <= single_area_range[1] and solidity >= solidity_thresh:
        category = ComponentCategory.SINGLE_GRAIN
    elif area <= pair_area_range[1] or (solidity < solidity_thresh and area <= pair_area_range[1]):
        category = ComponentCategory.TOUCHING_PAIR
    else:
        category = ComponentCategory.CLUSTER_COMPLEX

    return {
        "area": area,
        "perimeter": perimeter,
        "solidity": solidity,
        "major_axis": major,
        "minor_axis": minor,
        "aspect_ratio": aspect_ratio,
        "category": category
    }
