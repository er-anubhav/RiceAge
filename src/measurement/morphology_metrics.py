"""Per-grain morphological feature extraction module."""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import math
import numpy as np
import cv2

from src.preprocessing.calibration import CalibrationResult
from src.preprocessing.border_clearing import is_boundary_grain


@dataclass
class GrainMorphology:
    """Geometric and morphological features of an individual rice grain."""
    grain_id: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    centroid_x: float
    centroid_y: float
    area_px: float
    perimeter_px: float
    length_px: float
    width_px: float
    aspect_ratio: float
    orientation_deg: float
    circularity: float
    solidity: float
    extent: float
    is_boundary: bool
    separation_status: str
    length_mm: Optional[float] = None
    width_mm: Optional[float] = None
    area_mm2: Optional[float] = None


def extract_grain_morphology(
    grain_id: int,
    grain_mask: np.ndarray,
    image_shape: Tuple[int, int],
    cleared_margins: Tuple[int, int, int, int] = (0, 0, 0, 0),
    calibration: Optional[CalibrationResult] = None,
    separation_status: str = "isolated_single"
) -> Optional[GrainMorphology]:
    """Calculate geometric metrics for a single grain binary mask.
    
    Args:
        grain_id: Unique integer identifier.
        grain_mask: 2D binary uint8 mask of the grain across full image (or local ROI).
        image_shape: (height, width) of the full image.
        cleared_margins: (top, bottom, left, right) margins cleared during preprocessing.
        calibration: CalibrationResult for physical metric conversion.
        separation_status: Status string from separation stage.
    """
    contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # Pick largest contour if multiple small disjoint fragments exist
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 5:
        return None

    perimeter = cv2.arcLength(c, closed=True)

    # Moments for centroid
    m = cv2.moments(c)
    if m["m00"] != 0:
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
    else:
        cx, cy = float(c[:, 0, 0].mean()), float(c[:, 0, 1].mean())

    # Upright bounding box
    x, y, w, h = cv2.boundingRect(c)

    # Minimum area bounding box (rotated rectangle for length and width)
    rect = cv2.minAreaRect(c)
    (_, (dim1, dim2), angle) = rect
    length = max(dim1, dim2)
    width = min(dim1, dim2)
    
    # Orientation: align major axis in degrees [0, 180)
    # OpenCV minAreaRect returns angle of first edge; normalize to major axis
    if dim1 < dim2:
        major_angle = (angle + 90.0) % 180.0
    else:
        major_angle = angle % 180.0

    # Fallback to fitted ellipse if contour has >= 5 points for smoother dimensions
    if len(c) >= 5:
        try:
            ellipse = cv2.fitEllipse(c)
            (_, (e_dim1, e_dim2), e_angle) = ellipse
            e_major = max(e_dim1, e_dim2)
            e_minor = min(e_dim1, e_dim2)
            # Use ellipse if dimensions are consistent with minAreaRect
            if 0.7 * length <= e_major <= 1.3 * length and e_minor > 0:
                length = e_major
                width = e_minor
                major_angle = e_angle % 180.0
        except Exception:
            pass

    aspect_ratio = float(length / (width + 1e-5))

    # Circularity = 4 * pi * Area / Perimeter^2 (1.0 for perfect circle)
    if perimeter > 0:
        circularity = float(min(1.0, (4.0 * math.pi * area) / (perimeter ** 2)))
    else:
        circularity = 0.0

    # Solidity = Area / ConvexHullArea
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area / (hull_area + 1e-5)) if hull_area > 0 else 0.0

    # Extent = Area / BoundingBoxArea
    bbox_area = w * h
    extent = float(area / (bbox_area + 1e-5)) if bbox_area > 0 else 0.0

    # Boundary check
    boundary_flag = is_boundary_grain(
        (x, y, w, h),
        image_shape=image_shape,
        cleared_margins=cleared_margins,
        boundary_margin_px=4
    )

    # Physical metrics if calibration verified
    length_mm = None
    width_mm = None
    area_mm2 = None
    if calibration and calibration.pixel_scale_mm is not None:
        px_scale = calibration.pixel_scale_mm
        length_mm = round(length * px_scale, 3)
        width_mm = round(width * px_scale, 3)
        area_mm2 = round(area * (px_scale ** 2), 4)

    return GrainMorphology(
        grain_id=grain_id,
        bbox_x=x,
        bbox_y=y,
        bbox_w=w,
        bbox_h=h,
        centroid_x=round(cx, 2),
        centroid_y=round(cy, 2),
        area_px=round(area, 1),
        perimeter_px=round(perimeter, 1),
        length_px=round(length, 2),
        width_px=round(width, 2),
        aspect_ratio=round(aspect_ratio, 2),
        orientation_deg=round(major_angle, 1),
        circularity=round(circularity, 3),
        solidity=round(solidity, 3),
        extent=round(extent, 3),
        is_boundary=boundary_flag,
        separation_status=separation_status,
        length_mm=length_mm,
        width_mm=width_mm,
        area_mm2=area_mm2
    )
