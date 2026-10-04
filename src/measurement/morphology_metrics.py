# Per-grain morphological feature extraction module.

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import math
import numpy as np
import cv2

from src.preprocessing.calibration import CalibrationResult
from src.preprocessing.border_clearing import is_boundary_grain


@dataclass
class GrainMorphology:
    # Geometric and morphological features of an individual rice grain.
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


def _compute_centroid(contour: np.ndarray) -> Tuple[float, float]:
    m = cv2.moments(contour)
    if m["m00"] != 0:
        return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    return float(contour[:, 0, 0].mean()), float(contour[:, 0, 1].mean())


def _compute_oriented_dimensions(contour: np.ndarray) -> Tuple[float, float, float]:
    rect = cv2.minAreaRect(contour)
    (_, (dim1, dim2), angle) = rect
    length = max(dim1, dim2)
    width = min(dim1, dim2)
    major_angle = (angle + 90.0) % 180.0 if dim1 < dim2 else angle % 180.0

    if len(contour) >= 5:
        try:
            (_, (e_dim1, e_dim2), e_angle) = cv2.fitEllipse(contour)
            e_major, e_minor = max(e_dim1, e_dim2), min(e_dim1, e_dim2)
            if 0.7 * length <= e_major <= 1.3 * length and e_minor > 0:
                length, width, major_angle = e_major, e_minor, e_angle % 180.0
        except cv2.error:
            pass

    return length, width, major_angle


def _compute_shape_factors(
    contour: np.ndarray,
    area: float,
    perimeter: float,
    bbox_w: int,
    bbox_h: int
) -> Tuple[float, float, float]:
    circularity = float(min(1.0, (4.0 * math.pi * area) / (perimeter ** 2))) if perimeter > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area / (hull_area + 1e-5)) if hull_area > 0 else 0.0

    bbox_area = bbox_w * bbox_h
    extent = float(area / (bbox_area + 1e-5)) if bbox_area > 0 else 0.0

    return circularity, solidity, extent


def extract_grain_morphology(
    grain_id: int,
    grain_mask: np.ndarray,
    image_shape: Tuple[int, int],
    cleared_margins: Tuple[int, int, int, int] = (0, 0, 0, 0),
    calibration: Optional[CalibrationResult] = None,
    separation_status: str = "isolated_single"
) -> Optional[GrainMorphology]:
    # Calculate geometric metrics for a single grain binary mask.
    contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 5:
        return None

    perimeter = cv2.arcLength(c, closed=True)
    cx, cy = _compute_centroid(c)
    x, y, w, h = cv2.boundingRect(c)
    length, width, major_angle = _compute_oriented_dimensions(c)
    aspect_ratio = float(length / (width + 1e-5))

    circularity, solidity, extent = _compute_shape_factors(c, area, perimeter, w, h)

    boundary_flag = is_boundary_grain(
        (x, y, w, h),
        image_shape=image_shape,
        cleared_margins=cleared_margins,
        boundary_margin_px=4
    )

    length_mm = None
    width_mm = None
    area_mm2 = None
    if calibration and calibration.pixel_scale_mm is not None:
        scale = calibration.pixel_scale_mm
        length_mm = round(length * scale, 3)
        width_mm = round(width * scale, 3)
        area_mm2 = round(area * (scale ** 2), 4)

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

