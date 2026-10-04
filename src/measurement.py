# Per-grain measurement: morphology extraction, color metrics, and aggregation.

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import math
import numpy as np
import pandas as pd
import cv2

from src.preprocessing import CalibrationResult, is_boundary_grain


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


@dataclass
class GrainColor:
    # Photometric and color properties of an individual grain.
    grain_id: int
    mean_r: float
    mean_g: float
    mean_b: float
    mean_intensity: float
    std_intensity: float
    highlight_ratio: float
    uniformity: float


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

    return float(length), float(width), float(major_angle)


# Extract comprehensive morphology metrics for a single segmented grain mask.
def extract_grain_morphology(
    grain_id: int,
    grain_mask: np.ndarray,
    image_shape: Tuple[int, int],
    cleared_margins: Tuple[int, int, int, int] = (0, 0, 0, 0),
    calibration: Optional[CalibrationResult] = None,
    separation_status: str = "unknown"
) -> Optional[GrainMorphology]:
    contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area <= 0:
        return None

    peri = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    cx, cy = _compute_centroid(c)
    length, width, angle = _compute_oriented_dimensions(c)
    ar = float(length / (width + 1e-5))

    circ = float(4.0 * math.pi * area / ((peri + 1e-5) ** 2))
    circ = min(1.0, max(0.0, circ))

    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area / (hull_area + 1e-5))
    solidity = min(1.0, max(0.0, solidity))

    extent = float(area / (w * h + 1e-5))
    boundary_flag = is_boundary_grain((x, y, w, h), image_shape, cleared_margins)

    length_mm, width_mm, area_mm2 = None, None, None
    if calibration and calibration.pixel_scale_mm is not None:
        scale = calibration.pixel_scale_mm
        length_mm = round(length * scale, 3)
        width_mm = round(width * scale, 3)
        area_mm2 = round(area * (scale ** 2), 3)

    return GrainMorphology(
        grain_id=grain_id,
        bbox_x=int(x),
        bbox_y=int(y),
        bbox_w=int(w),
        bbox_h=int(h),
        centroid_x=round(cx, 2),
        centroid_y=round(cy, 2),
        area_px=round(area, 1),
        perimeter_px=round(peri, 2),
        length_px=round(length, 2),
        width_px=round(width, 2),
        aspect_ratio=round(ar, 3),
        orientation_deg=round(angle, 1),
        circularity=round(circ, 3),
        solidity=round(solidity, 3),
        extent=round(extent, 3),
        is_boundary=boundary_flag,
        separation_status=separation_status,
        length_mm=length_mm,
        width_mm=width_mm,
        area_mm2=area_mm2
    )


# Calculate color, brightness, and surface reflection metrics for a single grain.
def extract_grain_color(
    grain_id: int,
    rgb_image: np.ndarray,
    grain_mask: np.ndarray,
    specular_thresh: int = 240
) -> Optional[GrainColor]:
    grain_pixels = grain_mask > 0
    if not np.any(grain_pixels):
        return None

    r_vals = rgb_image[:, :, 0][grain_pixels]
    g_vals = rgb_image[:, :, 1][grain_pixels]
    b_vals = rgb_image[:, :, 2][grain_pixels]

    mean_r = float(np.mean(r_vals))
    mean_g = float(np.mean(g_vals))
    mean_b = float(np.mean(b_vals))

    gray_vals = 0.299 * r_vals + 0.587 * g_vals + 0.114 * b_vals
    mean_int = float(np.mean(gray_vals))
    std_int = float(np.std(gray_vals))

    highlight_count = np.sum(gray_vals >= specular_thresh)
    highlight_ratio = float(highlight_count / len(gray_vals))

    cov = std_int / (mean_int + 1e-5)
    uniformity = float(max(0.0, 1.0 - cov))

    return GrainColor(
        grain_id=grain_id,
        mean_r=round(mean_r, 1),
        mean_g=round(mean_g, 1),
        mean_b=round(mean_b, 1),
        mean_intensity=round(mean_int, 1),
        std_intensity=round(std_int, 1),
        highlight_ratio=round(highlight_ratio, 4),
        uniformity=round(uniformity, 3)
    )


# Calculate comprehensive morphology and color metrics for all detected grains.
def measure_all_grains(
    instance_mask: np.ndarray,
    image_rgb: np.ndarray,
    grain_diagnostics: Dict[int, Dict[str, Any]],
    cleared_margins: Tuple[int, int, int, int] = (0, 0, 0, 0),
    calibration: Optional[CalibrationResult] = None,
    min_area_px: int = 40
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    img_h, img_w = instance_mask.shape[:2]
    records = []
    
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]

    for gid in unique_ids:
        diag = grain_diagnostics.get(int(gid), {})
        status = diag.get("status", "unknown")

        grain_pixel_mask = (instance_mask == gid).astype(np.uint8) * 255
        
        morph = extract_grain_morphology(
            grain_id=int(gid),
            grain_mask=grain_pixel_mask,
            image_shape=(img_h, img_w),
            cleared_margins=cleared_margins,
            calibration=calibration,
            separation_status=status
        )

        if morph is None or morph.area_px < min_area_px:
            continue

        color = extract_grain_color(
            grain_id=int(gid),
            rgb_image=image_rgb,
            grain_mask=grain_pixel_mask
        )

        row = {
            "grain_id": morph.grain_id,
            "x": morph.bbox_x,
            "y": morph.bbox_y,
            "width": morph.bbox_w,
            "height": morph.bbox_h,
            "centroid_x": morph.centroid_x,
            "centroid_y": morph.centroid_y,
            "area_px": morph.area_px,
            "perimeter_px": morph.perimeter_px,
            "length_px": morph.length_px,
            "width_px": morph.width_px,
            "aspect_ratio": morph.aspect_ratio,
            "orientation_deg": morph.orientation_deg,
            "circularity": morph.circularity,
            "solidity": morph.solidity,
            "extent": morph.extent,
            "is_boundary": morph.is_boundary,
            "separation_status": morph.separation_status,
        }

        if morph.length_mm is not None:
            row["length_mm"] = morph.length_mm
            row["width_mm"] = morph.width_mm
            row["area_mm2"] = morph.area_mm2

        if color:
            row["mean_r"] = color.mean_r
            row["mean_g"] = color.mean_g
            row["mean_b"] = color.mean_b
            row["mean_intensity"] = color.mean_intensity
            row["std_intensity"] = color.std_intensity
            row["highlight_ratio"] = color.highlight_ratio
            row["uniformity"] = color.uniformity

        records.append(row)

    df = pd.DataFrame(records)
    return df, records
