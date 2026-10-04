# Image preprocessing: scanner calibration, bezel clearing, boundary checks, and denoising.

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import os
import numpy as np
import cv2
from PIL import Image
from PIL.ExifTags import TAGS


@dataclass
class CalibrationResult:
    # Stores verified or estimated pixel-to-metric calibration.
    is_verified: bool
    effective_dpi: Optional[float]
    pixel_scale_mm: Optional[float]
    pixels_per_mm: Optional[float]
    source_info: str
    scale_factor_to_raw: float = 1.0


# Inspect and extract all available metadata from the image file.
def inspect_metadata(image_path: str) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as img:
        width, height = img.size
        info = dict(img.info)
        img_format = img.format
        mode = img.mode

        exif_dict = {}
        try:
            exif_raw = img.getexif()
            if exif_raw:
                for tag_id, value in exif_raw.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    exif_dict[tag_name] = value
        except Exception:
            pass

    return {
        "file_path": image_path,
        "width_px": width,
        "height_px": height,
        "format": img_format,
        "mode": mode,
        "info": info,
        "dpi": info.get("dpi"),
        "jfif_density": info.get("jfif_density"),
        "jfif_unit": info.get("jfif_unit"),
        "exif": exif_dict,
    }


# Determine the physical calibration (mm/pixel) for an image.
def resolve_calibration(
    image_path: str,
    raw_reference_path: Optional[str] = None,
    assumed_dpi: Optional[float] = None
) -> CalibrationResult:
    meta = inspect_metadata(image_path)
    w, h = meta["width_px"], meta["height_px"]

    dpi = meta.get("dpi")
    if dpi and isinstance(dpi, (tuple, list)) and len(dpi) >= 2 and dpi[0] is not None and dpi[0] > 0:
        dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
        if abs(dpi_x - 1200.0) < 1.0 and abs(dpi_y - 1200.0) < 1.0:
            if w >= 9000 and h >= 12000:
                scale_mm = 25.4 / 1200.0
                return CalibrationResult(
                    is_verified=True,
                    effective_dpi=1200.0,
                    pixel_scale_mm=scale_mm,
                    pixels_per_mm=1200.0 / 25.4,
                    source_info="Direct verified JFIF 1200 DPI metadata matching flatbed A4 scan dimensions",
                    scale_factor_to_raw=1.0
                )

    if raw_reference_path and os.path.exists(raw_reference_path):
        raw_meta = inspect_metadata(raw_reference_path)
        raw_w, raw_h = raw_meta["width_px"], raw_meta["height_px"]
        raw_dpi = raw_meta.get("dpi")
        if raw_dpi and raw_dpi[0] == 1200:
            raw_aspect = raw_w / raw_h
            curr_aspect = w / h
            if abs(raw_aspect - curr_aspect) < 0.01:
                scale_factor_x = raw_w / w
                scale_factor_y = raw_h / h
                avg_scale = (scale_factor_x + scale_factor_y) / 2.0
                effective_dpi = 1200.0 / avg_scale
                scale_mm = (25.4 / 1200.0) * avg_scale
                return CalibrationResult(
                    is_verified=True,
                    effective_dpi=effective_dpi,
                    pixel_scale_mm=scale_mm,
                    pixels_per_mm=effective_dpi / 25.4,
                    source_info=(
                        f"Derived from verified 1200 DPI raw scan ({raw_w}x{raw_h}) "
                        f"with downsampling factor {avg_scale:.4f}x (effective {effective_dpi:.2f} DPI)"
                    ),
                    scale_factor_to_raw=avg_scale
                )

    if assumed_dpi is not None and assumed_dpi > 0:
        scale_mm = 25.4 / assumed_dpi
        return CalibrationResult(
            is_verified=False,
            effective_dpi=assumed_dpi,
            pixel_scale_mm=scale_mm,
            pixels_per_mm=assumed_dpi / 25.4,
            source_info=f"Unverified user-assumed DPI: {assumed_dpi}",
            scale_factor_to_raw=1.0
        )

    return CalibrationResult(
        is_verified=False,
        effective_dpi=None,
        pixel_scale_mm=None,
        pixels_per_mm=None,
        source_info="No verified DPI metadata or reference found. Reporting in pixels only.",
        scale_factor_to_raw=1.0
    )


def _scan_edge_margin(line_means: np.ndarray, max_search: int, saturation_thresh: int) -> int:
    margin = 0
    for idx in range(min(len(line_means), max_search)):
        if line_means[idx] >= saturation_thresh:
            margin = idx + 1
        else:
            break
    return margin


# Detect saturated scanner glass bezel lines along image borders.
def detect_scanner_margins(
    gray_image: np.ndarray,
    saturation_thresh: int = 240,
    max_search_ratio: float = 0.05
) -> Tuple[int, int, int, int]:
    h, w = gray_image.shape[:2]
    max_h = int(h * max_search_ratio)
    max_w = int(w * max_search_ratio)

    row_means = gray_image.mean(axis=1)
    col_means = gray_image.mean(axis=0)

    top = _scan_edge_margin(row_means, max_h, saturation_thresh)
    bottom = _scan_edge_margin(row_means[::-1], max_h, saturation_thresh)
    left = _scan_edge_margin(col_means, max_w, saturation_thresh)
    right = _scan_edge_margin(col_means[::-1], max_w, saturation_thresh)

    return top, bottom, left, right


# Detect and blank out saturated scanner edges.
def clear_scanner_bezel(
    image: np.ndarray,
    saturation_thresh: int = 240,
    safety_padding: int = 2
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    top, bottom, left, right = detect_scanner_margins(gray, saturation_thresh=saturation_thresh)

    top = top + safety_padding if top > 0 else 0
    bottom = bottom + safety_padding if bottom > 0 else 0
    left = left + safety_padding if left > 0 else 0
    right = right + safety_padding if right > 0 else 0

    cleaned = image.copy()
    h, w = cleaned.shape[:2]

    if top > 0:
        cleaned[:top] = 0
    if bottom > 0:
        cleaned[h - bottom:] = 0
    if left > 0:
        cleaned[:, :left] = 0
    if right > 0:
        cleaned[:, w - right:] = 0

    return cleaned, (top, bottom, left, right)


# Check if a grain touches or lies within boundary_margin_px of active image edges.
def is_boundary_grain(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    cleared_margins: Tuple[int, int, int, int] = (0, 0, 0, 0),
    boundary_margin_px: int = 4
) -> bool:
    x, y, bw, bh = bbox
    img_h, img_w = image_shape[:2]
    top_m, bot_m, left_m, right_m = cleared_margins

    min_x = left_m + boundary_margin_px
    max_x = img_w - right_m - boundary_margin_px
    min_y = top_m + boundary_margin_px
    max_y = img_h - bot_m - boundary_margin_px

    return x <= min_x or (x + bw) >= max_x or y <= min_y or (y + bh) >= max_y


# Convert image to grayscale and apply mild Gaussian filtering.
def preprocess_image(
    image: np.ndarray,
    gaussian_kernel: int = 5,
    gaussian_sigma: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gaussian_kernel > 1:
        if gaussian_kernel % 2 == 0:
            gaussian_kernel += 1
        smoothed = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), gaussian_sigma)
    else:
        smoothed = gray.copy()

    return gray, smoothed
