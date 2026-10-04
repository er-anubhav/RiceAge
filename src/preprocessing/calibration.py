# Image calibration and physical dimension resolution module.

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
from PIL import Image
from PIL.ExifTags import TAGS


@dataclass
class CalibrationResult:
    # Stores verified or estimated pixel-to-metric calibration.
    is_verified: bool
    effective_dpi: Optional[float]
    pixel_scale_mm: Optional[float]  # millimeters per pixel
    pixels_per_mm: Optional[float]   # pixels per millimeter
    source_info: str
    scale_factor_to_raw: float = 1.0


def inspect_metadata(image_path: str) -> Dict[str, Any]:
    # Inspect and extract all available metadata from the image file. Returns a dictionary of dimensions, format, DPI tags, JFIF, and EXIF fields.
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as img:
        width, height = img.size
        info = dict(img.info)
        img_format = img.format
        mode = img.mode

        # Extract EXIF tags if present
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


def resolve_calibration(
    image_path: str,
    raw_reference_path: Optional[str] = None,
    assumed_dpi: Optional[float] = None
) -> CalibrationResult:
    # Determine the physical calibration (mm/pixel) for an image. Strict rules: - Never assume 1200 DPI unless verified in metadata or derived via verified scale factor. - If image contains authentic 1200 DPI metadata: 1 inch = 25.4 mm -> 1 px = 25.4 / 1200 = 0.0211667 mm. - If image is a downsampled representation of a verified 1200 DPI raw scan, compute the exact relative downsampling factor and determine the effective DPI / scale. - If metadata has no verified DPI and no verified reference is provided, return is_verified=False.
    meta = inspect_metadata(image_path)
    w, h = meta["width_px"], meta["height_px"]

    # 1. Direct DPI metadata in image file
    dpi = meta.get("dpi")
    if dpi and isinstance(dpi, (tuple, list)) and len(dpi) >= 2 and dpi[0] is not None and dpi[0] > 0:
        dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
        # Verify if dimensions match a standard flatbed scan (e.g., A4 @ 1200 DPI is ~10200x14028)
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

    # 2. Check if this is a standardized downsample of a known raw reference scan
    if raw_reference_path and os.path.exists(raw_reference_path):
        raw_meta = inspect_metadata(raw_reference_path)
        raw_w, raw_h = raw_meta["width_px"], raw_meta["height_px"]
        raw_dpi = raw_meta.get("dpi")
        if raw_dpi and raw_dpi[0] == 1200:
            # Check aspect ratio consistency
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

    # 3. Fallback to assumed DPI if explicitly supplied
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

    # 4. Uncalibrated
    return CalibrationResult(
        is_verified=False,
        effective_dpi=None,
        pixel_scale_mm=None,
        pixels_per_mm=None,
        source_info="No verified DPI metadata or reference found. Reporting in pixels only.",
        scale_factor_to_raw=1.0
    )
