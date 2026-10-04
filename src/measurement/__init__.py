# Measurement package for Rice Grain Quality Analyzer.

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import cv2

from src.preprocessing.calibration import CalibrationResult
from .morphology_metrics import GrainMorphology, extract_grain_morphology
from .color_metrics import GrainColor, extract_grain_color


def measure_all_grains(
    instance_mask: np.ndarray,
    image_rgb: np.ndarray,
    grain_diagnostics: Dict[int, Dict[str, Any]],
    cleared_margins: Tuple[int, int, int, int] = (0, 0, 0, 0),
    calibration: Optional[CalibrationResult] = None,
    min_area_px: int = 40
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    # Calculate comprehensive morphology and color metrics for all detected grains. Returns: df: pandas DataFrame containing all grain measurements. records: list of dictionaries for each grain.
    img_h, img_w = instance_mask.shape[:2]
    records = []
    
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]

    for gid in unique_ids:
        diag = grain_diagnostics.get(int(gid), {})
        status = diag.get("status", "unknown")

        # Extract sub-mask
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

        # Calibration columns if verified
        if morph.length_mm is not None:
            row["length_mm"] = morph.length_mm
            row["width_mm"] = morph.width_mm
            row["area_mm2"] = morph.area_mm2

        # Color columns
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


__all__ = [
    "GrainMorphology",
    "extract_grain_morphology",
    "GrainColor",
    "extract_grain_color",
    "measure_all_grains",
]
