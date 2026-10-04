"""Data exporter for grain crops, masks, measurements CSV, and summary JSON."""

import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import cv2


def export_crops_and_masks(
    original_bgr: np.ndarray,
    instance_mask: np.ndarray,
    measurements: List[Dict[str, Any]],
    crops_dir: str = "outputs/crops",
    masks_dir: str = "outputs/masks",
    padding_px: int = 4,
    max_count: int = 1000
) -> Tuple[int, int]:
    """Save individual bounding-box crops and binary masks for each detected grain."""
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    img_h, img_w = original_bgr.shape[:2]
    saved_crops = 0
    saved_masks = 0

    for idx, row in enumerate(measurements[:max_count]):
        gid = row["grain_id"]
        x = row["x"]
        y = row["y"]
        w = row["width"]
        h = row["height"]

        # Apply padding
        x1 = max(0, x - padding_px)
        y1 = max(0, y - padding_px)
        x2 = min(img_w, x + w + padding_px)
        y2 = min(img_h, y + h + padding_px)

        # Crop original image
        crop_img = original_bgr[y1:y2, x1:x2]
        
        # Crop mask (only pixels belonging to this grain)
        sub_instance = instance_mask[y1:y2, x1:x2]
        crop_mask = ((sub_instance == gid) * 255).astype(np.uint8)

        # File paths formatted as grain_XXXX.png
        crop_name = f"grain_{gid:04d}.png"
        crop_path = os.path.join(crops_dir, crop_name)
        mask_path = os.path.join(masks_dir, crop_name)

        cv2.imwrite(crop_path, crop_img)
        cv2.imwrite(mask_path, crop_mask)

        saved_crops += 1
        saved_masks += 1

    return saved_crops, saved_masks


def export_measurements_csv(
    df: pd.DataFrame,
    output_path: str = "outputs/grain_measurements.csv"
) -> str:
    """Save grain measurements to CSV with proper header formatting."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def export_summary_json(
    summary_dict: Dict[str, Any],
    output_path: str = "outputs/summary.json"
) -> str:
    """Save pipeline summary statistics to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def convert_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, default=convert_types)

    return output_path
