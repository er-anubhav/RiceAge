"""
Grain Crop Extractor:
Extracts high-resolution, centered grain crops (128x128) from original scanner images
using the ground-truth segmentation masks, organized into train/val/test splits.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

CLASS_NAMES = [
    "whole",
    "broken",
    "chalky",
    "damaged",
    "discoloured",
    "dehusked",
    "foreign_matter"
]

def make_square_crop(crop: np.ndarray, target_size: int = 128) -> np.ndarray:
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    max_dim = max(h, w)
    # Create black square canvas
    canvas = np.zeros((max_dim, max_dim, 3), dtype=crop.dtype)
    y_off = (max_dim - h) // 2
    x_off = (max_dim - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = crop

    # Resize to target
    return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)

def extract_crops(dataset_dir: str = "dataset/segment", out_dir: str = "dataset/crops"):
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)

    splits = ["train", "val", "test"]
    stats = {s: {c: 0 for c in CLASS_NAMES} for s in splits}

    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Skipping {split}: directory not found")
            continue

        for c in CLASS_NAMES:
            (out_dir / split / c).mkdir(parents=True, exist_ok=True)

        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        print(f"Processing {split} split ({len(img_files)} images)...")

        for idx, img_path in enumerate(img_files, 1):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h_img, w_img = img.shape[:2]

            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for grain_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                if cls_id >= len(CLASS_NAMES):
                    continue
                cls_name = CLASS_NAMES[cls_id]

                coords = [float(x) for x in parts[1:]]
                # Coords are normalized x1, y1, x2, y2...
                pts_x = [coords[k] * w_img for k in range(0, len(coords), 2)]
                pts_y = [coords[k] * h_img for k in range(1, len(coords), 2)]

                x_min = max(0, int(min(pts_x)) - 4)
                y_min = max(0, int(min(pts_y)) - 4)
                x_max = min(w_img, int(max(pts_x)) + 4)
                y_max = min(h_img, int(max(pts_y)) + 4)

                if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                    continue

                raw_crop = img[y_min:y_max, x_min:x_max]
                sq_crop = make_square_crop(raw_crop, target_size=128)

                out_filename = f"{img_path.stem}_g{grain_idx:04d}.jpg"
                out_path = out_dir / split / cls_name / out_filename
                cv2.imwrite(str(out_path), sq_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                stats[split][cls_name] += 1

            if idx % 50 == 0 or idx == len(img_files):
                print(f"  Processed {idx}/{len(img_files)} images...")

    print("\nExtraction complete! Summary of grain crops created:")
    for s in splits:
        total = sum(stats[s].values())
        print(f"\n[{s.upper()}] Total: {total}")
        for c in CLASS_NAMES:
            print(f"  {c:15s}: {stats[s][c]}")

if __name__ == "__main__":
    extract_crops()
