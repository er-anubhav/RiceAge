"""
High-Resolution (224x224) Grain Crop Extractor for ResNet-50:
Extracts uncompressed, centered 224x224 crops from original scanner images
using ground-truth segmentation masks across train, val, and test splits.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

CLASS_NAMES = [
    "broken",
    "chalky",
    "damaged",
    "dehusked",
    "discoloured",
    "foreign_matter",
    "whole"
]

def make_square_crop_224(crop: np.ndarray, target_size: int = 224) -> np.ndarray:
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    max_dim = max(h, w)
    canvas = np.zeros((max_dim, max_dim, 3), dtype=crop.dtype)
    y_off = (max_dim - h) // 2
    x_off = (max_dim - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = crop

    return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4 if max_dim < target_size else cv2.INTER_AREA)

def extract_all(dataset_dir: str = "dataset/segment", out_dir: str = "dataset/crops_224"):
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)

    # In dataset/segment/data.yaml, class index mapping:
    # 0: whole, 1: broken, 2: chalky, 3: damaged, 4: discoloured, 5: dehusked, 6: foreign_matter
    IDX_TO_NAME = {
        0: "whole",
        1: "broken",
        2: "chalky",
        3: "damaged",
        4: "discoloured",
        5: "dehusked",
        6: "foreign_matter"
    }

    splits = ["train", "val", "test"]
    stats = {s: {c: 0 for c in CLASS_NAMES} for s in splits}

    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for c in CLASS_NAMES:
            (out_dir / split / c).mkdir(parents=True, exist_ok=True)

        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        print(f"Extracting {split} split at 224x224 ({len(img_files)} images)...")

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
                cls_name = IDX_TO_NAME.get(cls_id, None)
                if cls_name is None:
                    continue

                coords = [float(x) for x in parts[1:]]
                pts_x = [coords[k] * w_img for k in range(0, len(coords), 2)]
                pts_y = [coords[k] * h_img for k in range(1, len(coords), 2)]

                # Add a generous 6-pixel padding around grain boundary
                pad = 6
                x_min = max(0, int(min(pts_x)) - pad)
                y_min = max(0, int(min(pts_y)) - pad)
                x_max = min(w_img, int(max(pts_x)) + pad)
                y_max = min(h_img, int(max(pts_y)) + pad)

                if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                    continue

                raw_crop = img[y_min:y_max, x_min:x_max]
                sq_crop = make_square_crop_224(raw_crop, target_size=224)

                out_filename = f"{img_path.stem}_g{grain_idx:04d}.jpg"
                out_path = out_dir / split / cls_name / out_filename
                cv2.imwrite(str(out_path), sq_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                stats[split][cls_name] += 1

            if idx % 50 == 0 or idx == len(img_files):
                print(f"  Processed {idx}/{len(img_files)} images...")

    print("\nExtraction complete! High-Res 224x224 crops summary:")
    for s in splits:
        total = sum(stats[s].values())
        print(f"\n[{s.upper()}] Total: {total}")
        for c in sorted(CLASS_NAMES):
            print(f"  {c:15s}: {stats[s][c]}")

if __name__ == "__main__":
    extract_all()
