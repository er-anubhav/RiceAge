"""
Comprehensive Multi-Defect Archetype Breakdown across all 5,134 Damaged Grains.
Determines the percentage of damaged grains that fall into:
  1. Dark Pinhole / Insect Bore / Necrotic Punctures
  2. Mottled / High Surface Variance (Blemishes, Surface Chalk+Rot)
  3. Red / Brown / Amber Rot / Bran Remnants
  4. Contour Indentations / Micro-Cracks
"""

import sys
from pathlib import Path
import numpy as np
import cv2

def run_archetype_audit(sample_size: int = 1500):
    crops_dir = Path("dataset/crops_224/train/damaged")
    damaged_files = list(crops_dir.glob("*.jpg"))[:sample_size]

    print(f"Auditing {len(damaged_files)} Damaged Grains across all batch sources...\n")

    archetype_counts = {
        "Type 1: Dark Necrotic Spots (Insect Peck / Fungal Bore)": 0,
        "Type 2: High Internal Contrast / Mottled Texture": 0,
        "Type 3: Red / Brown / Amber Discoloration": 0,
        "Type 4: Surface Cracks / Fissure Gradients": 0,
        "At Least ONE of the 4 Above Defect Signatures": 0,
        "Clean / Borderline False Positives (Look almost Whole)": 0
    }

    for p in damaged_files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inner_mask = cv2.erode(otsu, kernel) > 0

        if np.sum(inner_mask) < 60:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[inner_mask]
        bgr = img[inner_mask]

        v = hsv[:, 2]
        s = hsv[:, 1]
        h = hsv[:, 0]
        mean_v = np.mean(v)
        std_v = np.std(v)

        # 1. Dark Pinhole: Any localized cluster of pixels < 65% of grain mean brightness
        has_dark_pinhole = np.sum(v < 0.65 * mean_v) >= 15

        # 2. High Texture Variance: std_v > 13.0
        has_high_mottled = std_v > 13.0

        # 3. Red / Brown Discoloration: Elevated red/green ratio and hue in warm amber range
        r = bgr[:, 2].astype(np.float32)
        g = bgr[:, 1].astype(np.float32)
        has_red_brown = np.mean((r / np.maximum(g, 1.0)) > 1.08) > 0.05

        # 4. Crack / Scratch Gradients: Sobel gradient variance
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)[inner_mask]
        has_crack_gradient = np.var(sobel) > 12.0

        is_defective = False
        if has_dark_pinhole:
            archetype_counts["Type 1: Dark Necrotic Spots (Insect Peck / Fungal Bore)"] += 1
            is_defective = True
        if has_high_mottled:
            archetype_counts["Type 2: High Internal Contrast / Mottled Texture"] += 1
            is_defective = True
        if has_red_brown:
            archetype_counts["Type 3: Red / Brown / Amber Discoloration"] += 1
            is_defective = True
        if has_crack_gradient:
            archetype_counts["Type 4: Surface Cracks / Fissure Gradients"] += 1
            is_defective = True

        if is_defective:
            archetype_counts["At Least ONE of the 4 Above Defect Signatures"] += 1
        else:
            archetype_counts["Clean / Borderline False Positives (Look almost Whole)"] += 1

    total = len(damaged_files)
    print(f"{'DEFECT ARCHETYPE':<60} | {'COUNT':<8} | {'PERCENTAGE'}")
    print("-" * 85)
    for k, count in archetype_counts.items():
        pct = (count / total) * 100.0
        print(f"{k:<60} | {count:<8} | {pct:6.2f}%")

if __name__ == "__main__":
    run_archetype_audit(sample_size=2000)
