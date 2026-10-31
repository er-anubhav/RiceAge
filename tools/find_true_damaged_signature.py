"""
Find the True Optical & Morphological Signature of Damaged Rice:
Uses feature importance on 2,000 damaged vs 2,000 whole grains
to discover the exact separating boundary.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

def extract_deep_signature(img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 100:
        return None

    # 1. MORPHOLOGY: Solidity & Extent
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / max(hull_area, 1.0) # Nibbles, chunks missing, broken dents reduce solidity

    perimeter = cv2.arcLength(c, True)
    compactness = (perimeter ** 2) / max(area, 1.0) # Smooth oval = ~16, jagged jagged nibbled = 20-30+

    # 2. INTERNAL DEFECT: Erode to inspect only the central meat, ignoring 3D edge shadows
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner = cv2.erode(otsu, kernel) > 0
    if np.sum(inner) < 40:
        inner = otsu > 0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[inner]
    bgr = img[inner]

    v = hsv[:, 2].astype(np.float32)
    s = hsv[:, 1].astype(np.float32)
    h = hsv[:, 0].astype(np.float32)

    median_v = np.median(v)
    mean_v = np.mean(v)

    # Ratio of localized deep black spots inside the grain meat
    deep_black_ratio = np.sum(v < 0.50 * median_v) / float(len(v))

    # Yellowing / Browning shift: high saturation in yellow/amber hue range (H in 10-30, S > 50)
    amber_discolor_ratio = np.sum((h >= 10) & (h <= 30) & (s > 45)) / float(len(v))

    # Red/Green and Red/Blue ratios
    r_val = bgr[:, 2].astype(np.float32)
    g_val = bgr[:, 1].astype(np.float32)
    b_val = bgr[:, 0].astype(np.float32)
    rg_ratio = np.mean(r_val / np.maximum(g_val, 1.0))
    rb_ratio = np.mean(r_val / np.maximum(b_val, 1.0))

    # Laplacial sharpness / fissure crack intensity
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)[inner]
    fissure_score = float(np.var(laplacian))

    return [
        solidity,               # 0: Shape completeness
        compactness,            # 1: Contour jaggedness
        deep_black_ratio,       # 2: Pinhole insect marks
        amber_discolor_ratio,   # 3: Fungal/heat rot
        rg_ratio,               # 4: Red color bias
        rb_ratio,               # 5: Red vs Blue
        fissure_score,          # 6: Cracks & fissures
        mean_v,                 # 7: Brightness
        np.std(v)               # 8: Mottled variance
    ]

def main():
    crops_dir = Path("dataset/crops_224/train")
    damaged_files = list((crops_dir / "damaged").glob("*.jpg"))[:1500]
    whole_files = list((crops_dir / "whole").glob("*.jpg"))[:1500]

    print(f"Extracting signatures from {len(damaged_files)} Damaged and {len(whole_files)} Whole grains...")

    X = []
    y = []

    for f in damaged_files:
        sig = extract_deep_signature(f)
        if sig:
            X.append(sig)
            y.append(1) # 1 = Damaged

    for f in whole_files:
        sig = extract_deep_signature(f)
        if sig:
            X.append(sig)
            y.append(0) # 0 = Whole

    X = np.array(X)
    y = np.array(y)

    feature_names = [
        "Solidity (Shape Completeness)",
        "Compactness (Jagged Nibbled Border)",
        "Deep Black Pinhole Ratio (<50% mean)",
        "Amber / Fungal Rot Ratio",
        "Red / Green Ratio",
        "Red / Blue Ratio",
        "Fissure / Crack Variance",
        "Mean Brightness",
        "Internal Texture Variance"
    ]

    print(f"\nTotal Valid Samples: {len(y)} ({np.sum(y==1)} Damaged, {np.sum(y==0)} Whole)\n")

    print(f"{'FEATURE':<38} | {'DAMAGED (Mean)':<15} | {'WHOLE (Mean)':<15} | {'DISCRIMINATION POWER'}")
    print("-" * 95)

    for i, name in enumerate(feature_names):
        d_mean = np.mean(X[y==1, i])
        w_mean = np.mean(X[y==0, i])
        d_std = np.std(X[y==1, i])
        w_std = np.std(X[y==0, i])
        # Cohen's d effect size: |mean1 - mean2| / pooled_std
        pooled_std = np.sqrt((d_std**2 + w_std**2) / 2.0)
        cohens_d = abs(d_mean - w_mean) / max(pooled_std, 1e-4)
        
        rating = "STRONG" if cohens_d > 0.5 else ("MEDIUM" if cohens_d > 0.25 else "WEAK")
        print(f"{name:<38} | {d_mean:15.3f} | {w_mean:15.3f} | {cohens_d:6.2f} ({rating})")

if __name__ == "__main__":
    main()
