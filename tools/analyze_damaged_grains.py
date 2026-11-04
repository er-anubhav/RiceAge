"""
Damaged Grain Forensic Analysis:
Compares hundreds of ground-truth 'damaged' grains against 'whole' grains
across Color (HSV, LAB), Texture Variance, Dark Necrotic Spots, and Geometry
to discover the universal optical signature of damaged rice.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

def analyze_crop(img_path: Path):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Segment grain from black canvas
    # Background in 224x224 crops is pure black (0, 0, 0)
    grain_mask = (bgr[:, :, 0] > 10) | (bgr[:, :, 1] > 10) | (bgr[:, :, 2] > 10)
    total_px = np.sum(grain_mask)
    if total_px < 100:
        return None

    # Erode slightly (2 px) to remove border edge transitions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_mask = cv2.erode(grain_mask.astype(np.uint8), kernel) > 0
    if np.sum(eroded_mask) < 80:
        eroded_mask = grain_mask

    grain_gray = gray[eroded_mask]
    grain_v = hsv[:, :, 2][eroded_mask]
    grain_s = hsv[:, :, 1][eroded_mask]
    grain_h = hsv[:, :, 0][eroded_mask]
    grain_l = lab[:, :, 0][eroded_mask]
    grain_a = lab[:, :, 1][eroded_mask]
    grain_b = lab[:, :, 2][eroded_mask]

    # Metrics
    mean_val = np.mean(grain_v)
    std_val = np.std(grain_v)          # Texture variation (spots cause high std)
    min_val = np.min(grain_v)          # Darkest spot inside grain
    pct_5th_val = np.percentile(grain_v, 5) # 5th percentile brightness (local dark spot)

    # Dark necrotic spots (pinholes, fungus, insect punctures)
    dark_spot_ratio_60 = np.sum(grain_v < 60) / float(len(grain_v))
    dark_spot_ratio_80 = np.sum(grain_v < 80) / float(len(grain_v))

    # Red/Brown/Amber bran/rot spots (LAB A channel > 132 or HSV Hue in brown range)
    red_brown_ratio = np.sum((grain_a > 132) | ((grain_h >= 5) & (grain_h <= 25) & (grain_s > 60))) / float(len(grain_v))

    # High contrast gradient (edge detection inside the grain)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)[eroded_mask]
    internal_edge_density = np.mean(grad_mag)

    return {
        "mean_v": mean_val,
        "std_v": std_val,
        "min_v": min_val,
        "pct_5th_v": pct_5th_val,
        "dark_spot_60_pct": dark_spot_ratio_60 * 100.0,
        "dark_spot_80_pct": dark_spot_ratio_80 * 100.0,
        "red_brown_pct": red_brown_ratio * 100.0,
        "edge_density": internal_edge_density,
        "mean_saturation": np.mean(grain_s),
    }

def run_forensic_comparison(sample_size: int = 500):
    crops_dir = Path("dataset/crops_224/train")
    damaged_dir = crops_dir / "damaged"
    whole_dir = crops_dir / "whole"

    damaged_files = list(damaged_dir.glob("*.jpg"))[:sample_size]
    whole_files = list(whole_dir.glob("*.jpg"))[:sample_size]

    print(f"Analyzing {len(damaged_files)} DAMAGED grains vs {len(whole_files)} WHOLE grains...\n")

    damaged_metrics = []
    for f in damaged_files:
        m = analyze_crop(f)
        if m:
            damaged_metrics.append(m)

    whole_metrics = []
    for f in whole_files:
        m = analyze_crop(f)
        if m:
            whole_metrics.append(m)

    def stats(data, key):
        vals = [d[key] for d in data]
        return np.mean(vals), np.median(vals), np.percentile(vals, 25), np.percentile(vals, 75)

    keys = [
        ("Dark Spot Intensity (5th %ile V)", "pct_5th_v", "Lower = Darker localized spots present"),
        ("Minimum Pixel Brightness (Min V)", "min_v", "Lowest single pixel in grain"),
        ("Internal Contrast / Texture (Std V)", "std_v", "Higher = Uneven mottled surface / blemishes"),
        ("Internal Edge / Scratch Density", "edge_density", "Higher = Cracks, fissures, nibbled borders"),
        ("Dark Necrotic Pixels (< 80)", "dark_spot_80_pct", "% of grain covered in dark rot/puncture"),
        ("Red / Brown / Amber Discoloration", "red_brown_pct", "% of grain with red/brown rot or bran"),
        ("Mean Saturation", "mean_saturation", "Higher = More vivid yellow/brown/pigment"),
        ("Overall Mean Brightness (Mean V)", "mean_v", "Overall grain brightness"),
    ]

    print(f"{'METRIC':<36} | {'DAMAGED (Median)':<18} | {'WHOLE (Median)':<18} | {'SEPARATION':<12} | {'WHAT IT MEANS'}")
    print("-" * 125)

    separations = {}

    for label, key, note in keys:
        d_mean, d_med, d_25, d_75 = stats(damaged_metrics, key)
        w_mean, w_med, w_25, w_75 = stats(whole_metrics, key)

        # Ratio or difference
        if w_med != 0:
            ratio = (d_med - w_med) / max(abs(w_med), 1e-3) * 100.0
            sep_str = f"{ratio:+6.1f}%"
        else:
            diff = d_med - w_med
            sep_str = f"{diff:+6.1f} diff"

        separations[key] = (d_med, w_med, ratio if w_med != 0 else diff)
        print(f"{label:<36} | {d_med:16.2f}   | {w_med:16.2f}   | {sep_str:<12} | {note}")

    print("\n" + "=" * 125)
    print("KEY FINDINGS SUMMARY:")

    # Check top discriminators
    print("\n1. Top Statistical Signatures of Damaged Grains:")
    for label, key, note in keys:
        d_med, w_med, diff = separations[key]
        if abs(diff) > 20:
            direction = "HIGHER" if d_med > w_med else "LOWER"
            print(f"   * {label}: Damaged grains are {abs(diff):.1f}% {direction} than whole grains.")

if __name__ == "__main__":
    run_forensic_comparison(sample_size=800)
