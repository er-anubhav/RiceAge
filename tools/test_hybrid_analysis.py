"""
Test script for High-Resolution Hybrid Rice Analysis.
Evaluates physical grain metrics and optical defect parameters from uncompressed image pixels.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not installed.")
    sys.exit(1)

def run_test(image_path: str, model_path: str):
    image_path = Path(image_path)
    model_path = Path(model_path)

    print(f"Reading full-resolution image: {image_path}")
    orig_bgr = cv2.imread(str(image_path))
    if orig_bgr is None:
        print("Failed to read image.")
        return

    h_orig, w_orig = orig_bgr.shape[:2]
    print(f"Original Resolution: {w_orig} x {h_orig}")

    print(f"Loading YOLO weights: {model_path}")
    model = YOLO(str(model_path))

    # Run inference with YOLO to get precise grain polygons
    results = model.predict(
        source=str(image_path),
        imgsz=1280,
        conf=0.25,
        max_det=1000,
        device=0,
        save=False
    )
    result = results[0]

    if result.masks is None:
        print("No masks detected.")
        return

    detected_count = len(result.masks)
    print(f"Total Grain Contours Found: {detected_count}")

    # Convert image to HSV and Grayscale for optical inspection
    orig_hsv = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)
    orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)

    grains = []
    lengths = []

    for i in range(detected_count):
        poly = result.masks.xy[i]
        if len(poly) < 3:
            continue

        pts = np.array(poly, dtype=np.int32)
        rect = cv2.minAreaRect(pts.astype(np.float32))
        (cx, cy), (w, h), angle = rect
        length = max(w, h)
        width = min(w, h)
        area = cv2.contourArea(pts)

        if area < 100:  # Ignore tiny artifact specks
            continue

        yolo_cls_id = int(result.boxes.cls[i].item())
        yolo_conf = float(result.boxes.conf[i].item())

        grains.append({
            "idx": i,
            "pts": pts,
            "center": (cx, cy),
            "length": length,
            "width": width,
            "aspect_ratio": length / max(width, 1e-4),
            "area": area,
            "yolo_cls": yolo_cls_id,
            "yolo_conf": yolo_conf,
        })
        # Record candidate whole grain lengths (exclude obvious tiny fragments)
        lengths.append(length)

    if not grains:
        print("No valid grains extracted.")
        return

    # Calculate median length of the upper 60% of grains (reference whole grain length)
    sorted_lengths = sorted(lengths)
    upper_bracket = sorted_lengths[int(len(sorted_lengths) * 0.4):]
    ref_length = float(np.median(upper_bracket)) if upper_bracket else float(np.median(sorted_lengths))
    broken_cutoff = 0.75 * ref_length

    print(f"Reference Whole Grain Length: {ref_length:.1f} px")
    print(f"ISO Broken Cutoff (75%): {broken_cutoff:.1f} px\n")

    # Analyze each grain optically
    hybrid_counts = {
        "whole": 0,
        "broken": 0,
        "chalky": 0,
        "damaged": 0,
        "discoloured": 0,
        "foreign_matter": 0,
    }

    annotated = orig_bgr.copy()

    for g in grains:
        pts = g["pts"]
        x, y, w, h = cv2.boundingRect(pts)
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_orig, x + w + pad)
        y2 = min(h_orig, y + h + pad)

        # Grain crop mask
        crop_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        shifted_pts = pts - np.array([x1, y1])
        cv2.fillPoly(crop_mask, [shifted_pts], 255)

        crop_bgr = orig_bgr[y1:y2, x1:x2]
        crop_gray = orig_gray[y1:y2, x1:x2]
        crop_hsv = orig_hsv[y1:y2, x1:x2]

        grain_pixels = crop_mask > 0
        total_grain_px = int(np.sum(grain_pixels))
        if total_grain_px == 0:
            continue

        # 1. Chalkiness metric:
        # Translucent rice absorbs light; chalky rice has high diffuse reflectance
        # High value (V >= 180), Low saturation (S <= 45)
        v_channel = crop_hsv[:, :, 2]
        s_channel = crop_hsv[:, :, 1]
        chalky_mask = (crop_mask > 0) & (v_channel >= 180) & (s_channel <= 50)
        chalky_ratio = float(np.sum(chalky_mask)) / total_grain_px

        # 2. Damaged / Dark necrotic spot metric:
        # Pinhole insect marks, fungus, cracks (V <= 65)
        damaged_mask = (crop_mask > 0) & (v_channel <= 70)
        damaged_ratio = float(np.sum(damaged_mask)) / total_grain_px

        # 3. Discoloured metric:
        # Yellowish / brownish / amber shift
        # Hue between 15 and 35, Saturation >= 75
        h_channel = crop_hsv[:, :, 0]
        discoloured_mask = (crop_mask > 0) & (h_channel >= 15) & (h_channel <= 35) & (s_channel >= 75)
        discoloured_ratio = float(np.sum(discoloured_mask)) / total_grain_px

        # Class decision logic
        if g["yolo_cls"] == 6:  # foreign matter detected by YOLO
            final_cls = "foreign_matter"
        elif g["length"] < broken_cutoff:
            final_cls = "broken"
        elif damaged_ratio >= 0.04:
            final_cls = "damaged"
        elif chalky_ratio >= 0.20:
            final_cls = "chalky"
        elif discoloured_ratio >= 0.20:
            final_cls = "discoloured"
        else:
            final_cls = "whole"

        hybrid_counts[final_cls] += 1

        # Draw contour with color
        colors = {
            "whole": (0, 220, 0),
            "broken": (0, 165, 255),
            "chalky": (255, 255, 0),
            "damaged": (0, 0, 255),
            "discoloured": (180, 0, 180),
            "foreign_matter": (255, 0, 0),
        }
        color = colors.get(final_cls, (255, 255, 255))
        cv2.polylines(annotated, [pts], True, color, 3)

        # Label near grain
        cx, cy = int(g["center"][0]), int(g["center"][1])
        cv2.putText(annotated, final_cls[:3].upper(), (cx - 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    out_dir = Path("runs/hybrid_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_dir / f"hybrid_{image_path.name}"
    cv2.imwrite(str(out_img_path), annotated)

    print("=== HYBRID GRADING RESULTS ===")
    for k, v in hybrid_counts.items():
        print(f"  {k:15s}: {v}")
    print(f"\nAnnotated image saved to: {out_img_path}")

if __name__ == "__main__":
    img = "dataset/segment/images/test/actual_mixed__mixed_6.jpg"
    weights = "runs/rice_seg_yolo11s_1280/weights/best.pt"
    run_test(img, weights)
