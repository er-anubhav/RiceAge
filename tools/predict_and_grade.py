"""
Rice Crop Analyzer - Prediction & Quality Grading Pipeline
Takes an input scan, runs instance segmentation, extracts geometric grain metrics
(length, width, L/W aspect ratio, area), and computes commercial quality metrics:
- Head Rice Recovery (HRR) %
- Broken Grain %
- Chalkiness Index %
- Damaged & Discoloured %
- Foreign Matter count
"""

import sys
import json
import math
from pathlib import Path
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics is not installed.")
    sys.exit(1)

CLASS_NAMES = [
    "whole",
    "broken",
    "chalky",
    "damaged",
    "discoloured",
    "dehusked",
    "foreign_matter"
]

def analyze_rice_image(image_path: str, model_path: str, conf_threshold: float = 0.35, output_dir: str = "runs/grade_results"):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Analyzing scan: {image_path.name}")
    # Inference with high resolution and generous max_det for crowded trays
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        imgsz=1280,
        max_det=1000,
        device=0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 or True else "cpu",
        save=False,
    )

    result = results[0]
    orig_img = result.orig_img.copy()

    # Per-class counts
    class_counts = {name: 0 for name in CLASS_NAMES}
    grain_measurements = []

    has_masks = result.masks is not None

    boxes = result.boxes
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy()

        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        length_px = 0.0
        width_px = 0.0
        aspect_ratio = 0.0
        area_px = 0.0

        if has_masks and i < len(result.masks):
            polygon = result.masks.xy[i]
            if len(polygon) >= 5:
                pts = np.array(polygon, dtype=np.float32)
                rect = cv2.minAreaRect(pts)
                (cx, cy), (w, h), angle = rect
                length_px = max(w, h)
                width_px = min(w, h)
                aspect_ratio = round(length_px / max(width_px, 1e-4), 2)
                area_px = cv2.contourArea(pts.astype(np.int32))
        else:
            # Fallback to bounding box dimensions
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            length_px = max(w, h)
            width_px = min(w, h)
            aspect_ratio = round(length_px / max(width_px, 1e-4), 2)
            area_px = float(w * h)

        grain_item = {
            "id": i + 1,
            "class": cls_name,
            "confidence": round(conf, 3),
            "length_px": round(float(length_px), 1),
            "width_px": round(float(width_px), 1),
            "aspect_ratio": aspect_ratio,
            "area_px": round(float(area_px), 1),
            "bbox": [float(x) for x in xyxy],
        }
        if has_masks and i < len(result.masks):
            polygon = result.masks.xy[i]
            if len(polygon) >= 3:
                grain_item["polygon"] = [[float(p[0]), float(p[1])] for p in polygon]

        grain_measurements.append(grain_item)

    # Summary statistics
    total_detected = len(grain_measurements)
    total_rice_grains = total_detected - class_counts.get("foreign_matter", 0)

    quality_summary = {
        "image_name": image_path.name,
        "total_detected_instances": total_detected,
        "total_rice_grains": total_rice_grains,
        "class_breakdown": class_counts,
        "commercial_percentages": {
            "head_rice_ratio_pct": round((class_counts.get("whole", 0) / max(total_rice_grains, 1)) * 100, 2),
            "broken_pct": round((class_counts.get("broken", 0) / max(total_rice_grains, 1)) * 100, 2),
            "chalky_pct": round((class_counts.get("chalky", 0) / max(total_rice_grains, 1)) * 100, 2),
            "damaged_pct": round((class_counts.get("damaged", 0) / max(total_rice_grains, 1)) * 100, 2),
            "discoloured_pct": round((class_counts.get("discoloured", 0) / max(total_rice_grains, 1)) * 100, 2),
            "foreign_matter_count": class_counts.get("foreign_matter", 0),
        },
        "grain_metrics": {
            "mean_length_px": round(float(np.mean([g["length_px"] for g in grain_measurements])) if grain_measurements else 0.0, 1),
            "mean_width_px": round(float(np.mean([g["width_px"] for g in grain_measurements])) if grain_measurements else 0.0, 1),
            "mean_aspect_ratio": round(float(np.mean([g["aspect_ratio"] for g in grain_measurements])) if grain_measurements else 0.0, 2),
        }
    }

    # Save visual prediction cleanly with OpenCV (avoids PyTorch 57GB upsampling crash on 5000x7000 px scans)
    CLASS_COLORS = {
        "whole": (0, 200, 0),         # Green
        "broken": (0, 165, 255),      # Orange
        "chalky": (255, 255, 0),      # Cyan
        "damaged": (0, 0, 255),       # Red
        "discoloured": (180, 0, 180), # Purple
        "dehusked": (0, 255, 255),    # Yellow
        "foreign_matter": (255, 0, 0) # Blue
    }

    annotated_img = orig_img.copy()
    for item in grain_measurements:
        cls_name = item["class"]
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
        if "polygon" in item and len(item["polygon"]) >= 3:
            pts = np.array(item["polygon"], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_img, [pts], isClosed=True, color=color, thickness=2)
        elif "bbox" in item:
            x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

    annotated_path = output_dir / f"annotated_{image_path.name}"
    cv2.imwrite(str(annotated_path), annotated_img)

    # Save JSON report
    report_path = output_dir / f"grade_report_{image_path.stem}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_summary, f, indent=2)

    print(f"\nAnalysis Report saved to: {report_path}")
    print(f"Annotated scan saved to: {annotated_path}")
    print(json.dumps(quality_summary["commercial_percentages"], indent=2))

    return quality_summary

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_and_grade.py <path_to_image> [path_to_model]")
        sys.exit(1)

    input_img = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) > 2 else "runs/rice_seg_yolo11s_1280/weights/best.pt"
    analyze_rice_image(input_img, model_file)
