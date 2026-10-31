"""
Production Two-Stage Rice Quality Grading & Defect Inspection Pipeline
Combines:
  - Stage 1: YOLO11-seg (97% Recall Grain Localization & Polygon Contours)
  - Stage 2: MobileNetV3 (High-Resolution 128x128 Defect Classifier)
  - Geometric Measurements: Length, Width, Aspect Ratio, Area
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not installed.")
    sys.exit(1)

CLASS_NAMES = [
    "broken",
    "chalky",
    "damaged",
    "dehusked",
    "discoloured",
    "foreign_matter",
    "whole"
]

def make_square_crop(crop: np.ndarray, target_size: int = 128) -> np.ndarray:
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    max_dim = max(h, w)
    canvas = np.zeros((max_dim, max_dim, 3), dtype=crop.dtype)
    y_off = (max_dim - h) // 2
    x_off = (max_dim - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = crop
    return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)

class RiceQualityAnalyzer:
    def __init__(
        self,
        seg_weights: str = "runs/rice_seg_yolo11s_1280/weights/best.pt",
        cls_weights: str = "runs/stage2_resnet50/best_resnet50.pth",
        model_type: str = "resnet50",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        self.model_type = model_type.lower()
        self.img_size = 224 if self.model_type == "resnet50" else 128
        print(f"Initializing RiceQualityAnalyzer ({self.model_type.upper()}) on {self.device}...")

        # 1. Load Stage 1 YOLO Segmenter
        print(f"Loading Stage 1 Segmenter: {seg_weights}")
        self.seg_model = YOLO(seg_weights)

        # 2. Load Stage 2 Crop Classifier
        # Fallback to MobileNetV3 if resnet50 weights do not exist yet
        if not Path(cls_weights).exists() and Path("runs/stage2_classifier/best_classifier.pth").exists():
            print(f"Note: {cls_weights} not found, falling back to MobileNetV3 classifier")
            cls_weights = "runs/stage2_classifier/best_classifier.pth"
            self.model_type = "mobilenet"
            self.img_size = 128

        print(f"Loading Stage 2 Classifier: {cls_weights}")
        if self.model_type == "resnet50":
            self.cls_model = models.resnet50(weights=None)
            in_features = self.cls_model.fc.in_features
            self.cls_model.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, len(CLASS_NAMES))
            )
        else:
            self.cls_model = models.mobilenet_v3_small(weights=None)
            in_features = self.cls_model.classifier[3].in_features
            self.cls_model.classifier[3] = nn.Linear(in_features, len(CLASS_NAMES))

        checkpoint = torch.load(cls_weights, map_location=self.device)
        self.cls_model.load_state_dict(checkpoint["model_state_dict"])
        self.cls_model.to(self.device)
        self.cls_model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def analyze(self, image_path: str, conf_threshold: float = 0.30, output_dir: str = "runs/two_stage_results"):
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.time()
        orig_bgr = cv2.imread(str(image_path))
        if orig_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {image_path}")

        h_orig, w_orig = orig_bgr.shape[:2]

        # Stage 1: Segment all grains
        seg_res = self.seg_model.predict(
            source=str(image_path),
            conf=conf_threshold,
            imgsz=1280,
            max_det=1000,
            device=0 if "cuda" in str(self.device) else "cpu",
            save=False
        )[0]

        if seg_res.masks is None or len(seg_res.masks) == 0:
            return {"error": "No grains detected"}

        num_instances = len(seg_res.masks)

        # Extract crops and geometry
        grain_data = []
        crop_tensors = []
        lengths = []

        for i in range(num_instances):
            poly = seg_res.masks.xy[i]
            if len(poly) < 3:
                continue

            pts = np.array(poly, dtype=np.int32)
            rect = cv2.minAreaRect(pts.astype(np.float32))
            (cx, cy), (w, h), angle = rect
            length = max(w, h)
            width = min(w, h)
            area = cv2.contourArea(pts)

            if area < 40:
                continue

            x_min = max(0, int(np.min(pts[:, 0])) - 4)
            y_min = max(0, int(np.min(pts[:, 1])) - 4)
            x_max = min(w_orig, int(np.max(pts[:, 0])) + 4)
            y_max = min(h_orig, int(np.max(pts[:, 1])) + 4)

            raw_crop = orig_bgr[y_min:y_max, x_min:x_max]
            sq_crop = make_square_crop(raw_crop, target_size=self.img_size)
            sq_rgb = cv2.cvtColor(sq_crop, cv2.COLOR_BGR2RGB)

            tensor = self.transform(sq_rgb)
            crop_tensors.append(tensor)

            grain_data.append({
                "id": i + 1,
                "center": (cx, cy),
                "pts": pts,
                "length_px": round(float(length), 1),
                "width_px": round(float(width), 1),
                "aspect_ratio": round(float(length / max(width, 1e-4)), 2),
                "area_px": round(float(area), 1),
            })
            lengths.append(length)

        # Stage 2: Batch classification of all crops
        classified_results = []
        batch_size = 64
        with torch.no_grad():
            for b_idx in range(0, len(crop_tensors), batch_size):
                batch = torch.stack(crop_tensors[b_idx:b_idx + batch_size]).to(self.device)
                with torch.amp.autocast("cuda") if "cuda" in str(self.device) else torch.no_grad():
                    logits = self.cls_model(batch)
                probs = torch.softmax(logits.float(), dim=1)

                for prob_vec in probs.cpu().numpy():
                    classified_results.append(prob_vec)

        # Sizing standard: ISO 75% cutoff for broken
        sorted_lens = sorted(lengths)
        ref_bracket = sorted_lens[int(len(sorted_lens) * 0.4):]
        ref_whole_len = float(np.median(ref_bracket)) if ref_bracket else float(np.median(sorted_lens))
        broken_cutoff = 0.75 * ref_whole_len

        class_counts = {c: 0 for c in CLASS_NAMES}

        # Render clean annotated image
        annotated_img = orig_bgr.copy()
        CLASS_COLORS = {
            "whole": (0, 220, 0),         # Green
            "broken": (0, 165, 255),      # Orange
            "chalky": (255, 255, 0),      # Cyan
            "damaged": (0, 0, 255),       # Red
            "discoloured": (180, 0, 180), # Purple
            "dehusked": (0, 255, 255),    # Yellow
            "foreign_matter": (255, 0, 0) # Blue
        }

        # Class index mapping in CLASS_NAMES:
        # ['broken', 'chalky', 'damaged', 'dehusked', 'discoloured', 'foreign_matter', 'whole']
        IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

        # Pass 1: Extract core optical metrics for all grains
        optical_metrics = []
        for item in grain_data:
            pts = item["pts"]
            x_min = max(0, int(np.min(pts[:, 0])))
            y_min = max(0, int(np.min(pts[:, 1])))
            x_max = min(w_orig, int(np.max(pts[:, 0])))
            y_max = min(h_orig, int(np.max(pts[:, 1])))

            grain_patch = orig_bgr[y_min:y_max, x_min:x_max]
            patch_mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
            shifted_pts = pts - np.array([x_min, y_min])
            cv2.fillPoly(patch_mask, [shifted_pts], 255)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            inner_mask = cv2.erode(patch_mask, kernel) > 0
            if np.sum(inner_mask) < 30:
                inner_mask = patch_mask > 0

            core_bgr = grain_patch[inner_mask]
            core_hsv = cv2.cvtColor(grain_patch, cv2.COLOR_BGR2HSV)[inner_mask]

            r_vals = core_bgr[:, 2].astype(np.float32)
            g_vals = core_bgr[:, 1].astype(np.float32)
            rg_ratio = float(np.mean(r_vals / np.maximum(g_vals, 1.0)))

            v_vals = core_hsv[:, 2].astype(np.float32)
            s_vals = core_hsv[:, 1].astype(np.float32)
            core_std_v = float(np.std(v_vals))
            median_v = float(np.median(v_vals))

            pinhole_ratio = float(np.sum(v_vals < 0.60 * median_v)) / max(float(len(v_vals)), 1.0)
            chalky_ratio = float(np.sum((v_vals >= 170) & (s_vals <= 35))) / max(float(len(v_vals)), 1.0)

            optical_metrics.append({
                "rg_ratio": rg_ratio,
                "core_std_v": core_std_v,
                "pinhole_ratio": pinhole_ratio,
                "chalky_ratio": chalky_ratio
            })

        # Calculate tray baselines (immune to scanner lighting/exposure differences)
        tray_median_rg = float(np.median([m["rg_ratio"] for m in optical_metrics])) if optical_metrics else 1.0
        tray_median_std = float(np.median([m["core_std_v"] for m in optical_metrics])) if optical_metrics else 15.0

        # Pass 2: Calibrated Decision Fusion Engine
        for item, prob_vec, opt in zip(grain_data, classified_results, optical_metrics):
            rg_ratio = opt["rg_ratio"]
            core_std_v = opt["core_std_v"]
            pinhole_ratio = opt["pinhole_ratio"]
            chalky_ratio = opt["chalky_ratio"]

            # Relative defect signals (outliers against the tray baseline)
            has_color_oxidation = (rg_ratio > 1.03 * tray_median_rg)
            has_mottled_texture = (core_std_v > 1.30 * tray_median_std)
            has_pinhole_defect = (pinhole_ratio > 0.05)

            p_broken = prob_vec[IDX["broken"]]
            p_chalky = prob_vec[IDX["chalky"]]
            p_damaged = prob_vec[IDX["damaged"]]
            p_dehusked = prob_vec[IDX["dehusked"]]
            p_discoloured = prob_vec[IDX["discoloured"]]
            p_foreign = prob_vec[IDX["foreign_matter"]]
            p_whole = prob_vec[IDX["whole"]]

            if p_foreign > 0.40 or rg_ratio > 1.15:
                final_cls = "foreign_matter"
                conf = float(p_foreign)
            elif item["length_px"] < broken_cutoff:
                final_cls = "broken"
                conf = float(max(p_broken, 0.85))
            elif p_dehusked > 0.40:
                final_cls = "dehusked"
                conf = float(p_dehusked)
            elif (p_damaged > p_whole) or (p_damaged > 0.35 and (has_color_oxidation or has_pinhole_defect)):
                final_cls = "damaged"
                conf = float(p_damaged)
            elif (p_chalky > p_whole) or (p_chalky > 0.35 and chalky_ratio > 0.20):
                final_cls = "chalky"
                conf = float(p_chalky)
            elif (p_discoloured > p_whole) or (p_discoloured > 0.35 and has_color_oxidation):
                final_cls = "discoloured"
                conf = float(p_discoloured)
            else:
                final_cls = "whole"
                conf = float(p_whole)

            class_counts[final_cls] = class_counts.get(final_cls, 0) + 1
            item["class"] = final_cls
            item["confidence"] = round(conf, 3)
            item["rg_ratio"] = round(rg_ratio, 3)
            item["texture_std"] = round(core_std_v, 1)

            color = CLASS_COLORS.get(final_cls, (255, 255, 255))
            cv2.polylines(annotated_img, [item["pts"]], isClosed=True, color=color, thickness=2)

        total_detected = len(grain_data)
        total_rice = total_detected - class_counts.get("foreign_matter", 0)
        elapsed_sec = time.time() - t_start

        # Commercial Milling Inspection Report
        summary = {
            "image_name": image_path.name,
            "analysis_time_sec": round(elapsed_sec, 2),
            "total_detected_instances": total_detected,
            "total_rice_grains": total_rice,
            "class_counts": class_counts,
            "commercial_percentages": {
                "head_rice_recovery_pct": round((class_counts.get("whole", 0) / max(total_rice, 1)) * 100, 2),
                "broken_grain_pct": round((class_counts.get("broken", 0) / max(total_rice, 1)) * 100, 2),
                "chalkiness_index_pct": round((class_counts.get("chalky", 0) / max(total_rice, 1)) * 100, 2),
                "damaged_grain_pct": round((class_counts.get("damaged", 0) / max(total_rice, 1)) * 100, 2),
                "discoloured_grain_pct": round((class_counts.get("discoloured", 0) / max(total_rice, 1)) * 100, 2),
                "dehusked_pct": round((class_counts.get("dehusked", 0) / max(total_rice, 1)) * 100, 2),
                "foreign_matter_count": class_counts.get("foreign_matter", 0),
            },
            "milling_quality_grade": "Grade A" if (class_counts.get("broken", 0) / max(total_rice, 1)) < 0.05 else "Grade B",
            "physical_dimensions": {
                "reference_whole_length_px": round(ref_whole_len, 1),
                "broken_cutoff_px": round(broken_cutoff, 1),
                "mean_grain_length_px": round(float(np.mean([g["length_px"] for g in grain_data])), 1),
                "mean_grain_width_px": round(float(np.mean([g["width_px"] for g in grain_data])), 1),
                "mean_aspect_ratio": round(float(np.mean([g["aspect_ratio"] for g in grain_data])), 2),
            }
        }

        # Save annotated image and report
        annotated_path = output_dir / f"graded_{image_path.name}"
        cv2.imwrite(str(annotated_path), annotated_img)

        # Remove polygon coordinates from JSON to keep file compact
        clean_grains = [{k: v for k, v in g.items() if k != "pts"} for g in grain_data]
        summary["grains_detail"] = clean_grains

        report_path = output_dir / f"report_{image_path.stem}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nAnalysis completed in {elapsed_sec:.2f}s!")
        print(f"Annotated scan saved to: {annotated_path}")
        print(f"Inspection report saved to: {report_path}")
        print(json.dumps(summary["commercial_percentages"], indent=2))

        return summary

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/two_stage_grade.py <image_path>")
        sys.exit(1)

    analyzer = RiceQualityAnalyzer()
    analyzer.analyze(sys.argv[1])
