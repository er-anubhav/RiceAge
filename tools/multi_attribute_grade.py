"""
Multi-Attribute Rice Quality Grading Engine (Pillar 1 Architecture)
Decouples Physical Morphology from Surface Defect Inspection:
  1. Morphology: Whole Grain (Head Rice) vs. Broken Grain (ISO 75% standard)
  2. Surface Defects: Chalky, Damaged, Discoloured, Dehusked, Foreign Matter (Independent Flags)
  3. Commercial Standards: Codex Alimentarius & ISO 7301 compliance reporting
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

class MultiAttributeRiceAnalyzer:
    def __init__(
        self,
        seg_weights: str = "runs/rice_seg_yolo11s_1280/weights/best.pt",
        cls_weights: str = "runs/stage2_resnet50/best_resnet50.pth",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        print(f"Initializing Multi-Attribute Analyzer on {self.device}...")

        # Stage 1: YOLO Segmenter
        print(f"Loading Stage 1 Segmenter: {seg_weights}")
        self.seg_model = YOLO(seg_weights)

        # Stage 2: ResNet-50 Feature Extractor
        print(f"Loading Stage 2 ResNet-50: {cls_weights}")
        self.cls_model = models.resnet50(weights=None)
        in_features = self.cls_model.fc.in_features
        self.cls_model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, len(CLASS_NAMES))
        )
        checkpoint = torch.load(cls_weights, map_location=self.device)
        self.cls_model.load_state_dict(checkpoint["model_state_dict"])
        self.cls_model.to(self.device)
        self.cls_model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def analyze(self, image_path: str, output_dir: str = "runs/multi_attribute_results"):
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t_start = time.time()
        orig_bgr = cv2.imread(str(image_path))
        if orig_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {image_path}")

        h_orig, w_orig = orig_bgr.shape[:2]

        # Stage 1: Segment all grains on tray
        seg_res = self.seg_model.predict(
            source=str(image_path),
            conf=0.25,
            imgsz=1280,
            max_det=1000,
            device=0 if "cuda" in str(self.device) else "cpu",
            save=False
        )[0]

        if seg_res.masks is None or len(seg_res.masks) == 0:
            return {"error": "No grains detected"}

        num_instances = len(seg_res.masks)

        grain_candidates = []
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

            x_min = max(0, int(np.min(pts[:, 0])) - 6)
            y_min = max(0, int(np.min(pts[:, 1])) - 6)
            x_max = min(w_orig, int(np.max(pts[:, 0])) + 6)
            y_max = min(h_orig, int(np.max(pts[:, 1])) + 6)

            raw_crop = orig_bgr[y_min:y_max, x_min:x_max]
            sq_crop = make_square_crop_224(raw_crop, target_size=224)
            sq_rgb = cv2.cvtColor(sq_crop, cv2.COLOR_BGR2RGB)

            crop_tensors.append(self.transform(sq_rgb))

            yolo_cls = int(seg_res.boxes.cls[i].item())
            yolo_conf = float(seg_res.boxes.conf[i].item())

            grain_candidates.append({
                "id": i + 1,
                "center": (cx, cy),
                "pts": pts,
                "length_px": round(float(length), 1),
                "width_px": round(float(width), 1),
                "aspect_ratio": round(float(length / max(width, 1e-4)), 2),
                "area_px": round(float(area), 1),
                "yolo_cls": yolo_cls,
                "yolo_conf": yolo_conf,
            })
            lengths.append(length)

        # Batch Inference for Defect Probabilities
        classified_probs = []
        batch_size = 64
        with torch.no_grad():
            for b_idx in range(0, len(crop_tensors), batch_size):
                batch = torch.stack(crop_tensors[b_idx:b_idx + batch_size]).to(self.device)
                with torch.amp.autocast("cuda") if "cuda" in str(self.device) else torch.no_grad():
                    logits = self.cls_model(batch)
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                classified_probs.extend(probs)

        # 1. MORPHOLOGY CALCULATION (ISO 7301 Standard)
        sorted_lens = sorted(lengths)
        ref_bracket = sorted_lens[int(len(sorted_lens) * 0.4):]
        ref_whole_len = float(np.median(ref_bracket)) if ref_bracket else float(np.median(sorted_lens))
        broken_cutoff = 0.75 * ref_whole_len

        # Extract core optical metrics
        optical_metrics = []
        for item in grain_candidates:
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

        tray_median_rg = float(np.median([m["rg_ratio"] for m in optical_metrics])) if optical_metrics else 1.0
        tray_median_std = float(np.median([m["core_std_v"] for m in optical_metrics])) if optical_metrics else 15.0

        IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

        final_grains = []
        foreign_matter_items = []

        total_whole = 0
        total_broken = 0

        defect_counts = {
            "sound_clean": 0,
            "chalky": 0,
            "damaged": 0,
            "discoloured": 0,
            "dehusked": 0,
        }

        annotated_img = orig_bgr.copy()

        for item, prob_vec, opt in zip(grain_candidates, classified_probs, optical_metrics):
            rg_ratio = opt["rg_ratio"]
            core_std_v = opt["core_std_v"]
            pinhole_ratio = opt["pinhole_ratio"]
            chalky_ratio = opt["chalky_ratio"]

            p_foreign = prob_vec[IDX["foreign_matter"]]
            p_damaged = prob_vec[IDX["damaged"]]
            p_chalky = prob_vec[IDX["chalky"]]
            p_discoloured = prob_vec[IDX["discoloured"]]
            p_dehusked = prob_vec[IDX["dehusked"]]
            p_whole = prob_vec[IDX["whole"]]

            # 1. Foreign Matter Check
            if item["yolo_cls"] == 6 or p_foreign > 0.40 or rg_ratio > 1.15:
                foreign_matter_items.append(item)
                cv2.polylines(annotated_img, [item["pts"]], True, (255, 0, 0), 2)
                continue

            # 2. Physical Morphology Head (Whole vs Broken)
            is_broken = item["length_px"] < broken_cutoff
            morphology = "broken" if is_broken else "whole"
            if is_broken:
                total_broken += 1
            else:
                total_whole += 1

            # 3. Multi-Defect Flags Head (Grains can have multiple surface defects simultaneously)
            defects = []

            has_oxidation = (rg_ratio > 1.025 * tray_median_rg)
            has_mottled = (core_std_v > 1.25 * tray_median_std)
            has_pinhole = (pinhole_ratio > 0.04)

            # Check Damaged Flag
            if (p_damaged > p_whole) or (p_damaged > 0.30 and (has_oxidation or has_mottled or has_pinhole)):
                defects.append("damaged")
                defect_counts["damaged"] += 1

            # Check Chalky Flag
            if (p_chalky > p_whole) or (p_chalky > 0.32 and chalky_ratio > 0.20):
                defects.append("chalky")
                defect_counts["chalky"] += 1

            # Check Discoloured Flag
            if (p_discoloured > p_whole) or (p_discoloured > 0.30 and has_oxidation and not has_mottled):
                defects.append("discoloured")
                defect_counts["discoloured"] += 1

            # Check Dehusked / Paddy Flag
            if p_dehusked > 0.35:
                defects.append("dehusked")
                defect_counts["dehusked"] += 1

            is_sound = (len(defects) == 0)
            if is_sound:
                defect_counts["sound_clean"] += 1

            # Color coding for visualization:
            # Clean Whole = Green, Clean Broken = Orange
            # Damaged = Red, Chalky = Cyan, Discoloured = Purple, Dehusked = Yellow
            if "damaged" in defects:
                color = (0, 0, 255)       # Red
            elif "chalky" in defects:
                color = (255, 255, 0)     # Cyan
            elif "discoloured" in defects:
                color = (180, 0, 180)     # Purple
            elif "dehusked" in defects:
                color = (0, 255, 255)     # Yellow
            elif is_broken:
                color = (0, 165, 255)     # Orange
            else:
                color = (0, 220, 0)       # Green (Clean Whole)

            cv2.polylines(annotated_img, [item["pts"]], True, color, 2)

            item_info = {
                "id": item["id"],
                "center": item["center"],
                "length_px": item["length_px"],
                "width_px": item["width_px"],
                "aspect_ratio": item["aspect_ratio"],
                "area_px": item["area_px"],
                "morphology": morphology,
                "is_broken": is_broken,
                "defects": defects,
                "is_sound": is_sound,
            }
            final_grains.append(item_info)

        total_grains = total_whole + total_broken
        total_detected = total_grains + len(foreign_matter_items)
        elapsed_sec = time.time() - t_start

        # Commercial Grade Certificate (Codex & ISO 7301 standards)
        hrr_pct = round((total_whole / max(total_grains, 1)) * 100.0, 2)
        broken_pct = round((total_broken / max(total_grains, 1)) * 100.0, 2)

        damaged_pct = round((defect_counts["damaged"] / max(total_grains, 1)) * 100.0, 2)
        chalky_pct = round((defect_counts["chalky"] / max(total_grains, 1)) * 100.0, 2)
        discoloured_pct = round((defect_counts["discoloured"] / max(total_grains, 1)) * 100.0, 2)
        dehusked_pct = round((defect_counts["dehusked"] / max(total_grains, 1)) * 100.0, 2)
        sound_pct = round((defect_counts["sound_clean"] / max(total_grains, 1)) * 100.0, 2)

        # Standard Commercial Grade assignment
        if broken_pct <= 5.0 and damaged_pct <= 2.0 and chalky_pct <= 5.0:
            quality_grade = "Premium Grade (Grade 1 / Export Top Choice)"
        elif broken_pct <= 15.0 and damaged_pct <= 5.0 and chalky_pct <= 10.0:
            quality_grade = "Standard Commercial Grade (Grade 2)"
        else:
            quality_grade = "Milling Utility Grade (Grade 3)"

        certificate = {
            "sample_name": image_path.name,
            "analysis_time_sec": round(elapsed_sec, 2),
            "summary": {
                "total_instances_detected": total_detected,
                "total_rice_grains": total_grains,
                "foreign_matter_count": len(foreign_matter_items),
                "commercial_quality_grade": quality_grade
            },
            "morphological_analysis (ISO 7301)": {
                "head_rice_recovery_pct (HRR)": hrr_pct,
                "broken_rice_pct": broken_pct,
                "whole_grains_count": total_whole,
                "broken_grains_count": total_broken,
                "reference_whole_length_px": round(ref_whole_len, 1),
                "broken_cutoff_px": round(broken_cutoff, 1),
                "mean_grain_length_px": round(float(np.mean([g["length_px"] for g in final_grains])), 1) if final_grains else 0,
                "mean_grain_width_px": round(float(np.mean([g["width_px"] for g in final_grains])), 1) if final_grains else 0,
                "mean_aspect_ratio": round(float(np.mean([g["aspect_ratio"] for g in final_grains])), 2) if final_grains else 0,
            },
            "defect_audit_analysis": {
                "sound_clean_grains_pct": sound_pct,
                "sound_clean_count": defect_counts["sound_clean"],
                "damaged_grain_pct": damaged_pct,
                "damaged_count": defect_counts["damaged"],
                "chalkiness_index_pct": chalky_pct,
                "chalky_count": defect_counts["chalky"],
                "discoloured_grain_pct": discoloured_pct,
                "discoloured_count": defect_counts["discoloured"],
                "dehusked_paddy_pct": dehusked_pct,
                "dehusked_count": defect_counts["dehusked"],
            },
            "export_trade_compliance": {
                "broken_within_limit": bool(broken_pct <= 15.0),
                "damage_within_limit": bool(damaged_pct <= 4.0),
                "chalkiness_within_limit": bool(chalky_pct <= 8.0),
                "foreign_matter_zero_tolerance": bool(len(foreign_matter_items) == 0)
            }
        }

        # Save annotated tray and certificate
        annotated_path = output_dir / f"multi_attr_{image_path.name}"
        cv2.imwrite(str(annotated_path), annotated_img)

        report_path = output_dir / f"certificate_{image_path.stem}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(certificate, f, indent=2)

        print(f"\nMulti-Attribute Quality Inspection Complete in {elapsed_sec:.2f}s!")
        print(f"Annotated Tray: {annotated_path}")
        print(f"Inspection Certificate: {report_path}\n")
        print("=== COMMERCIAL MILLING CERTIFICATE ===")
        print(f"Quality Grade: {quality_grade}")
        print(f"Head Rice Recovery (HRR): {hrr_pct}% | Broken: {broken_pct}%")
        print(f"Sound Clean Grains: {sound_pct}%")
        print(f"Damaged: {damaged_pct}% | Chalky: {chalky_pct}% | Discoloured: {discoloured_pct}% | Dehusked: {dehusked_pct}%")
        print(f"Foreign Matter: {len(foreign_matter_items)} items")

        return certificate

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/multi_attribute_grade.py <image_path>")
        sys.exit(1)

    analyzer = MultiAttributeRiceAnalyzer()
    analyzer.analyze(sys.argv[1])
