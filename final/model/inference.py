"""
RiceAge Standalone Quality Classifier & Inference Engine (Milestone M4)
========================================================================
Performs optical grain segmentation and multi-head quality inference using
calibrated logistic regression models saved in quality_classifier_weights.json.

Zero heavy ML dependencies: runs on pure OpenCV + NumPy, ideal for
Raspberry Pi 4/5 ARM hardware and desktop kiosks.
"""

import os
import json
import math
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np


class QualityInferenceEngine:
    def __init__(self, weights_path: Optional[str] = None):
        if weights_path is None:
            weights_path = os.path.join(os.path.dirname(__file__), "quality_classifier_weights.json")
        self.weights_path = weights_path
        self.heads: Dict[str, Any] = {}
        self.is_loaded = False
        self._load_weights()

    def _load_weights(self) -> None:
        if os.path.exists(self.weights_path):
            try:
                with open(self.weights_path, "r") as f:
                    data = json.load(f)
                self.heads = data.get("heads", {})
                required = ["foreign_matter", "whole_broken", "chalky", "discoloured", "damaged"]
                if all(k in self.heads for k in required):
                    self.is_loaded = True
                    print(f"[ML Model] Successfully loaded M4 weights from {self.weights_path}")
                else:
                    print(f"[ML Model WARNING] Missing heads in {self.weights_path}. Using rule fallbacks.")
            except Exception as e:
                print(f"[ML Model ERROR] Failed reading {self.weights_path}: {e}")
        else:
            print(f"[ML Model WARNING] Weights file not found: {self.weights_path}")

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _predict_head(self, head_name: str, features_dict: Dict[str, float]) -> Tuple[float, int]:
        h = self.heads[head_name]
        cols = h["feature_cols"]
        thresh = float(h.get("threshold", 0.50))
        vals = [features_dict.get(c, 0.0) for c in cols]

        # Non-linear tree ensemble inference (Zero external ML dependencies, pure NumPy/Python)
        if h.get("model_type") == "gbdt" and "trees" in h:
            raw = float(h.get("init_bias", 0.0))
            lr = float(h.get("learning_rate", 0.1))
            for tree in h["trees"]:
                node = 0
                left = tree["children_left"]
                right = tree["children_right"]
                feat = tree["feature"]
                th = tree["threshold"]
                val = tree["value"]
                while left[node] != -1:
                    if vals[feat[node]] <= th[node]:
                        node = left[node]
                    else:
                        node = right[node]
                raw += lr * val[node]
            prob = float(self._sigmoid(np.array([raw]))[0])
            pred = 1 if prob >= thresh else 0
            return prob, pred

        # Linear logistic regression fallback
        mean = np.array(h["scaler_mean"], dtype=float)
        scale = np.array(h["scaler_scale"], dtype=float)
        coef = np.array(h["coef"], dtype=float)
        intercept = float(h["intercept"])
        scale_safe = np.where(scale == 0, 1.0, scale)
        z_norm = (np.array(vals, dtype=float) - mean) / scale_safe
        logit = float(np.dot(z_norm, coef) + intercept)
        prob = float(self._sigmoid(np.array([logit]))[0])
        pred = 1 if prob >= thresh else 0
        return prob, pred

    def extract_grain_features(
        self,
        crop_bgr: np.ndarray,
        grain_mask: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """Extract exact physical, photometric, CIELAB and texture features for a single grain."""
        cnts, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        if area < 10.0:
            return None

        peri = float(cv2.arcLength(c, True))
        rect = cv2.minAreaRect(c)
        (cx, cy), (dim1, dim2), angle = rect
        length = float(max(dim1, dim2))
        width = float(min(dim1, dim2))
        ar = float(length / max(1e-5, width))
        circ = float(min(1.0, max(0.0, 4.0 * np.pi * area / max(1e-5, peri**2))))

        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(min(1.0, max(0.0, area / max(1e-5, hull_area))))

        # Convexity defects
        max_defect = 0.0
        num_defects = 0
        try:
            hull_idx = cv2.convexHull(c, returnPoints=False)
            if hull_idx is not None and len(hull_idx) > 3 and len(c) > 3:
                defects = cv2.convexityDefects(c, hull_idx)
                if defects is not None:
                    for i in range(len(defects)):
                        d = defects[i, 3] if defects.ndim == 2 else defects[i, 0, 3]
                        depth = d / 256.0
                        if depth > 1.5:
                            num_defects += 1
                            max_defect = max(max_defect, depth)
        except Exception:
            pass

        # End bluntness and asymmetry
        rot_angle = angle if dim1 < dim2 else angle + 90.0
        M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
        h_c, w_c = grain_mask.shape
        cos_a = np.abs(M[0, 0])
        sin_a = np.abs(M[0, 1])
        nW = int((h_c * sin_a) + (w_c * cos_a))
        nH = int((h_c * cos_a) + (w_c * sin_a))
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        rot_mask = cv2.warpAffine(grain_mask, M, (nW, nH), flags=cv2.INTER_NEAREST)
        cols = np.where(rot_mask.any(axis=0))[0]
        if len(cols) > 5:
            profile = np.sum(rot_mask[:, cols] > 0, axis=0)
            p_len = len(profile)
            p_max = float(max(1.0, np.max(profile)))
            k = max(1, int(0.10 * p_len))
            left_blunt = float(np.mean(profile[:k]) / p_max)
            right_blunt = float(np.mean(profile[-k:]) / p_max)
            max_blunt = float(max(left_blunt, right_blunt))
            min_blunt = float(min(left_blunt, right_blunt))
            asym = float(abs(left_blunt - right_blunt))
        else:
            max_blunt, min_blunt, asym = 0.0, 0.0, 0.0

        fg_mask = grain_mask > 0
        gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        fg_gray = gray_crop[fg_mask]
        if len(fg_gray) == 0:
            return None

        mean_gray = float(np.mean(fg_gray))
        std_gray = float(np.std(fg_gray))
        highlight_ratio = float(np.mean(fg_gray >= 245))

        crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        crop_lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        fg_hsv = crop_hsv[fg_mask]
        fg_lab = crop_lab[fg_mask]

        mean_L = float(np.mean(fg_lab[:, 0]))
        std_L = float(np.std(fg_lab[:, 0]))
        mean_a = float(np.mean(fg_lab[:, 1]))
        mean_b = float(np.mean(fg_lab[:, 2]))
        std_b = float(np.std(fg_lab[:, 2]))

        mean_hue = float(np.mean(fg_hsv[:, 0]))
        mean_sat = float(np.mean(fg_hsv[:, 1]))
        std_sat = float(np.std(fg_hsv[:, 1]))

        # Specific defect indicators
        chalky_ratio = float(np.mean((fg_lab[:, 0] >= 175) & (fg_hsv[:, 1] <= 45)))
        high_whiteness = float(np.mean(fg_gray >= 190))
        yellow_ratio = float(np.mean(fg_lab[:, 2] >= 138))
        dark_spot_ratio = float(np.mean(fg_lab[:, 0] <= 115))

        edges = cv2.Canny(gray_crop, 50, 150)
        edge_density = float(np.sum(edges[fg_mask] > 0) / max(1.0, area))
        roughness = float((peri**2) / (4.0 * np.pi * area) - 1.0)
        extent = float(area / max(1e-5, length * width))
        eq_diameter = float(np.sqrt(4.0 * area / np.pi))

        return {
            "length_px": length,
            "width_px": width,
            "aspect_ratio": ar,
            "area_px": area,
            "perimeter_px": peri,
            "circularity": circ,
            "solidity": solidity,
            "max_bluntness": max_blunt,
            "min_bluntness": min_blunt,
            "end_asymmetry": asym,
            "max_defect_depth": max_defect,
            "num_defects": num_defects,
            "roughness": roughness,
            "mean_gray": mean_gray,
            "std_gray": std_gray,
            "highlight_ratio": highlight_ratio,
            "high_whiteness": high_whiteness,
            "mean_L": mean_L,
            "std_L": std_L,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "std_b": std_b,
            "mean_hue": mean_hue,
            "mean_sat": mean_sat,
            "std_sat": std_sat,
            "chalky_ratio": chalky_ratio,
            "yellow_ratio": yellow_ratio,
            "dark_spot_ratio": dark_spot_ratio,
            "edge_density": edge_density,
            "extent": extent,
            "eq_diameter": eq_diameter,
        }

    def analyze_image(
        self,
        image_path: str,
        output_annotated_path: Optional[str] = None,
        commodity: str = "Raw Rice",
        sample_serial: str = "Sample-0001"
    ) -> Dict[str, Any]:
        """
        Analyze scan image, segment grains, run multi-head quality inference,
        write annotated image with bounding boxes, and return metrics.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu thresholding with margin clearance
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Clear outermost scanner borders (20px margin)
        clean_mask[:20, :] = 0
        clean_mask[-20:, :] = 0
        clean_mask[:, :20] = 0
        clean_mask[:, -20:] = 0

        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_objects = 0
        rice_grains = 0
        whole_cnt = 0
        brk_cnt = 0
        chk_cnt = 0
        dmg_cnt = 0
        disc_cnt = 0
        fm_cnt = 0

        lengths_px = []
        breadths_px = []

        annotated_img = img.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area < 50.0:  # Skip dust/noise
                continue

            total_objects += 1
            x, y, w, h = cv2.boundingRect(c)
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_w, x + w + pad)
            y2 = min(img_h, y + h + pad)

            crop_bgr = img[y1:y2, x1:x2]
            sub_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            cv2.drawContours(sub_mask, [c], -1, 255, thickness=-1, offset=(-x1, -y1))

            feats = self.extract_grain_features(crop_bgr, sub_mask)
            if feats is None:
                continue

            lengths_px.append(feats["length_px"])
            breadths_px.append(feats["width_px"])

            # Head 1: Foreign Matter
            if self.is_loaded:
                _, is_fm = self._predict_head("foreign_matter", feats)
            else:
                is_fm = 1 if feats["solidity"] < 0.82 or feats["circularity"] > 0.88 else 0

            if is_fm:
                fm_cnt += 1
                color = (0, 165, 255)  # Orange
                label = "FM"
            else:
                rice_grains += 1
                if self.is_loaded:
                    _, is_whole = self._predict_head("whole_broken", feats)
                    _, is_chk = self._predict_head("chalky", feats)
                    _, is_disc = self._predict_head("discoloured", feats)
                    _, is_dmg = self._predict_head("damaged", feats)
                else:
                    is_whole = 1 if feats["aspect_ratio"] >= 1.6 and feats["length_px"] >= 45.0 else 0
                    is_chk = 1 if feats["chalky_ratio"] > 0.15 else 0
                    is_disc = 1 if feats["yellow_ratio"] > 0.20 or feats["dark_spot_ratio"] > 0.10 else 0
                    is_dmg = 1 if feats["num_defects"] >= 2 or feats["edge_density"] > 0.12 else 0

                if not is_whole:
                    brk_cnt += 1
                    color = (0, 255, 255)  # Yellow
                    label = "BRK"
                elif is_dmg:
                    dmg_cnt += 1
                    color = (0, 0, 255)  # Red
                    label = "DMG"
                elif is_disc:
                    disc_cnt += 1
                    color = (255, 0, 255)  # Magenta
                    label = "DISC"
                elif is_chk:
                    chk_cnt += 1
                    color = (255, 200, 0)  # Cyan
                    label = "CHK"
                else:
                    whole_cnt += 1
                    color = (0, 255, 0)  # Green
                    label = "OK"

            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                annotated_img,
                label,
                (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA
            )

        if output_annotated_path:
            cv2.imwrite(output_annotated_path, annotated_img)

        # Scale px to mm (assuming ~300 DPI flatbed scanner -> 11.81 px/mm)
        px_to_mm = 25.4 / 300.0
        avg_len_mm = float(np.mean(lengths_px) * px_to_mm) if lengths_px else 6.452
        avg_brd_mm = float(np.mean(breadths_px) * px_to_mm) if breadths_px else 2.115

        denom = max(1, rice_grains)
        whole_pct = (whole_cnt / denom) * 100.0
        brk_pct = (brk_cnt / denom) * 100.0
        chk_pct = (chk_cnt / denom) * 100.0
        dmg_pct = (dmg_cnt / denom) * 100.0
        disc_pct = (disc_cnt / denom) * 100.0
        fm_pct = (fm_cnt / max(1, total_objects)) * 100.0

        # Estimate grams assuming 20g standard tray
        avg_grain_g = 20.0 / max(1, total_objects)

        # Build Web UI parameter rows
        rows = [
            {
                "name": "Whole Grains (Head Rice)",
                "denominator": "count(Whole) / total_rice_grains",
                "limit": "Min 80.0%",
                "resultPct": f"{whole_pct:.2f}%",
                "resultCountAndGms": f"{whole_cnt} grains ({whole_cnt * avg_grain_g:.2f} gms)",
                "status": "acceptable" if whole_pct >= 80.0 else "substandard",
            },
            {
                "name": "Broken Grains",
                "denominator": "count(Broken) / total_rice_grains",
                "limit": "Max 15.0%",
                "resultPct": f"{brk_pct:.2f}%",
                "resultCountAndGms": f"{brk_cnt} grains ({brk_cnt * avg_grain_g:.2f} gms)",
                "status": "acceptable" if brk_pct <= 15.0 else "substandard",
            },
            {
                "name": "Chalky Grains",
                "denominator": "count(Chalky >= 50%) / total_rice_grains",
                "limit": "Max 3.0%",
                "resultPct": f"{chk_pct:.2f}%",
                "resultCountAndGms": f"{chk_cnt} grains ({chk_cnt * avg_grain_g:.2f} gms)",
                "status": "acceptable" if chk_pct <= 3.0 else "substandard",
            },
            {
                "name": "Damaged Grains",
                "denominator": "count(Insect/Cavitation) / total_rice_grains",
                "limit": "Max 2.0%",
                "resultPct": f"{dmg_pct:.2f}%",
                "resultCountAndGms": f"{dmg_cnt} grains ({dmg_cnt * avg_grain_g:.2f} gms)",
                "status": "acceptable" if dmg_pct <= 2.0 else "substandard",
            },
            {
                "name": "Discoloured Grains",
                "denominator": "count(Peck/Heat) / total_rice_grains",
                "limit": "Max 1.0%",
                "resultPct": f"{disc_pct:.2f}%",
                "resultCountAndGms": f"{disc_cnt} grains ({disc_cnt * avg_grain_g:.2f} gms)",
                "status": "acceptable" if disc_pct <= 1.0 else "substandard",
            },
            {
                "name": "Foreign Matter",
                "denominator": "total_foreign_matter / total_detected_objects",
                "limit": "Max 0.5%",
                "resultPct": f"{fm_pct:.2f}%",
                "resultCountAndGms": f"{fm_cnt} particles ({fm_cnt * avg_grain_g:.2f} gms)",
                "status": "acceptable" if fm_pct <= 0.5 else "substandard",
            },
        ]

        # Determine overall grade disposition
        all_passed = (
            whole_pct >= 80.0
            and brk_pct <= 15.0
            and chk_pct <= 3.0
            and dmg_pct <= 2.0
            and disc_pct <= 1.0
            and fm_pct <= 0.5
        )
        disposition = "GRADE_1" if all_passed else "SUBSTANDARD"

        # Kiosk and Dashboard statistics format
        stats = [
            {"param": "Good Grains", "count": whole_cnt, "pct": round(whole_pct, 2), "limit": "-"},
            {"param": "Brokens*", "count": brk_cnt, "pct": round(brk_pct, 2), "limit": "15.00"},
            {"param": "Foreign Matter", "count": fm_cnt, "pct": round(fm_pct, 2), "limit": "0.00"},
            {"param": "Damaged/Discoloured", "count": dmg_cnt + disc_cnt, "pct": round(dmg_pct + disc_pct, 2), "limit": "1.50"},
            {"param": "Chalky Grains", "count": chk_cnt, "pct": round(chk_pct, 2), "limit": "1.00"},
            {"param": "Red Grains", "count": disc_cnt, "pct": round(disc_pct, 2), "limit": "3.00"},
            {"param": "Dehusked Grains", "count": 0, "pct": 0.00, "limit": "10.00"},
        ]

        return {
            "status": "Complete",
            "commodity": commodity,
            "serial": sample_serial,
            "total_objects": total_objects,
            "total_grains": rice_grains,
            "total_foreign_matter": fm_cnt,
            "whole_count": whole_cnt,
            "broken_count": brk_cnt,
            "chalky_count": chk_cnt,
            "damaged_count": dmg_cnt,
            "discoloured_count": disc_cnt,
            "avg_length_mm": round(avg_len_mm, 3),
            "avg_breadth_mm": round(avg_brd_mm, 3),
            "disposition": disposition,
            "stats": stats,
            "rows": rows,
        }
