# Multi-head quality classification engine for RiceAge Milestone M3.
# Decoupled heads: Foreign Matter, Whole vs Broken, Chalky, Damaged, and Discoloured.

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional, Union
import os
import json
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import find_objects

from src.quality_grading import GrainClassification

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


@dataclass
class GrainQualityFeatures:
    """Complete morphological, photometric, color, and texture feature vector for a grain."""
    # Identification
    grain_id: str
    image_name: str
    category: str
    gt_condition: str

    # Morphology & Silhouette
    length_px: float
    width_px: float
    aspect_ratio: float
    area_px: float
    perimeter_px: float
    circularity: float
    solidity: float
    max_bluntness: float
    min_bluntness: float
    end_asymmetry: float
    max_defect_depth: float
    num_defects: int
    roughness: float

    # Photometry & Grayscale
    mean_gray: float
    std_gray: float
    highlight_ratio: float
    high_whiteness: float

    # Color Space (CIELAB & HSV)
    mean_L: float
    std_L: float
    mean_a: float
    mean_b: float
    std_b: float
    mean_hue: float
    mean_sat: float
    std_sat: float

    # Targeted Defect Indicators
    chalky_ratio: float
    yellow_ratio: float
    dark_spot_ratio: float
    edge_density: float

    # Morphological derived features
    extent: float = 0.0
    eq_diameter: float = 0.0


def extract_quality_features_from_crop(
    crop_bgr: np.ndarray,
    grain_mask: np.ndarray,
    grain_id: str = "GRAIN_UNKNOWN",
    image_name: str = "UNKNOWN.jpg",
    category: str = "unknown",
    gt_condition: str = "Unknown"
) -> Optional[GrainQualityFeatures]:
    """
    Extract comprehensive physical, photometric, color, and texture features
    from an isolated binary mask and corresponding BGR crop.
    """
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

    # Convexity defects with exception handling for non-monotone hulls
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

    # Major-axis alignment for end-profile analysis
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

    return GrainQualityFeatures(
        grain_id=grain_id,
        image_name=image_name,
        category=category,
        gt_condition=gt_condition,
        length_px=length,
        width_px=width,
        aspect_ratio=ar,
        area_px=area,
        perimeter_px=peri,
        circularity=circ,
        solidity=solidity,
        max_bluntness=max_blunt,
        min_bluntness=min_blunt,
        end_asymmetry=asym,
        max_defect_depth=max_defect,
        num_defects=num_defects,
        roughness=roughness,
        mean_gray=mean_gray,
        std_gray=std_gray,
        highlight_ratio=highlight_ratio,
        high_whiteness=high_whiteness,
        mean_L=mean_L,
        std_L=std_L,
        mean_a=mean_a,
        mean_b=mean_b,
        std_b=std_b,
        mean_hue=mean_hue,
        mean_sat=mean_sat,
        std_sat=std_sat,
        chalky_ratio=chalky_ratio,
        yellow_ratio=yellow_ratio,
        dark_spot_ratio=dark_spot_ratio,
        edge_density=edge_density,
        extent=extent,
        eq_diameter=eq_diameter
    )


# -------------------------------------------------------------------------
# CLASSICAL-CV BASELINES
# -------------------------------------------------------------------------

class BaselineForeignMatterRuleClassifier:
    """
    Classical rule baseline separating rice from non-rice contaminants
    based on physical morphology, solidity, roughness, and non-rice color.
    """
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        is_fm = (
            (df["solidity"] < 0.88) |
            (df["roughness"] > 1.80) |
            ((df["yellow_ratio"] > 0.50) & (df["aspect_ratio"] > 3.0)) |
            ((df["area_px"] < 300) & (df["mean_L"] < 120)) |
            (df["area_px"] > 3000)
        )
        return is_fm.values.astype(int)


class BaselineChalkyRuleClassifier:
    """
    Classical rule baseline identifying chalky/opaque endosperm
    based on normalized lightness L* and low saturation (opaque milky white).
    Codex CXS 198-1995: kernels with >= 50% chalky area.
    """
    def __init__(self, chalky_threshold: float = 0.50):
        self.chalky_threshold = chalky_threshold

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        is_chalky = (df["chalky_ratio"] >= self.chalky_threshold) | (df["high_whiteness"] >= 0.35)
        return is_chalky.values.astype(int)


class BaselineDiscolouredRuleClassifier:
    """
    Classical rule baseline identifying discoloured/yellowed/peck kernels
    based on yellow chroma (b* in CIELAB), saturation, and dark blemishes.
    """
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        is_discoloured = (
            (df["mean_b"] >= 133.5) |
            (df["mean_sat"] >= 22.0) |
            (df["dark_spot_ratio"] >= 0.03)
        )
        return is_discoloured.values.astype(int)


class BaselineDamagedRuleClassifier:
    """
    Classical rule baseline identifying physical damage
    based on internal edge density (cracks), perimeter roughness, and cavitation defects.
    """
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        is_damaged = (
            (df["edge_density"] >= 0.08) |
            (df["roughness"] >= 0.80) |
            (df["num_defects"] >= 2)
        )
        return is_damaged.values.astype(int)


# -------------------------------------------------------------------------
# SUPERVISED FEATURE-BASED CLASSIFIERS
# -------------------------------------------------------------------------

class SupervisedQualityHead:
    """
    Generic leak-free supervised classification head with internal scaling.
    """
    def __init__(self, feature_cols: List[str], class_weight: str = "balanced", C: float = 1.0):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(class_weight=class_weight, C=C, max_iter=1000, random_state=42)

    def fit(self, train_df: pd.DataFrame, y_train: np.ndarray):
        X_train = self.scaler.fit_transform(train_df[self.feature_cols].values)
        self.clf.fit(X_train, y_train)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X_test = self.scaler.transform(test_df[self.feature_cols].values)
        return self.clf.predict(X_test)

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        X_test = self.scaler.transform(test_df[self.feature_cols].values)
        return self.clf.predict_proba(X_test)[:, 1]


def get_default_feature_sets() -> Dict[str, List[str]]:
    return {
        "foreign_matter": [
            "solidity", "circularity", "roughness", "edge_density",
            "aspect_ratio", "length_px", "area_px", "yellow_ratio",
            "mean_sat", "mean_b", "mean_L"
        ],
        "whole_broken": [
            "length_px", "width_px", "aspect_ratio", "area_px",
            "perimeter_px", "circularity", "solidity", "extent",
            "eq_diameter", "max_bluntness", "min_bluntness", "end_asymmetry"
        ],
        "chalky": [
            "mean_L", "std_L", "chalky_ratio", "high_whiteness",
            "mean_gray", "std_gray", "mean_sat", "std_sat"
        ],
        "discoloured": [
            "mean_b", "std_b", "mean_sat", "std_sat",
            "mean_hue", "yellow_ratio", "dark_spot_ratio",
            "mean_L", "chalky_ratio"
        ],
        "damaged": [
            "edge_density", "roughness", "solidity", "circularity",
            "max_defect_depth", "num_defects", "dark_spot_ratio", "std_gray"
        ]
    }


class MultiHeadQualityClassifier:
    """
    Production multi-head quality classifier for Milestone M4.
    Evaluates:
      1. Foreign Matter (non-rice contaminant)
      2. Whole vs Broken rice morphology
      3. Chalky endosperm
      4. Discoloured / peck / yellowed
      5. Damaged (cracked / insect bite / cavitation)

    Loads calibrated weights from configs/quality_classifier_weights.json.
    Falls back gracefully to classical rule baselines if weights are unavailable.
    """
    def __init__(self, weights_path: str = "configs/quality_classifier_weights.json"):
        self.weights_path = weights_path
        self.is_loaded = False
        self.heads: Dict[str, Any] = {}
        self.fallback_warning: Optional[str] = None
        self._load_weights()

    def _load_weights(self) -> None:
        if os.path.exists(self.weights_path):
            try:
                with open(self.weights_path, "r") as f:
                    data = json.load(f)
                self.heads = data.get("heads", {})
                if all(h in self.heads for h in ["foreign_matter", "whole_broken", "chalky", "discoloured", "damaged"]):
                    self.is_loaded = True
                else:
                    self.fallback_warning = f"Missing heads in {self.weights_path}. Using rule baselines."
                    self._init_fallback_rules()
            except Exception as e:
                self.fallback_warning = f"Failed to load weights from {self.weights_path}: {e}. Using rule baselines."
                self._init_fallback_rules()
        else:
            self.fallback_warning = f"Weights file {self.weights_path} not found. Using rule baselines."
            self._init_fallback_rules()

    def _init_fallback_rules(self) -> None:
        self.rule_fm = BaselineForeignMatterRuleClassifier()
        self.rule_ch = BaselineChalkyRuleClassifier()
        self.rule_disc = BaselineDiscolouredRuleClassifier()
        self.rule_dmg = BaselineDamagedRuleClassifier()

    def _predict_head(self, head_name: str, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        h = self.heads[head_name]
        cols = h["feature_cols"]
        thresh = float(h.get("threshold", 0.50))
        X = df[cols].values.astype(float)

        # High-accuracy non-linear GBDT tree ensemble support
        if h.get("model_type") == "gbdt" and "trees" in h:
            raw_scores = np.full(len(X), float(h.get("init_bias", 0.0)), dtype=float)
            lr = float(h.get("learning_rate", 0.1))
            for tree in h["trees"]:
                left = np.array(tree["children_left"])
                right = np.array(tree["children_right"])
                feat = np.array(tree["feature"])
                th = np.array(tree["threshold"])
                val = np.array(tree["value"])

                node = np.zeros(len(X), dtype=int)
                while True:
                    active = left[node] != -1
                    if not np.any(active):
                        break
                    active_idx = np.where(active)[0]
                    curr_node = node[active_idx]
                    split_feat = feat[curr_node]
                    split_th = th[curr_node]
                    go_left = X[active_idx, split_feat] <= split_th
                    node[active_idx] = np.where(go_left, left[curr_node], right[curr_node])
                raw_scores += lr * val[node]

            raw_scores = np.clip(raw_scores, -500.0, 500.0)
            probs = 1.0 / (1.0 + np.exp(-raw_scores))
            preds = (probs >= thresh).astype(int)
            return probs, preds

        # Linear regression fallback
        mean = np.array(h["scaler_mean"], dtype=float)
        scale = np.array(h["scaler_scale"], dtype=float)
        coef = np.array(h["coef"], dtype=float)
        intercept = float(h["intercept"])

        scale_safe = np.where(scale == 0, 1.0, scale)
        Z = (X - mean) / scale_safe
        logits = np.dot(Z, coef) + intercept
        logits = np.clip(logits, -500.0, 500.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= thresh).astype(int)
        return probs, preds

    def predict_features_df(self, df: pd.DataFrame) -> List[GrainClassification]:
        """Classify a DataFrame of extracted grain features into GrainClassification objects."""
        if df.empty:
            return []

        working_df = df.copy()
        if "extent" not in working_df.columns:
            working_df["extent"] = working_df["area_px"] / np.maximum(1e-5, working_df["length_px"] * working_df["width_px"])
        if "eq_diameter" not in working_df.columns:
            working_df["eq_diameter"] = np.sqrt(4.0 * working_df["area_px"] / np.pi)

        n_samples = len(working_df)
        if self.is_loaded:
            fm_probs, fm_preds = self._predict_head("foreign_matter", working_df)
            wb_probs, wb_preds = self._predict_head("whole_broken", working_df)
            ch_probs, ch_preds = self._predict_head("chalky", working_df)
            disc_probs, disc_preds = self._predict_head("discoloured", working_df)
            dmg_probs, dmg_preds = self._predict_head("damaged", working_df)
        else:
            fm_preds = self.rule_fm.predict(working_df)
            fm_probs = fm_preds.astype(float)
            wb_preds = ((working_df["aspect_ratio"] >= 1.6) & (working_df["length_px"] >= 45.0)).values.astype(int)
            wb_probs = wb_preds.astype(float)
            ch_preds = self.rule_ch.predict(working_df)
            ch_probs = ch_preds.astype(float)
            disc_preds = self.rule_disc.predict(working_df)
            disc_probs = disc_preds.astype(float)
            dmg_preds = self.rule_dmg.predict(working_df)
            dmg_probs = dmg_preds.astype(float)

        classifications: List[GrainClassification] = []
        for i in range(n_samples):
            obj_id = str(working_df.iloc[i].get("grain_id", f"OBJ_{i+1:04d}"))
            is_fm = bool(fm_preds[i])

            conf = {
                "foreign_matter": float(fm_probs[i]),
                "whole": float(wb_probs[i]),
                "chalky": float(ch_probs[i]),
                "discoloured": float(disc_probs[i]),
                "damaged": float(dmg_probs[i])
            }

            if is_fm:
                gc = GrainClassification(
                    object_id=obj_id,
                    is_rice=False,
                    is_foreign_matter=True,
                    morphology="N/A",
                    is_chalky=False,
                    is_damaged=False,
                    is_discoloured=False,
                    confidences=conf
                )
            else:
                morph = "Whole" if wb_preds[i] else "Broken"
                gc = GrainClassification(
                    object_id=obj_id,
                    is_rice=True,
                    is_foreign_matter=False,
                    morphology=morph,
                    is_chalky=bool(ch_preds[i]),
                    is_damaged=bool(dmg_preds[i]),
                    is_discoloured=bool(disc_preds[i]),
                    confidences=conf
                )
            classifications.append(gc)

        return classifications

    def classify_detected_objects(
        self,
        original_bgr: np.ndarray,
        instance_mask: np.ndarray,
        image_name: str = "raw_scan.jpg"
    ) -> Tuple[List[GrainClassification], List[GrainQualityFeatures], pd.DataFrame]:
        """
        Extract crops and features for all detected objects in instance_mask,
        then classify them using the multi-head models.
        """
        max_label = int(instance_mask.max())
        if max_label == 0:
            return [], [], pd.DataFrame()

        slices = find_objects(instance_mask)
        feature_rows: List[Dict[str, Any]] = []
        features_list: List[GrainQualityFeatures] = []

        h_img, w_img = instance_mask.shape[:2]

        for idx, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            y_sl, x_sl = sl
            y1 = max(0, y_sl.start - 2)
            y2 = min(h_img, y_sl.stop + 2)
            x1 = max(0, x_sl.start - 2)
            x2 = min(w_img, x_sl.stop + 2)

            sub_inst = instance_mask[y1:y2, x1:x2]
            c_bgr = original_bgr[y1:y2, x1:x2]
            c_mask = ((sub_inst == idx) * 255).astype(np.uint8)

            grain_id = f"GRAIN_{idx:04d}"
            f = extract_quality_features_from_crop(
                crop_bgr=c_bgr,
                grain_mask=c_mask,
                grain_id=grain_id,
                image_name=image_name
            )
            if f is None:
                continue

            features_list.append(f)
            feature_rows.append(asdict(f))

        if not feature_rows:
            return [], [], pd.DataFrame()

        feat_df = pd.DataFrame(feature_rows)
        classifications = self.predict_features_df(feat_df)
        return classifications, features_list, feat_df
