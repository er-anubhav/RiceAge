# Whole vs Broken grain classification engine, feature extraction, and image-level cross-validation.

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.classification import RealDataTaxonomy


def extract_annotated_grain_silhouette(
    bgr_image: np.ndarray,
    points: List[Tuple[float, float]],
    pad: int = 12
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract the physical grain silhouette from an annotated region.
    Over 91% of ground truth annotations in CVAT are coarse 4-point oriented bounding boxes
    rather than exact grain silhouettes. This function binarizes the actual grain inside the
    coarse region to extract the true physical kernel boundary.
    """
    pts = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    h_img, w_img = bgr_image.shape[:2]

    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w_img, x + w + pad)
    y1 = min(h_img, y + h + pad)

    crop = bgr_image[y0:y1, x0:x1]
    if crop.size == 0:
        return None, None

    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    poly_in_crop = pts - [x0, y0]
    poly_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(poly_mask, [poly_in_crop], 255)

    grain_mask = cv2.bitwise_and(binary, binary, mask=poly_mask)
    return grain_mask, gray_crop


def extract_grain_features_from_mask(
    grain_mask: np.ndarray,
    gray_crop: Optional[np.ndarray] = None
) -> Optional[Dict[str, float]]:
    """
    Extract comprehensive geometric, contour, end-shape, and photometric features
    from an isolated binary grain silhouette mask.
    """
    contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 30.0:
        return None

    peri = float(cv2.arcLength(c, True))
    rect = cv2.minAreaRect(c)
    (cx, cy), (dim1, dim2), angle = rect

    length = float(max(dim1, dim2))
    width = float(min(dim1, dim2))
    aspect_ratio = float(length / max(1e-5, width))
    circularity = float(4.0 * np.pi * area / max(1e-5, peri ** 2))
    circularity = min(1.0, max(0.0, circularity))

    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    hull_peri = float(cv2.arcLength(hull, True))
    solidity = float(area / max(1e-5, hull_area))
    solidity = min(1.0, max(0.0, solidity))

    convexity = float(hull_peri / max(1e-5, peri))
    convexity = min(1.0, max(0.0, convexity))

    bx, by, bw, bh = cv2.boundingRect(c)
    extent = float(area / max(1e-5, bw * bh))
    extent = min(1.0, max(0.0, extent))

    eq_diameter = float(np.sqrt(4.0 * area / np.pi))

    # Major-axis alignment for end-shape bluntness and asymmetry analysis
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
        max_bluntness = float(max(left_blunt, right_blunt))
        min_bluntness = float(min(left_blunt, right_blunt))
        end_asymmetry = float(abs(left_blunt - right_blunt))
    else:
        max_bluntness, min_bluntness, end_asymmetry = 0.0, 0.0, 0.0

    # Photometric features
    mean_int = 0.0
    std_int = 0.0
    highlight_ratio = 0.0
    if gray_crop is not None and gray_crop.shape == grain_mask.shape:
        fg_pix = gray_crop[grain_mask > 0]
        if len(fg_pix) > 0:
            mean_int = float(np.mean(fg_pix))
            std_int = float(np.std(fg_pix))
            highlight_ratio = float(np.mean(fg_pix >= 240))

    return {
        "length_px": length,
        "width_px": width,
        "aspect_ratio": aspect_ratio,
        "area_px": area,
        "perimeter_px": peri,
        "circularity": circularity,
        "solidity": solidity,
        "convexity": convexity,
        "extent": extent,
        "eq_diameter": eq_diameter,
        "max_bluntness": max_bluntness,
        "min_bluntness": min_bluntness,
        "end_asymmetry": end_asymmetry,
        "mean_intensity": mean_int,
        "std_intensity": std_int,
        "highlight_ratio": highlight_ratio,
    }


class BaselineRuleClassifier:
    """
    Existing baseline rule-based classifier.
    Computes whole grain median dynamically from unbroken candidates within each test image.
    When evaluated on pure broken images, this causes severe reference drift.
    """
    def __init__(self, length_ratio_thresh: float = 0.75, aspect_ratio_thresh: float = 1.6):
        self.length_ratio_thresh = length_ratio_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh

    def fit(self, train_df: pd.DataFrame):
        # Baseline does not train; it relies on runtime sample median.
        return self

    def predict_image(self, img_df: pd.DataFrame) -> np.ndarray:
        cands = img_df[
            (img_df.get("solidity", 1.0) >= 0.90) &
            (img_df["aspect_ratio"] >= 1.8)
        ]
        if not cands.empty:
            med_len = float(cands["length_px"].median())
        else:
            med_len = float(img_df["length_px"].median())

        preds = []
        for _, row in img_df.iterrows():
            l = row["length_px"]
            ar = row["aspect_ratio"]
            if l < (self.length_ratio_thresh * med_len) or ar < self.aspect_ratio_thresh:
                preds.append(0)  # Broken
            else:
                preds.append(1)  # Whole
        return np.array(preds, dtype=int)

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        df_local = test_df.reset_index(drop=True)
        preds = np.zeros(len(df_local), dtype=int)
        for _, grp in df_local.groupby("filename", sort=False):
            img_preds = self.predict_image(grp)
            preds[grp.index] = img_preds
        return preds


class CalibratedRuleClassifier:
    """
    Calibrated rule-based classifier.
    Estimates the reference unbroken whole grain length strictly from training fold Whole instances,
    preventing reference collapse on pure broken or broken-heavy images.
    """
    def __init__(self, length_ratio_thresh: float = 0.75, aspect_ratio_thresh: float = 1.6):
        self.length_ratio_thresh = length_ratio_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.reference_whole_length_px: float = 70.0

    def fit(self, train_df: pd.DataFrame):
        whole_train = train_df[train_df["gt_condition"] == "Whole"]
        if not whole_train.empty:
            self.reference_whole_length_px = float(whole_train["length_px"].median())
        else:
            self.reference_whole_length_px = float(train_df["length_px"].median())
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        thresh_len = self.length_ratio_thresh * self.reference_whole_length_px
        is_whole = (test_df["length_px"] >= thresh_len) & (test_df["aspect_ratio"] >= self.aspect_ratio_thresh)
        return is_whole.astype(int).values


class OptimizedRuleClassifier:
    """
    Rule-based classifier with joint grid search inside the training fold
    for optimal length ratio threshold and aspect ratio threshold.
    """
    def __init__(self):
        self.reference_whole_length_px: float = 70.0
        self.best_length_ratio: float = 0.75
        self.best_aspect_ratio: float = 1.6

    def fit(self, train_df: pd.DataFrame):
        whole_train = train_df[train_df["gt_condition"] == "Whole"]
        if not whole_train.empty:
            self.reference_whole_length_px = float(whole_train["length_px"].median())
        else:
            self.reference_whole_length_px = float(train_df["length_px"].median())

        y_true = train_df["y"].values
        best_score = -1.0

        for l_ratio in np.linspace(0.65, 0.85, 21):
            for ar_t in np.linspace(1.4, 2.2, 17):
                th = l_ratio * self.reference_whole_length_px
                y_p = ((train_df["length_px"] >= th) & (train_df["aspect_ratio"] >= ar_t)).astype(int).values
                score = balanced_accuracy_score(y_true, y_p)
                if score > best_score:
                    best_score = score
                    self.best_length_ratio = float(l_ratio)
                    self.best_aspect_ratio = float(ar_t)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        thresh_len = self.best_length_ratio * self.reference_whole_length_px
        is_whole = (test_df["length_px"] >= thresh_len) & (test_df["aspect_ratio"] >= self.best_aspect_ratio)
        return is_whole.astype(int).values


class SupervisedGrainClassifier:
    """
    Supervised machine learning classifier with internal StandardScaler.
    Feature scaling and model fitting are strictly confined to the training fold.
    """
    def __init__(
        self,
        model_type: str = "logistic_regression",
        feature_cols: Optional[List[str]] = None,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.feature_cols = feature_cols or [
            "length_px", "width_px", "aspect_ratio", "area_px", "perimeter_px",
            "circularity", "solidity", "extent", "eq_diameter",
            "max_bluntness", "min_bluntness", "end_asymmetry"
        ]
        self.scaler = StandardScaler()
        if model_type == "logistic_regression":
            self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        elif model_type == "svm_rbf":
            self.model = SVC(C=1.0, kernel="rbf", random_state=random_state)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=random_state)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, train_df: pd.DataFrame):
        X = self.scaler.fit_transform(train_df[self.feature_cols].values)
        y = train_df["y"].values
        self.model.fit(X, y)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(test_df[self.feature_cols].values)
        return self.model.predict(X)

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(test_df[self.feature_cols].values)
        return self.model.predict_proba(X)


@dataclass
class FoldMetrics:
    fold: int
    train_images: int
    val_images: int
    train_samples: int
    val_samples: int
    accuracy: float
    balanced_accuracy: float
    whole_precision: float
    whole_recall: float
    whole_f1: float
    broken_precision: float
    broken_recall: float
    broken_f1: float
    macro_f1: float


@dataclass
class CrossValidationEvaluationResult:
    model_name: str
    evaluation_level: str
    fold_metrics: List[FoldMetrics]
    pooled_accuracy: float
    pooled_balanced_accuracy: float
    pooled_confusion_matrix: np.ndarray  # [[TN_broken, FP_whole], [FN_broken, TP_whole]] or [[TP_whole, FN_broken], [FP_whole, TN_broken]]
    pooled_whole_precision: float
    pooled_whole_recall: float
    pooled_whole_f1: float
    pooled_broken_precision: float
    pooled_broken_recall: float
    pooled_broken_f1: float
    pooled_macro_f1: float
    mean_fold_accuracy: float
    std_fold_accuracy: float
    mean_fold_balanced_accuracy: float
    std_fold_balanced_accuracy: float
    mean_fold_macro_f1: float
    std_fold_macro_f1: float
    out_of_fold_predictions: np.ndarray

    def to_summary_row(self) -> Dict[str, Any]:
        cm = self.pooled_confusion_matrix
        # cm is 2x2: row 0=Whole (y=1), row 1=Broken (y=0); col 0=Pred Whole (1), col 1=Pred Broken (0)
        tp_w = int(cm[0, 0])
        fn_w = int(cm[0, 1])
        fp_w = int(cm[1, 0])
        tn_w = int(cm[1, 1])

        return {
            "model_name": self.model_name,
            "evaluation_level": self.evaluation_level,
            "total_samples": len(self.out_of_fold_predictions),
            "tp_whole": tp_w,
            "fn_broken_as_whole": fp_w,
            "fn_whole_as_broken": fn_w,
            "tn_broken": tn_w,
            "pooled_accuracy": round(self.pooled_accuracy, 4),
            "pooled_balanced_accuracy": round(self.pooled_balanced_accuracy, 4),
            "whole_precision": round(self.pooled_whole_precision, 4),
            "whole_recall": round(self.pooled_whole_recall, 4),
            "whole_f1": round(self.pooled_whole_f1, 4),
            "broken_precision": round(self.pooled_broken_precision, 4),
            "broken_recall": round(self.pooled_broken_recall, 4),
            "broken_f1": round(self.pooled_broken_f1, 4),
            "macro_f1": round(self.pooled_macro_f1, 4),
            "fold_accuracy_mean": round(self.mean_fold_accuracy, 4),
            "fold_accuracy_std": round(self.std_fold_accuracy, 4),
            "fold_balanced_acc_mean": round(self.mean_fold_balanced_accuracy, 4),
            "fold_balanced_acc_std": round(self.std_fold_balanced_accuracy, 4),
            "fold_macro_f1_mean": round(self.mean_fold_macro_f1, 4),
            "fold_macro_f1_std": round(self.std_fold_macro_f1, 4),
        }


def run_stratified_grouped_cv(
    df: pd.DataFrame,
    model_factory,
    model_name: str,
    evaluation_level: str = "Pure_Classification_GT_Instances",
    n_splits: int = 5,
    random_state: int = 42
) -> CrossValidationEvaluationResult:
    """
    Executes leak-free Stratified Grouped K-Fold cross validation.
    Splits data strictly by source image (using group_id to ensure duplicate
    image pairs whole 4.jpg and whole 5.jpg belong to the exact same fold).
    All training, scaling, and threshold tuning are executed strictly inside train folds.
    """
    df_eval = df.copy().reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds = np.zeros(len(df_eval), dtype=int)
    fold_metric_list: List[FoldMetrics] = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_eval, df_eval["category"], groups=df_eval["group_id"]), start=1):
        train_df = df_eval.iloc[train_idx].copy().reset_index(drop=True)
        val_df = df_eval.iloc[val_idx].copy().reset_index(drop=True)

        model = model_factory()
        model.fit(train_df)
        preds = model.predict(val_df)
        oof_preds[val_idx] = preds

        y_val = val_df["y"].values
        acc = float(accuracy_score(y_val, preds))
        bal_acc = float(balanced_accuracy_score(y_val, preds))
        prec_w = float(precision_score(y_val, preds, pos_label=1, zero_division=0))
        rec_w = float(recall_score(y_val, preds, pos_label=1, zero_division=0))
        f1_w = float(f1_score(y_val, preds, pos_label=1, zero_division=0))
        prec_b = float(precision_score(y_val, preds, pos_label=0, zero_division=0))
        rec_b = float(recall_score(y_val, preds, pos_label=0, zero_division=0))
        f1_b = float(f1_score(y_val, preds, pos_label=0, zero_division=0))
        m_f1 = float((f1_w + f1_b) / 2.0)

        fold_metric_list.append(FoldMetrics(
            fold=fold,
            train_images=int(train_df["filename"].nunique()),
            val_images=int(val_df["filename"].nunique()),
            train_samples=len(train_df),
            val_samples=len(val_df),
            accuracy=acc,
            balanced_accuracy=bal_acc,
            whole_precision=prec_w,
            whole_recall=rec_w,
            whole_f1=f1_w,
            broken_precision=prec_b,
            broken_recall=rec_b,
            broken_f1=f1_b,
            macro_f1=m_f1,
        ))

    y_all = df_eval["y"].values
    pooled_acc = float(accuracy_score(y_all, oof_preds))
    pooled_bal_acc = float(balanced_accuracy_score(y_all, oof_preds))
    pooled_prec_w = float(precision_score(y_all, oof_preds, pos_label=1, zero_division=0))
    pooled_rec_w = float(recall_score(y_all, oof_preds, pos_label=1, zero_division=0))
    pooled_f1_w = float(f1_score(y_all, oof_preds, pos_label=1, zero_division=0))
    pooled_prec_b = float(precision_score(y_all, oof_preds, pos_label=0, zero_division=0))
    pooled_rec_b = float(recall_score(y_all, oof_preds, pos_label=0, zero_division=0))
    pooled_f1_b = float(f1_score(y_all, oof_preds, pos_label=0, zero_division=0))
    pooled_m_f1 = float((pooled_f1_w + pooled_f1_b) / 2.0)

    # Confusion matrix: label 1=Whole, label 0=Broken
    # row 0 = GT Whole (1), row 1 = GT Broken (0)
    # col 0 = Pred Whole (1), col 1 = Pred Broken (0)
    cm = confusion_matrix(y_all, oof_preds, labels=[1, 0])

    acc_folds = [fm.accuracy for fm in fold_metric_list]
    bal_acc_folds = [fm.balanced_accuracy for fm in fold_metric_list]
    mf1_folds = [fm.macro_f1 for fm in fold_metric_list]

    return CrossValidationEvaluationResult(
        model_name=model_name,
        evaluation_level=evaluation_level,
        fold_metrics=fold_metric_list,
        pooled_accuracy=pooled_acc,
        pooled_balanced_accuracy=pooled_bal_acc,
        pooled_confusion_matrix=cm,
        pooled_whole_precision=pooled_prec_w,
        pooled_whole_recall=pooled_rec_w,
        pooled_whole_f1=pooled_f1_w,
        pooled_broken_precision=pooled_prec_b,
        pooled_broken_recall=pooled_rec_b,
        pooled_broken_f1=pooled_f1_b,
        pooled_macro_f1=pooled_m_f1,
        mean_fold_accuracy=float(np.mean(acc_folds)),
        std_fold_accuracy=float(np.std(acc_folds)),
        mean_fold_balanced_accuracy=float(np.mean(bal_acc_folds)),
        std_fold_balanced_accuracy=float(np.std(bal_acc_folds)),
        mean_fold_macro_f1=float(np.mean(mf1_folds)),
        std_fold_macro_f1=float(np.std(mf1_folds)),
        out_of_fold_predictions=oof_preds,
    )
