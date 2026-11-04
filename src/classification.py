# Grain quality taxonomy, rule-based morphology, and deep-learning multi-defect classifier.

import os
from enum import Enum
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import cv2
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    transforms = None
    models = None
    _TORCH_AVAILABLE = False


class RealDataTaxonomy(str, Enum):
    # Taxonomy verified and present in current controlled scanner clean rice data.
    WHOLE_RICE = "WHOLE_RICE"
    BROKEN_RICE = "BROKEN_RICE"


class FutureDefectTaxonomy(str, Enum):
    # Target defect taxonomy for supervised models with labeled data.
    WHOLE_RICE = "WHOLE_RICE"
    BROKEN_RICE = "BROKEN_RICE"
    CHALKY_RICE = "CHALKY_RICE"
    DISCOLORED_RICE = "DISCOLORED_RICE"
    DAMAGED_RICE = "DAMAGED_RICE"
    PEST_DAMAGED = "PEST_DAMAGED"
    UNRIPE_RICE = "UNRIPE_RICE"
    FOREIGN_MATERIAL = "FOREIGN_MATERIAL"
    OTHER_GRAIN = "OTHER_GRAIN"


AI_CLASSES = [
    "broken",
    "chalky",
    "damaged",
    "dehusked",
    "discoloured",
    "foreign_matter",
    "whole"
]

_CACHED_RESNET = None
_CACHED_DEVICE = None
_CACHED_ONNX_SESS = None
_CACHED_ONNX_PATH = None


def _get_onnx_session(onnx_path: str):
    global _CACHED_ONNX_SESS, _CACHED_ONNX_PATH
    if _CACHED_ONNX_SESS is not None and _CACHED_ONNX_PATH == onnx_path:
        return _CACHED_ONNX_SESS

    if not os.path.exists(onnx_path):
        return None

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = [p for p in ["DmlExecutionProvider", "CPUExecutionProvider"] if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        sess = ort.InferenceSession(onnx_path, providers=providers)
        _CACHED_ONNX_SESS = sess
        _CACHED_ONNX_PATH = onnx_path
        return sess
    except Exception as e:
        print(f"Warning: Failed to load ONNX session from {onnx_path}: {e}")
        return None


def _get_classifier_model(weights_path: str, device_str: Optional[str] = None):
    global _CACHED_RESNET, _CACHED_DEVICE
    if not _TORCH_AVAILABLE:
        return None, None
    device = torch.device(device_str if device_str else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    if _CACHED_RESNET is not None and _CACHED_DEVICE == device:
        return _CACHED_RESNET, device

    if not os.path.exists(weights_path):
        return None, device

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(AI_CLASSES))
    )
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _CACHED_RESNET = model
    _CACHED_DEVICE = device
    return model, device


def _make_square_crop_224(crop: np.ndarray, target_size: int = 224) -> np.ndarray:
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    max_dim = max(h, w)
    canvas = np.zeros((max_dim, max_dim, 3), dtype=crop.dtype)
    y_off = (max_dim - h) // 2
    x_off = (max_dim - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = crop
    return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4 if max_dim < target_size else cv2.INTER_AREA)


def classify_grains_ai(
    measurements_df: pd.DataFrame,
    original_bgr: np.ndarray,
    weights_path: str = "runs/stage2_resnet50/best_resnet50.pth",
    broken_length_ratio_thresh: float = 0.75,
    broken_aspect_ratio_thresh: float = 1.6
) -> pd.DataFrame:
    # Multi-Attribute AI Classifier: Combines physical morphology with ResNet-50 defect detection.
    df = measurements_df.copy()
    if df.empty:
        df["quality_class"] = []
        df["defects"] = []
        df["is_sound"] = []
        df["confidence"] = []
        df["classification_reason"] = []
        return df

    is_onnx = weights_path.endswith(".onnx")
    onnx_sess = _get_onnx_session(weights_path) if is_onnx else None
    model, device = (None, None) if is_onnx else _get_classifier_model(weights_path)

    if onnx_sess is None and model is None:
        # Fallback to rule-based if weights not found
        return classify_grains_rule_based(
            measurements_df,
            broken_length_ratio_thresh=broken_length_ratio_thresh,
            broken_aspect_ratio_thresh=broken_aspect_ratio_thresh
        )

    h_orig, w_orig = original_bgr.shape[:2]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) if (_TORCH_AVAILABLE and not is_onnx) else None

    # Reference unbroken length
    unbroken_candidates = df[
        (df["separation_status"].isin(["isolated_single", "separated_from_cluster"])) &
        (~df["is_boundary"]) &
        (df["solidity"] >= 0.90) &
        (df["aspect_ratio"] >= 1.8)
    ]
    if not unbroken_candidates.empty:
        median_whole_len = float(unbroken_candidates["length_px"].median())
    else:
        non_boundary = df[~df["is_boundary"]] if "is_boundary" in df.columns else df
        median_whole_len = float(non_boundary["length_px"].median()) if not non_boundary.empty else float(df["length_px"].median())

    broken_cutoff = broken_length_ratio_thresh * median_whole_len

    # Extract crops and optical features
    crop_tensors = []
    opt_features = []

    for _, row in df.iterrows():
        bx = int(row.get("x", row.get("bbox_x", 0)))
        by = int(row.get("y", row.get("bbox_y", 0)))
        bw = int(row.get("width", row.get("bbox_w", 0)))
        bh = int(row.get("height", row.get("bbox_h", 0)))

        pad = 6
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w_orig, bx + bw + pad)
        y2 = min(h_orig, by + bh + pad)

        raw_crop = original_bgr[y1:y2, x1:x2]
        sq_crop = _make_square_crop_224(raw_crop, target_size=224)
        sq_rgb = cv2.cvtColor(sq_crop, cv2.COLOR_BGR2RGB)
        if is_onnx:
            norm_rgb = sq_rgb.astype(np.float32) / 255.0
            norm_rgb = (norm_rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            crop_tensors.append(np.transpose(norm_rgb, (2, 0, 1)))
        else:
            crop_tensors.append(transform(sq_rgb))

        # Optical forensic features
        gray = cv2.cvtColor(sq_crop, cv2.COLOR_BGR2GRAY)
        ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inner_mask = cv2.erode(otsu, kernel) > 0
        if np.sum(inner_mask) < 30:
            inner_mask = otsu > 0

        core_bgr = sq_crop[inner_mask]
        core_hsv = cv2.cvtColor(sq_crop, cv2.COLOR_BGR2HSV)[inner_mask]

        r_v = core_bgr[:, 2].astype(np.float32)
        g_v = core_bgr[:, 1].astype(np.float32)
        rg = float(np.mean(r_v / np.maximum(g_v, 1.0)))

        v_vals = core_hsv[:, 2].astype(np.float32)
        s_vals = core_hsv[:, 1].astype(np.float32)
        std_v = float(np.std(v_vals))
        med_v = float(np.median(v_vals))

        pinhole_ratio = float(np.sum(v_vals < 0.60 * med_v)) / max(float(len(v_vals)), 1.0)
        chalky_ratio = float(np.sum((v_vals >= 170) & (s_vals <= 35))) / max(float(len(v_vals)), 1.0)

        opt_features.append({
            "rg_ratio": rg,
            "core_std_v": std_v,
            "pinhole_ratio": pinhole_ratio,
            "chalky_ratio": chalky_ratio
        })

    # Inference in batches (ONNX Runtime vs PyTorch)
    probs_list = []
    if is_onnx:
        batch_size = 128
        for b_start in range(0, len(crop_tensors), batch_size):
            batch = np.stack(crop_tensors[b_start:b_start + batch_size], axis=0)
            logits = onnx_sess.run(None, {"input": batch})[0]
            exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            probs_list.extend(probs)
    else:
        batch_size = 64
        with torch.no_grad():
            for b_start in range(0, len(crop_tensors), batch_size):
                batch = torch.stack(crop_tensors[b_start:b_start + batch_size]).to(device)
                with torch.amp.autocast("cuda") if "cuda" in str(device) else torch.no_grad():
                    logits = model(batch)
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                probs_list.extend(probs)

    tray_median_rg = float(np.median([m["rg_ratio"] for m in opt_features])) if opt_features else 1.0
    tray_median_std = float(np.median([m["core_std_v"] for m in opt_features])) if opt_features else 15.0

    IDX = {name: i for i, name in enumerate(AI_CLASSES)}

    quality_classes = []
    defects_list = []
    is_sound_list = []
    confidences = []
    reasons = []

    for (_, row), prob_vec, opt in zip(df.iterrows(), probs_list, opt_features):
        status = row.get("separation_status", "")
        length = row["length_px"]
        ar = row["aspect_ratio"]

        if status == "under_segmented_cluster":
            quality_classes.append("UNRESOLVED_CLUSTER")
            defects_list.append(["cluster"])
            is_sound_list.append(False)
            confidences.append(1.0)
            reasons.append("Multi-grain cluster unresolved by watershed")
            continue
        elif row.get("is_boundary", False):
            quality_classes.append("BOUNDARY_TRUNCATED")
            defects_list.append(["boundary"])
            is_sound_list.append(False)
            confidences.append(1.0)
            reasons.append("Touches scanner boundary margin")
            continue

        rg_ratio = opt["rg_ratio"]
        core_std_v = opt["core_std_v"]
        pinhole_ratio = opt["pinhole_ratio"]
        chalky_ratio = opt["chalky_ratio"]

        p_broken = prob_vec[IDX["broken"]]
        p_chalky = prob_vec[IDX["chalky"]]
        p_damaged = prob_vec[IDX["damaged"]]
        p_dehusked = prob_vec[IDX["dehusked"]]
        p_discoloured = prob_vec[IDX["discoloured"]]
        p_foreign = prob_vec[IDX["foreign_matter"]]
        p_whole = prob_vec[IDX["whole"]]

        is_broken = (length < broken_cutoff) or (ar < broken_aspect_ratio_thresh)

        has_oxidation = (rg_ratio > 1.025 * tray_median_rg)
        has_mottled = (core_std_v > 1.25 * tray_median_std)
        has_pinhole = (pinhole_ratio > 0.04)

        # Multi-attribute defect flags (Coupled with forensic optical validation)
        is_low_res = (median_whole_len < 65.0)

        grain_defects = []
        if not is_low_res:
            if (p_damaged > 0.35 and (has_oxidation or has_mottled or has_pinhole)) or (p_damaged > 0.65 and (has_oxidation or has_mottled)):
                grain_defects.append("damaged")
            if (p_chalky > p_whole and chalky_ratio > 0.15) or (p_chalky > 0.32 and chalky_ratio > 0.25):
                grain_defects.append("chalky")
            if (p_discoloured > p_whole and has_oxidation) or (p_discoloured > 0.35 and has_oxidation and not has_mottled):
                grain_defects.append("discoloured")
            if p_dehusked > 0.45:
                grain_defects.append("dehusked")
        else:
            # Low-resolution images (downsampled thumbnails): require high confidence + physical signatures
            if p_damaged > 0.75 and (has_oxidation or has_pinhole):
                grain_defects.append("damaged")
            if p_chalky > 0.60 and chalky_ratio > 0.30:
                grain_defects.append("chalky")
            if p_discoloured > 0.70 and has_oxidation:
                grain_defects.append("discoloured")
            if p_dehusked > 0.60:
                grain_defects.append("dehusked")

        # Assign primary Quality Class (FutureDefectTaxonomy)
        if p_foreign > 0.40 or rg_ratio > 1.15:
            q_class = FutureDefectTaxonomy.FOREIGN_MATERIAL.value
            conf = float(p_foreign)
            reason = "Foreign non-grain material detected"
        elif "damaged" in grain_defects:
            q_class = FutureDefectTaxonomy.DAMAGED_RICE.value
            conf = float(p_damaged)
            reason = f"Damaged: oxidation R/G={rg_ratio:.3f}, texture std={core_std_v:.1f}"
        elif "chalky" in grain_defects:
            q_class = FutureDefectTaxonomy.CHALKY_RICE.value
            conf = float(p_chalky)
            reason = f"Chalky endosperm core opacity ratio={chalky_ratio:.2f}"
        elif "discoloured" in grain_defects:
            q_class = FutureDefectTaxonomy.DISCOLORED_RICE.value
            conf = float(p_discoloured)
            reason = f"Discoloured amber/yellow tint R/G={rg_ratio:.3f}"
        elif "dehusked" in grain_defects:
            q_class = FutureDefectTaxonomy.OTHER_GRAIN.value
            conf = float(p_dehusked)
            reason = "Dehusked / Unmilled paddy grain"
        elif is_broken:
            q_class = FutureDefectTaxonomy.BROKEN_RICE.value
            conf = float(max(p_broken, 0.85))
            reason = f"Broken: Length {length:.1f}px < 75% of median ({median_whole_len:.1f}px)"
        else:
            q_class = FutureDefectTaxonomy.WHOLE_RICE.value
            conf = float(p_whole)
            reason = "Sound Whole Rice (Unbroken & Defect-Free)"

        quality_classes.append(q_class)
        defects_list.append(grain_defects)
        is_sound_list.append(len(grain_defects) == 0 and not is_broken)
        confidences.append(round(conf, 3))
        reasons.append(reason)

    df["quality_class"] = quality_classes
    df["defects"] = defects_list
    df["is_sound"] = is_sound_list
    df["confidence"] = confidences
    df["classification_reason"] = reasons

    return df


# Classify grains into WHOLE_RICE, BROKEN_RICE, or UNRESOLVED_CLUSTER based on morphology (Classic Rule-Based).
def classify_grains_rule_based(
    measurements_df: pd.DataFrame,
    broken_length_ratio_thresh: float = 0.75,
    broken_aspect_ratio_thresh: float = 1.6
) -> pd.DataFrame:
    df = measurements_df.copy()
    if df.empty:
        df["quality_class"] = []
        df["classification_reason"] = []
        return df

    unbroken_candidates = df[
        (df["separation_status"].isin(["isolated_single", "separated_from_cluster"])) &
        (~df["is_boundary"]) &
        (df["solidity"] >= 0.90) &
        (df["aspect_ratio"] >= 1.8)
    ]

    if not unbroken_candidates.empty:
        median_whole_length = float(unbroken_candidates["length_px"].median())
    else:
        non_boundary = df[~df["is_boundary"]] if "is_boundary" in df.columns else df
        median_whole_length = float(non_boundary["length_px"].median()) if not non_boundary.empty else float(df["length_px"].median())

    quality_classes = []
    reasons = []

    for _, row in df.iterrows():
        status = row.get("separation_status", "")
        length = row["length_px"]
        ar = row["aspect_ratio"]

        if status == "under_segmented_cluster":
            quality_classes.append("UNRESOLVED_CLUSTER")
            reasons.append("Multi-grain cluster unresolved by watershed")
        elif row.get("is_boundary", False):
            quality_classes.append("BOUNDARY_TRUNCATED")
            reasons.append("Grain touches image boundary (truncated measurements)")
        elif length < (broken_length_ratio_thresh * median_whole_length) or ar < broken_aspect_ratio_thresh:
            quality_classes.append(RealDataTaxonomy.BROKEN_RICE.value)
            reasons.append(f"Length {length:.1f}px < 75% of median unbroken ({median_whole_length:.1f}px)")
        else:
            quality_classes.append(RealDataTaxonomy.WHOLE_RICE.value)
            reasons.append("Length >= 75% median unbroken length and aspect ratio >= 1.6")

    df["quality_class"] = quality_classes
    df["classification_reason"] = reasons
    return df
