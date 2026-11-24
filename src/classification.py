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
    SMALL_BROKEN_RICE = "SMALL_BROKEN_RICE"


class FutureDefectTaxonomy(str, Enum):
    # Target defect taxonomy for supervised models with labeled data.
    WHOLE_RICE = "WHOLE_RICE"
    BROKEN_RICE = "BROKEN_RICE"
    SMALL_BROKEN_RICE = "SMALL_BROKEN_RICE"
    CHALKY_RICE = "CHALKY_RICE"
    DISCOLORED_RICE = "DISCOLORED_RICE"
    DAMAGED_RICE = "DAMAGED_RICE"
    PEST_DAMAGED = "PEST_DAMAGED"
    UNRIPE_RICE = "UNRIPE_RICE"
    FOREIGN_MATERIAL = "FOREIGN_MATERIAL"
    OTHER_GRAIN = "OTHER_GRAIN"


def estimate_grain_mass(
    length_val: float,
    width_val: float,
    pixel_scale_mm: Optional[float] = None,
    bulk_density_g_cm3: float = 1.40
) -> float:
    """
    Estimates single grain mass in grams using triaxial ellipsoid volume:
    V = (4/3) * pi * (L/2) * (W/2) * (T/2) = (pi / 6) * L * W * (0.7 * W)
    Mass (g) = Volume (mm^3) * 10^-3 (cm^3/mm^3) * Density (g/cm^3)
    """
    if pixel_scale_mm is not None and pixel_scale_mm > 0:
        l_mm = length_val * pixel_scale_mm
        w_mm = width_val * pixel_scale_mm
    else:
        # Default nominal 300 DPI conversion (0.08467 mm/px) if calibration is unverified
        l_mm = length_val * 0.08466667
        w_mm = width_val * 0.08466667

    t_mm = 0.70 * w_mm
    vol_mm3 = (np.pi / 6.0) * l_mm * w_mm * t_mm
    mass_g = vol_mm3 * 1e-3 * bulk_density_g_cm3
    return max(float(mass_g), 1e-6)


AI_CLASSES = [
    "broken",
    "chalky",
    "damaged",
    "discoloured",
    "foreign_matter",
    "whole"
]

_CACHED_RESNET = None
_CACHED_DEVICE = None
_CACHED_WEIGHTS_PATH = None
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

        # High-performance threading & graph optimizations for ARM Cortex-A76 4 cores
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        _CACHED_ONNX_SESS = sess
        _CACHED_ONNX_PATH = onnx_path
        return sess
    except Exception as e:
        print(f"Warning: Failed to load ONNX session from {onnx_path}: {e}")
        return None


_CACHED_MODELS = {}

def _get_classifier_model(weights_path: str, device_str: Optional[str] = None):
    global _CACHED_MODELS
    if not _TORCH_AVAILABLE:
        return None, None
    device = torch.device(device_str if device_str else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    cache_key = (os.path.abspath(weights_path), str(device))
    if cache_key in _CACHED_MODELS:
        return _CACHED_MODELS[cache_key], device

    if not os.path.exists(weights_path):
        return None, device

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    sd = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    fc_out = sd["fc.1.weight"].shape[0] if "fc.1.weight" in sd else len(AI_CLASSES)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, fc_out)
    )
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    _CACHED_MODELS[cache_key] = model
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
    return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def classify_grains_ai(
    measurements_df: pd.DataFrame,
    original_bgr: np.ndarray,
    weights_path: str = "runs/stage2_resnet50/best_resnet50.pth",
    broken_length_ratio_thresh: float = 0.75,
    small_broken_length_ratio_thresh: float = 0.25,
    broken_aspect_ratio_thresh: float = 1.6,
    pixel_scale_mm: Optional[float] = None
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
    small_broken_cutoff = small_broken_length_ratio_thresh * median_whole_len

    # Extract crops and optical features
    crop_tensors = []
    opt_features = []
    crops_to_infer_indices = []

    for idx_row, (_, row) in enumerate(df.iterrows()):
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

        # Fast optical forensic features for pre-screening & accuracy verification
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
        b_v = core_bgr[:, 0].astype(np.float32)
        rg = float(np.mean(r_v / np.maximum(g_v, 1.0)))
        rb_diff = float(np.mean(r_v - b_v))

        v_vals = core_hsv[:, 2].astype(np.float32)
        s_vals = core_hsv[:, 1].astype(np.float32)
        std_v = float(np.std(v_vals))
        med_v = float(np.median(v_vals))
        sat_mean = float(np.mean(s_vals))

        pinhole_ratio = float(np.sum(v_vals < 0.60 * med_v)) / max(float(len(v_vals)), 1.0)
        chalky_ratio = float(np.sum((v_vals >= 170) & (s_vals <= 35))) / max(float(len(v_vals)), 1.0)

        opt_features.append({
            "rg_ratio": rg,
            "rb_diff": rb_diff,
            "sat_mean": sat_mean,
            "core_std_v": std_v,
            "pinhole_ratio": pinhole_ratio,
            "chalky_ratio": chalky_ratio
        })

        # ACCELERATION FILTER (Selective Inference):
        # Only normalize and queue tensor if grain has potential defect or morphology anomaly.
        # Boundary grains and unresolved clusters are already non-AI classified.
        status = row.get("separation_status", "")
        is_boundary = row.get("is_boundary", False)
        length = row["length_px"]
        ar = row["aspect_ratio"]
        is_broken = (length < broken_cutoff) or (ar < 1.15) or (length < small_broken_cutoff)

        # Selective Deep Defect Inference:
        # Physical brokenness is already 100% measured by calibrated contour length (Codex CXS 198).
        # We only queue crops for ResNet-50 when optical signatures indicate possible defect:
        needs_ai = (not is_boundary) and (status != "under_segmented_cluster") and (
            chalky_ratio > 0.28 or rg > 1.030 or rb_diff > 12.0 or pinhole_ratio > 0.025 or std_v > 16.0
        )

        if needs_ai:
            sq_rgb = cv2.cvtColor(sq_crop, cv2.COLOR_BGR2RGB)
            if is_onnx:
                norm_rgb = sq_rgb.astype(np.float32) / 255.0
                norm_rgb = (norm_rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
                crop_tensors.append(np.transpose(norm_rgb, (2, 0, 1)))
            else:
                crop_tensors.append(transform(sq_rgb))
            crops_to_infer_indices.append(idx_row)

    # Determine exact output dimension of active model
    if onnx_sess:
        try:
            num_classes = onnx_sess.get_outputs()[0].shape[1]
            if isinstance(num_classes, str): num_classes = 5
        except Exception:
            num_classes = 5
    else:
        num_classes = 6

    default_sound_prob = np.zeros(num_classes, dtype=np.float32)
    if num_classes >= 6:
        default_sound_prob[-1] = 0.98  # whole class is at index -1

    probs_list = [default_sound_prob.copy() for _ in range(len(df))]

    if len(crop_tensors) > 0:
        inferred_probs = []
        if is_onnx:
            batch_size = 128
            for b_start in range(0, len(crop_tensors), batch_size):
                batch = np.stack(crop_tensors[b_start:b_start + batch_size], axis=0)
                logits = onnx_sess.run(None, {"input": batch})[0]
                exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
                inferred_probs.extend(probs)
        else:
            batch_size = 64
            with torch.no_grad():
                for b_start in range(0, len(crop_tensors), batch_size):
                    batch = torch.stack(crop_tensors[b_start:b_start + batch_size]).to(device)
                    with torch.amp.autocast("cuda") if "cuda" in str(device) else torch.no_grad():
                        logits = model(batch)
                    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                    inferred_probs.extend(probs)

        for orig_idx, prob_v in zip(crops_to_infer_indices, inferred_probs):
            probs_list[orig_idx] = prob_v

    tray_median_rg = float(np.median([m["rg_ratio"] for m in opt_features])) if opt_features else 1.0
    tray_median_std = float(np.median([m["core_std_v"] for m in opt_features])) if opt_features else 15.0

    prob_dim = len(probs_list[0]) if len(probs_list) > 0 else 5
    if prob_dim == 7:
        active_classes = ["broken", "chalky", "damaged", "dehusked", "discoloured", "foreign_matter", "whole"]
    elif prob_dim == 6:
        active_classes = ["broken", "chalky", "damaged", "discoloured", "foreign_matter", "whole"]
    else:
        # 5-class defect model: defects only, whole rice inferred by low defect sum
        active_classes = ["broken", "chalky", "damaged", "discoloured", "foreign_matter"]
    IDX = {name: i for i, name in enumerate(active_classes)}

    quality_classes = []
    defects_list = []
    is_sound_list = []
    is_broken_list = []
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
            is_broken_list.append(False)
            confidences.append(1.0)
            reasons.append("Multi-grain cluster unresolved by watershed")
            continue
        elif row.get("is_boundary", False):
            quality_classes.append("BOUNDARY_TRUNCATED")
            defects_list.append(["boundary"])
            is_sound_list.append(False)
            is_broken_list.append(False)
            confidences.append(1.0)
            reasons.append("Touches scanner boundary margin")
            continue

        rg_ratio = opt["rg_ratio"]
        core_std_v = opt["core_std_v"]
        pinhole_ratio = opt["pinhole_ratio"]
        chalky_ratio = opt["chalky_ratio"]

        p_broken = float(prob_vec[IDX["broken"]]) if "broken" in IDX and IDX["broken"] < len(prob_vec) else 0.0
        p_chalky = float(prob_vec[IDX["chalky"]]) if "chalky" in IDX and IDX["chalky"] < len(prob_vec) else 0.0
        p_damaged = float(prob_vec[IDX["damaged"]]) if "damaged" in IDX and IDX["damaged"] < len(prob_vec) else 0.0
        p_discoloured = float(prob_vec[IDX["discoloured"]]) if "discoloured" in IDX and IDX["discoloured"] < len(prob_vec) else 0.0
        p_foreign = float(prob_vec[IDX["foreign_matter"]]) if "foreign_matter" in IDX and IDX["foreign_matter"] < len(prob_vec) else 0.0
        if "whole" in IDX and IDX["whole"] < len(prob_vec):
            p_whole = float(prob_vec[IDX["whole"]])
        else:
            p_whole = max(0.0, 1.0 - (p_broken + p_chalky + p_damaged + p_discoloured + p_foreign))

        is_small_broken = (length < small_broken_cutoff)
        # Broken strictly governed by standard length cutoff (< 75% of median unbroken length)
        # Aspect ratio alone does not make a grain broken unless extremely fragmented (ar < 1.15)
        is_broken = (length < broken_cutoff) or (ar < 1.15) or is_small_broken

        has_oxidation = (rg_ratio > 1.012 * tray_median_rg) and (rg_ratio >= 1.042 or p_discoloured > 0.40)
        rb_diff = opt.get("rb_diff", 0.0)
        sat_mean = opt.get("sat_mean", 0.0)

        has_mottled = (core_std_v > 1.25 * tray_median_std)
        has_pinhole = (pinhole_ratio > 0.04)

        # Multi-attribute defect flags (Coupled with forensic optical validation)
        is_low_res = (median_whole_len < 65.0)

        grain_defects = []
        if not is_low_res:
            if (p_damaged > 0.40 and (has_oxidation or has_mottled or has_pinhole)) or (p_damaged > 0.65):
                grain_defects.append("damaged")
            if (p_chalky > 0.40 and chalky_ratio > 0.50) or (chalky_ratio >= 0.78):
                grain_defects.append("chalky")
            if (p_discoloured > 0.35 and rg_ratio >= 1.042) or (p_discoloured > 0.70 and rg_ratio >= 1.035) or (has_oxidation and p_discoloured > p_whole):
                grain_defects.append("discoloured")
        else:
            # Low-resolution images (downsampled thumbnails): require high confidence + physical signatures
            if p_damaged > 0.70 and (has_oxidation or has_pinhole):
                grain_defects.append("damaged")
            if (p_chalky > 0.60 and chalky_ratio > 0.50) or (chalky_ratio >= 0.78):
                grain_defects.append("chalky")
            if p_discoloured > 0.70 and has_oxidation:
                grain_defects.append("discoloured")

        # Assign primary Quality Class (FutureDefectTaxonomy)
        if p_foreign >= 0.85 or (rb_diff > 25.0 and p_foreign > 0.30):
            q_class = FutureDefectTaxonomy.FOREIGN_MATERIAL.value
            conf = float(max(p_foreign, 0.95))
            reason = f"Foreign non-grain material / paddy husk detected (R-B={rb_diff:.1f})"
        elif "damaged" in grain_defects or p_damaged > 0.60:
            q_class = FutureDefectTaxonomy.DAMAGED_RICE.value
            conf = float(p_damaged)
            reason = f"Damaged: oxidation R/G={rg_ratio:.3f}, texture std={core_std_v:.1f}"
        elif "discoloured" in grain_defects:
            q_class = FutureDefectTaxonomy.DISCOLORED_RICE.value
            conf = float(max(p_discoloured, 0.88))
            reason = f"Discoloured amber/yellow tint R/G={rg_ratio:.3f}"
        elif "chalky" in grain_defects:
            q_class = FutureDefectTaxonomy.CHALKY_RICE.value
            conf = float(max(p_chalky, min(1.0, chalky_ratio)))
            reason = f"Chalky endosperm core opacity ratio={chalky_ratio:.2f}"
        elif is_small_broken:
            q_class = FutureDefectTaxonomy.SMALL_BROKEN_RICE.value
            conf = float(max(p_broken, 0.90))
            reason = f"Small Broken (<25%): Length {length:.1f}px < 25% of median ({median_whole_len:.1f}px)"
        elif is_broken:
            q_class = FutureDefectTaxonomy.BROKEN_RICE.value
            conf = float(max(p_broken, 0.85))
            reason = f"Broken (25-75%): Length {length:.1f}px < 75% of median ({median_whole_len:.1f}px)"
        else:
            q_class = FutureDefectTaxonomy.WHOLE_RICE.value
            conf = float(p_whole)
            reason = "Sound Whole Rice (Unbroken & Defect-Free)"

        quality_classes.append(q_class)
        defects_list.append(grain_defects)
        is_sound_list.append(len(grain_defects) == 0 and not is_broken)
        is_broken_list.append(bool(is_broken or is_small_broken))
        confidences.append(round(conf, 3))
        reasons.append(reason)

    df["quality_class"] = quality_classes
    df["defects"] = defects_list
    df["is_sound"] = is_sound_list
    df["is_broken"] = is_broken_list
    df["confidence"] = confidences
    df["classification_reason"] = reasons
    df["estimated_mass_g"] = [
        estimate_grain_mass(row["length_px"], row.get("width_px", row.get("length_px", 1.0) / max(row.get("aspect_ratio", 1.0), 1.0)), pixel_scale_mm)
        for _, row in df.iterrows()
    ]

    return df


# Classify grains into WHOLE_RICE, BROKEN_RICE, or UNRESOLVED_CLUSTER based on morphology (Classic Rule-Based).
def classify_grains_rule_based(
    measurements_df: pd.DataFrame,
    broken_length_ratio_thresh: float = 0.75,
    small_broken_length_ratio_thresh: float = 0.25,
    broken_aspect_ratio_thresh: float = 1.6,
    pixel_scale_mm: Optional[float] = None,
    reference_whole_length_mm: Optional[float] = None
) -> pd.DataFrame:
    df = measurements_df.copy()
    if df.empty:
        df["quality_class"] = []
        df["classification_reason"] = []
        df["estimated_mass_g"] = []
        return df

    unbroken_candidates = df[
        (df["separation_status"].isin(["isolated_single", "separated_from_cluster"])) &
        (~df["is_boundary"]) &
        (df["solidity"] >= 0.90) &
        (df["aspect_ratio"] >= 1.8)
    ]

    if reference_whole_length_mm is not None and pixel_scale_mm is not None and pixel_scale_mm > 0:
        median_whole_length = float(reference_whole_length_mm / pixel_scale_mm)
    elif not unbroken_candidates.empty:
        median_whole_length = float(unbroken_candidates["length_px"].median())
    else:
        non_boundary = df[~df["is_boundary"]] if "is_boundary" in df.columns else df
        median_whole_length = float(non_boundary["length_px"].median()) if not non_boundary.empty else float(df["length_px"].median())

    quality_classes = []
    reasons = []

    small_broken_cutoff = small_broken_length_ratio_thresh * median_whole_length
    broken_cutoff = broken_length_ratio_thresh * median_whole_length

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
        elif length < small_broken_cutoff:
            quality_classes.append(RealDataTaxonomy.SMALL_BROKEN_RICE.value)
            reasons.append(f"Small Broken: Length {length:.1f}px < 25% of median unbroken ({median_whole_length:.1f}px)")
        elif length < broken_cutoff or ar < broken_aspect_ratio_thresh:
            quality_classes.append(RealDataTaxonomy.BROKEN_RICE.value)
            reasons.append(f"Broken: Length {length:.1f}px < 75% of median unbroken ({median_whole_length:.1f}px)")
        elif (row.get("mean_r", 0) - row.get("mean_b", 0) >= 6.0) or (float(row.get("mean_r", 0)) / max(float(row.get("mean_g", 1.0)), 1.0) >= 1.015):
            quality_classes.append(FutureDefectTaxonomy.DISCOLORED_RICE.value)
            reasons.append(f"Discoloured amber/yellow tint: R-B={row.get('mean_r', 0) - row.get('mean_b', 0):.1f}, R/G={float(row.get('mean_r', 0))/max(float(row.get('mean_g', 1.0)), 1.0):.3f}")
        else:
            quality_classes.append(RealDataTaxonomy.WHOLE_RICE.value)
            reasons.append("Length >= 75% median unbroken length and aspect ratio >= 1.6")

    df["quality_class"] = quality_classes
    df["classification_reason"] = reasons
    df["estimated_mass_g"] = [
        estimate_grain_mass(row["length_px"], row.get("width_px", row.get("length_px", 1.0) / max(row.get("aspect_ratio", 1.0), 1.0)), pixel_scale_mm)
        for _, row in df.iterrows()
    ]
    return df


def compute_quality_metrics(
    df: pd.DataFrame,
    pixel_scale_mm: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes comprehensive Count Basis and Mass Basis (% w/w) quality metrics
    under Codex Alimentarius Standard 198-1995 and APSCSCL procurement specifications.
    """
    if df.empty:
        return {
            "total_grains": 0,
            "total_mass_g": 0.0,
            "count_basis": {},
            "mass_basis": {},
            "commercial_grade": "Unclassified"
        }

    # Ensure mass column exists
    if "estimated_mass_g" not in df.columns:
        df["estimated_mass_g"] = [
            estimate_grain_mass(row["length_px"], row.get("width_px", row.get("length_px", 1.0) / max(row.get("aspect_ratio", 1.0), 1.0)), pixel_scale_mm)
            for _, row in df.iterrows()
        ]

    total_count = len(df)
    total_mass = float(df["estimated_mass_g"].sum())

    is_foreign = df["quality_class"] == FutureDefectTaxonomy.FOREIGN_MATERIAL.value
    df_rice = df[~is_foreign]
    rice_count = len(df_rice)
    rice_mass = float(df_rice["estimated_mass_g"].sum()) if rice_count > 0 else 1e-6

    whole_mask = df_rice["quality_class"] == FutureDefectTaxonomy.WHOLE_RICE.value
    broken_mask = df_rice["quality_class"] == FutureDefectTaxonomy.BROKEN_RICE.value
    small_broken_mask = df_rice["quality_class"] == FutureDefectTaxonomy.SMALL_BROKEN_RICE.value

    if "defects" in df_rice.columns:
        chalky_mask = (df_rice["quality_class"] == FutureDefectTaxonomy.CHALKY_RICE.value) | df_rice["defects"].apply(lambda d: "chalky" in d if isinstance(d, (list, set, tuple)) else False)
        damaged_mask = (df_rice["quality_class"] == FutureDefectTaxonomy.DAMAGED_RICE.value) | df_rice["defects"].apply(lambda d: "damaged" in d if isinstance(d, (list, set, tuple)) else False)
        discoloured_mask = (df_rice["quality_class"] == FutureDefectTaxonomy.DISCOLORED_RICE.value) | df_rice["defects"].apply(lambda d: "discoloured" in d if isinstance(d, (list, set, tuple)) else False)
    else:
        chalky_mask = df_rice["quality_class"] == FutureDefectTaxonomy.CHALKY_RICE.value
        damaged_mask = df_rice["quality_class"] == FutureDefectTaxonomy.DAMAGED_RICE.value
        discoloured_mask = df_rice["quality_class"] == FutureDefectTaxonomy.DISCOLORED_RICE.value

    # Counts
    c_whole = int(whole_mask.sum())
    c_broken = int(broken_mask.sum())
    c_small_broken = int(small_broken_mask.sum())
    c_chalky = int(chalky_mask.sum())
    c_damaged = int(damaged_mask.sum())
    c_discoloured = int(discoloured_mask.sum())
    c_foreign = int(is_foreign.sum())

    # Masses (g)
    m_whole = float(df_rice.loc[whole_mask, "estimated_mass_g"].sum())
    m_broken = float(df_rice.loc[broken_mask, "estimated_mass_g"].sum())
    m_small_broken = float(df_rice.loc[small_broken_mask, "estimated_mass_g"].sum())
    m_chalky = float(df_rice.loc[chalky_mask, "estimated_mass_g"].sum())
    m_damaged = float(df_rice.loc[damaged_mask, "estimated_mass_g"].sum())
    m_discoloured = float(df_rice.loc[discoloured_mask, "estimated_mass_g"].sum())
    m_foreign = float(df.loc[is_foreign, "estimated_mass_g"].sum())

    count_basis = {
        "whole_pct": round(100.0 * c_whole / max(rice_count, 1), 2),
        "broken_pct": round(100.0 * c_broken / max(rice_count, 1), 2),
        "small_broken_pct": round(100.0 * c_small_broken / max(rice_count, 1), 2),
        "total_broken_pct": round(100.0 * (c_broken + c_small_broken) / max(rice_count, 1), 2),
        "chalky_pct": round(100.0 * c_chalky / max(rice_count, 1), 2),
        "damaged_pct": round(100.0 * c_damaged / max(rice_count, 1), 2),
        "discoloured_pct": round(100.0 * c_discoloured / max(rice_count, 1), 2),
        "foreign_matter_pct": round(100.0 * c_foreign / max(total_count, 1), 2),
        "whole_count": c_whole,
        "broken_count": c_broken,
        "small_broken_count": c_small_broken,
        "total_broken_count": c_broken + c_small_broken,
        "chalky_count": c_chalky,
        "damaged_count": c_damaged,
        "discoloured_count": c_discoloured,
        "foreign_matter_count": c_foreign,
        "total_rice_grains": rice_count,
        "total_objects": total_count,
    }

    mass_basis = {
        "whole_pct": round(100.0 * m_whole / max(rice_mass, 1e-6), 2),
        "broken_pct": round(100.0 * m_broken / max(rice_mass, 1e-6), 2),
        "small_broken_pct": round(100.0 * m_small_broken / max(rice_mass, 1e-6), 2),
        "total_broken_pct": round(100.0 * (m_broken + m_small_broken) / max(rice_mass, 1e-6), 2),
        "chalky_pct": round(100.0 * m_chalky / max(rice_mass, 1e-6), 2),
        "damaged_pct": round(100.0 * m_damaged / max(rice_mass, 1e-6), 2),
        "discoloured_pct": round(100.0 * m_discoloured / max(rice_mass, 1e-6), 2),
        "foreign_matter_pct": round(100.0 * m_foreign / max(total_mass, 1e-6), 2),
        "whole_mass_g": round(m_whole, 4),
        "broken_mass_g": round(m_broken, 4),
        "small_broken_mass_g": round(m_small_broken, 4),
        "total_broken_mass_g": round(m_broken + m_small_broken, 4),
        "chalky_mass_g": round(m_chalky, 4),
        "damaged_mass_g": round(m_damaged, 4),
        "discoloured_mass_g": round(m_discoloured, 4),
        "foreign_matter_mass_g": round(m_foreign, 4),
        "rice_mass_g": round(rice_mass, 4),
        "total_sample_mass_g": round(total_mass, 4),
    }

    # Explicitly mark optical mass as NOT_PHYSICALLY_MEASURED and suspend statutory mass grading
    mass_basis["mass_measurement_status"] = "NOT_PHYSICALLY_MEASURED"
    mass_basis["is_physically_measured"] = False
    mass_basis["statutory_grading_suspended"] = True
    mass_basis["statutory_mass_disclaimer"] = "Optical mass is an uncalibrated geometric proxy. Prohibited from statutory procurement grading without calibrated physical load cell."

    # 3-tier pre-inspection optical screening evaluation (Advisory only; non-certified)
    # Uses count-basis optical measurements; preserves count metrics separately
    cb = count_basis
    optical_meets_thresholds = (
        cb["whole_pct"] >= 75.0 and
        cb["broken_pct"] <= 15.0 and
        cb["small_broken_pct"] <= 1.0 and
        cb["chalky_pct"] <= 3.0 and
        cb["damaged_pct"] <= 2.0 and
        cb["discoloured_pct"] <= 1.0 and
        cb["foreign_matter_pct"] <= 0.5
    )

    if optical_meets_thresholds:
        screening_verdict = "PASS"
        grade = "Grade 1 (Pre-Inspection Optical Criteria Met - Lab Confirmation Required)"
    else:
        screening_verdict = "FLAG"
        grade = "FLAG (Optical Defect Limit Exceeded - Lab Confirmation Required)"

    return {
        "count_basis": count_basis,
        "mass_basis": mass_basis,
        "commercial_grade": grade,
        "screening_verdict": screening_verdict,
        "decision_tier": screening_verdict,
        "statutory_compliance_status": "REQUIRES_LAB_CONFIRMATION",
        "mass_measurement_status": "NOT_PHYSICALLY_MEASURED",
        "statutory_mass_usable": False,
        "statutory_grading_suspended": True,
        "total_grains": total_count,
        "total_mass_g": round(total_mass, 4)
    }
