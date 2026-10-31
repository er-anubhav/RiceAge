# Image segmentation: thresholding, morphological cleanup, and deep learning YOLO segmentation.

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2
from skimage import filters
from scipy import ndimage as ndi

_CACHED_YOLO = None
_CACHED_YOLO_PATH = None


def segment_yolo(
    image_bgr: np.ndarray,
    weights_path: str = "runs/rice_seg_yolo11s_1280/weights/best.pt",
    conf_threshold: float = 0.25,
    imgsz: int = 1280,
    max_det: int = 2000,
    device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, Any]], Dict[str, Any]]:
    # Perform instance segmentation using trained YOLO11s-seg model.
    global _CACHED_YOLO, _CACHED_YOLO_PATH
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("ultralytics package is required for YOLO segmentation engine.") from e

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO segmentation weights not found: {weights_path}")

    if _CACHED_YOLO is None or _CACHED_YOLO_PATH != weights_path:
        _CACHED_YOLO = YOLO(weights_path)
        _CACHED_YOLO_PATH = weights_path

    if device is None:
        if weights_path.endswith(".onnx"):
            device = "cpu"
        else:
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"

    results = _CACHED_YOLO.predict(
        source=image_bgr,
        imgsz=imgsz,
        conf=conf_threshold,
        max_det=max_det,
        device=device,
        verbose=False
    )
    res = results[0]

    img_h, img_w = image_bgr.shape[:2]
    raw_instance_mask = np.zeros((img_h, img_w), dtype=np.int32)
    clean_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    temp_diagnostics: Dict[int, Dict[str, Any]] = {}

    if res.masks is not None and len(res.masks.xy) > 0:
        masks_xy = res.masks.xy
        confidences = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.ones(len(masks_xy))

        draw_id = 1
        for i, poly in enumerate(masks_xy):
            if len(poly) < 3:
                continue
            poly_int = poly.astype(np.int32)
            poly_reshaped = poly_int.reshape((-1, 1, 2))

            area = cv2.contourArea(poly_reshaped)
            if area < 20:
                continue

            hull = cv2.convexHull(poly_reshaped)
            hull_area = cv2.contourArea(hull)
            solidity = float(area / hull_area) if hull_area > 0 else 1.0

            cv2.fillPoly(raw_instance_mask, [poly_int], draw_id)
            cv2.fillPoly(clean_mask, [poly_int], 255)

            temp_diagnostics[draw_id] = {
                "status": "isolated_single",
                "parent_component_id": draw_id,
                "solidity": round(solidity, 3),
                "component_area": int(area),
                "was_split": False,
                "yolo_conf": round(float(confidences[i]), 3) if i < len(confidences) else 1.0
            }
            draw_id += 1

    # Contiguous reindexing 1..N based on surviving pixels
    surviving_ids = np.unique(raw_instance_mask)
    surviving_ids = surviving_ids[surviving_ids > 0]

    max_val = int(raw_instance_mask.max()) if len(surviving_ids) > 0 else 0
    lut = np.zeros(max_val + 1, dtype=np.int32)
    grain_diagnostics: Dict[int, Dict[str, Any]] = {}

    for new_id, old_id in enumerate(surviving_ids, start=1):
        lut[old_id] = new_id
        diag = temp_diagnostics[old_id]
        diag["parent_component_id"] = new_id
        grain_diagnostics[new_id] = diag

    instance_mask = lut[raw_instance_mask]
    total_grains = len(grain_diagnostics)

    separation_stats = {
        "engine": "yolo11s-seg",
        "raw_connected_components": total_grains,
        "total_detected_grains": total_grains,
        "isolated_single_grains": total_grains,
        "clusters_detected": 0,
        "clusters_successfully_split": 0,
        "clusters_unresolved": 0,
        "grains_derived_from_clusters": 0,
        "fragments_count": 0,
    }

    return instance_mask, clean_mask, grain_diagnostics, separation_stats


# Segment image using Otsu's optimal global thresholding.
def segment_otsu(gray_image: np.ndarray) -> Tuple[np.ndarray, float]:
    thresh_val, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, float(thresh_val)


# Segment image using Adaptive Gaussian thresholding.
def segment_adaptive(
    gray_image: np.ndarray,
    block_size: int = 51,
    constant_c: int = 8
) -> Tuple[np.ndarray, float]:
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant_c
    )
    return binary, float(constant_c)


# Segment image using Triangle thresholding.
def segment_triangle(gray_image: np.ndarray) -> Tuple[np.ndarray, float]:
    thresh_val, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return binary, float(thresh_val)


# Segment image using Yen's entropy-based thresholding.
def segment_yen(gray_image: np.ndarray) -> Tuple[np.ndarray, float]:
    val = filters.threshold_yen(gray_image)
    binary = (gray_image > val).astype(np.uint8) * 255
    return binary, float(val)


# Segment image using Li's minimum cross entropy thresholding.
def segment_li(gray_image: np.ndarray) -> Tuple[np.ndarray, float]:
    val = filters.threshold_li(gray_image)
    binary = (gray_image > val).astype(np.uint8) * 255
    return binary, float(val)


def _compute_contour_gradient_energy(mask: np.ndarray, grad_mag: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    edge_mask = np.zeros_like(mask)
    cv2.drawContours(edge_mask, contours, -1, 255, 1)
    edge_pixels = edge_mask > 0
    return float(np.mean(grad_mag[edge_pixels])) if np.any(edge_pixels) else 0.0


# Run and quantitatively compare multiple thresholding approaches.
def compare_segmentation_methods(gray_image: np.ndarray) -> Dict[str, Dict[str, Any]]:
    methods = {
        "otsu": segment_otsu,
        "triangle": segment_triangle,
        "yen": segment_yen,
        "li": segment_li,
        "adaptive_gaussian": segment_adaptive,
    }

    grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    results = {}
    for name, func in methods.items():
        mask, val = func(gray_image)
        fg_ratio = float(np.mean(mask > 0))
        num_cc, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        results[name] = {
            "threshold_value": val,
            "foreground_ratio": fg_ratio,
            "raw_connected_components": num_cc - 1,
            "edge_gradient_magnitude": _compute_contour_gradient_energy(mask, grad_mag),
            "mask": mask
        }

    return results


# Fill internal cavities and hollow specular holes inside rice grains.
def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    filled = ndi.binary_fill_holes(mask > 0)
    return filled.astype(np.uint8) * 255


# Filter out noise, dust, and minute scanner artifacts smaller than min_area_px.
def remove_small_objects(mask: np.ndarray, min_area_px: int = 40) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area_px:
            clean_mask[labels == label] = 255
    return clean_mask


# Perform hole filling, morphological opening, and small noise removal.
def cleanup_mask(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
    min_area_px: int = 40,
    fill_holes: bool = True
) -> np.ndarray:
    processed = fill_binary_holes(binary_mask) if fill_holes else binary_mask.copy()

    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel, iterations=1)

    if min_area_px > 0:
        processed = remove_small_objects(processed, min_area_px=min_area_px)

    return processed
