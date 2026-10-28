# Image segmentation: thresholding and morphological cleanup.

from typing import Tuple, Dict, Any
import numpy as np
import cv2
from skimage import filters
from scipy import ndimage as ndi


# Segment image using Otsu's optimal global thresholding with trimodal collapse guardrail.
def segment_otsu(gray_image: np.ndarray, max_fg_ratio: float = 0.20) -> Tuple[np.ndarray, float]:
    thresh_val, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_ratio = float(np.mean(binary > 0))
    if fg_ratio > max_fg_ratio:
        valid_pixels = gray_image[gray_image > 20]
        if len(valid_pixels) > 1000:
            alt_thresh, _ = cv2.threshold(valid_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bg_pix = valid_pixels[valid_pixels < 130]
            noise_floor = float(np.percentile(bg_pix, 99.9)) if len(bg_pix) > 100 else 100.0
            alt_thresh = max(alt_thresh, noise_floor, 100.0)
            alt_binary = (gray_image > alt_thresh).astype(np.uint8) * 255
            if float(np.mean(alt_binary > 0)) <= max_fg_ratio:
                return alt_binary, float(alt_thresh)
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
