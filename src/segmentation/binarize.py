"""Foreground/background segmentation and threshold comparison module."""

from typing import Tuple, Dict, Any
import numpy as np
import cv2
from skimage import filters


def segment_otsu(gray_image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Segment image using Otsu's optimal global thresholding.
    
    Returns:
        binary_mask: uint8 array (0 = background, 255 = foreground)
        thresh_value: float threshold chosen by Otsu's method.
    """
    thresh_val, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, float(thresh_val)


def segment_adaptive(
    gray_image: np.ndarray,
    block_size: int = 51,
    constant_c: int = 8
) -> Tuple[np.ndarray, float]:
    """Segment image using Adaptive Gaussian thresholding.
    
    Returns:
        binary_mask: uint8 array (0 = background, 255 = foreground)
        thresh_value: mean effective threshold across image.
    """
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


def segment_triangle(gray_image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Segment image using Triangle thresholding (well-suited for unimodal with shoulder).
    
    Returns:
        binary_mask: uint8 array (0 = background, 255 = foreground)
        thresh_val: float threshold chosen by Triangle method.
    """
    thresh_val, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return binary, float(thresh_val)


def segment_yen(gray_image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Segment image using Yen's entropy-based thresholding."""
    val = filters.threshold_yen(gray_image)
    binary = (gray_image > val).astype(np.uint8) * 255
    return binary, float(val)


def segment_li(gray_image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Segment image using Li's minimum cross entropy thresholding."""
    val = filters.threshold_li(gray_image)
    binary = (gray_image > val).astype(np.uint8) * 255
    return binary, float(val)


def compare_segmentation_methods(gray_image: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Run and quantitatively compare multiple thresholding approaches.
    
    Evaluates:
    - Threshold value
    - Foreground pixel ratio
    - Total connected components (raw)
    - Average contour edge gradient magnitude
    """
    methods = {
        "otsu": segment_otsu,
        "triangle": segment_triangle,
        "yen": segment_yen,
        "li": segment_li,
        "adaptive_gaussian": segment_adaptive,
    }
    
    # Precompute Sobel gradient magnitude for edge sharpness assessment
    grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    results = {}
    for name, func in methods.items():
        mask, val = func(gray_image)
        fg_ratio = float(np.mean(mask > 0))
        num_cc, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Edge gradient energy: compute gradient magnitude along mask contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge_mask = np.zeros_like(mask)
        cv2.drawContours(edge_mask, contours, -1, 255, 1)
        edge_pixels = edge_mask > 0
        if np.any(edge_pixels):
            avg_edge_grad = float(np.mean(grad_mag[edge_pixels]))
        else:
            avg_edge_grad = 0.0

        results[name] = {
            "threshold_value": val,
            "foreground_ratio": fg_ratio,
            "raw_connected_components": num_cc - 1,
            "edge_gradient_magnitude": avg_edge_grad,
            "mask": mask
        }

    return results
