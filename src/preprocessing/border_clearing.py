"""Scanner bezel detection, margin clearing, and boundary grain identification."""

from typing import Tuple
import numpy as np
import cv2


def detect_scanner_margins(
    gray_image: np.ndarray,
    saturation_thresh: int = 240,
    max_search_ratio: float = 0.05
) -> Tuple[int, int, int, int]:
    """Detect saturated scanner glass bezel / document cover lines along image borders.
    
    Returns:
        (top_margin, bottom_margin, left_margin, right_margin) in pixels.
    """
    h, w = gray_image.shape[:2]
    max_h_search = int(h * max_search_ratio)
    max_w_search = int(w * max_search_ratio)

    top_margin = 0
    bottom_margin = 0
    left_margin = 0
    right_margin = 0

    # Top margin (check from row 0 downwards)
    for r in range(max_h_search):
        if gray_image[r, :].mean() >= saturation_thresh:
            top_margin = r + 1
        else:
            break

    # Bottom margin (check from row h-1 upwards)
    for r in range(h - 1, h - 1 - max_h_search, -1):
        if gray_image[r, :].mean() >= saturation_thresh:
            bottom_margin = h - r
        else:
            break

    # Left margin (check from col 0 rightwards)
    for c in range(max_w_search):
        if gray_image[:, c].mean() >= saturation_thresh:
            left_margin = c + 1
        else:
            break

    # Right margin (check from col w-1 leftwards)
    for c in range(w - 1, w - 1 - max_w_search, -1):
        if gray_image[:, c].mean() >= saturation_thresh:
            right_margin = w - c
        else:
            break

    return top_margin, bottom_margin, left_margin, right_margin


def clear_scanner_bezel(
    image: np.ndarray,
    saturation_thresh: int = 240,
    safety_padding: int = 2
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Detect and blank out saturated scanner edges.
    
    Returns:
        cleaned_image: image with bezel pixels set to 0.
        margins: (top, bottom, left, right) cleared margins.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    top, bottom, left, right = detect_scanner_margins(gray, saturation_thresh=saturation_thresh)
    
    # Add a small safety padding to ensure no edge remnants
    if top > 0:
        top += safety_padding
    if bottom > 0:
        bottom += safety_padding
    if left > 0:
        left += safety_padding
    if right > 0:
        right += safety_padding

    cleaned = image.copy()
    h, w = cleaned.shape[:2]

    if top > 0:
        cleaned[:top] = 0
    if bottom > 0:
        cleaned[h - bottom:] = 0
    if left > 0:
        cleaned[:, :left] = 0
    if right > 0:
        cleaned[:, w - right:] = 0

    return cleaned, (top, bottom, left, right)


def is_boundary_grain(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    cleared_margins: Tuple[int, int, int, int] = (0, 0, 0, 0),
    boundary_margin_px: int = 4
) -> bool:
    """Check if a grain touches or lies within boundary_margin_px of active image edges.
    
    Args:
        bbox: (x, y, w, h)
        image_shape: (height, width)
        cleared_margins: (top, bottom, left, right)
        boundary_margin_px: safety distance
    """
    x, y, bw, bh = bbox
    img_h, img_w = image_shape[:2]
    top_m, bot_m, left_m, right_m = cleared_margins

    effective_min_x = left_m + boundary_margin_px
    effective_max_x = img_w - right_m - boundary_margin_px
    effective_min_y = top_m + boundary_margin_px
    effective_max_y = img_h - bot_m - boundary_margin_px

    if x <= effective_min_x or (x + bw) >= effective_max_x:
        return True
    if y <= effective_min_y or (y + bh) >= effective_max_y:
        return True

    return False
