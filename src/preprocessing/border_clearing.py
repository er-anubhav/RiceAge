"""Scanner bezel detection, margin clearing, and boundary grain identification."""

from typing import Tuple
import numpy as np
import cv2


def _scan_edge_margin(line_means: np.ndarray, max_search: int, saturation_thresh: int) -> int:
    margin = 0
    for idx in range(min(len(line_means), max_search)):
        if line_means[idx] >= saturation_thresh:
            margin = idx + 1
        else:
            break
    return margin


def detect_scanner_margins(
    gray_image: np.ndarray,
    saturation_thresh: int = 240,
    max_search_ratio: float = 0.05
) -> Tuple[int, int, int, int]:
    """Detect saturated scanner glass bezel lines along image borders."""
    h, w = gray_image.shape[:2]
    max_h = int(h * max_search_ratio)
    max_w = int(w * max_search_ratio)

    row_means = gray_image.mean(axis=1)
    col_means = gray_image.mean(axis=0)

    top = _scan_edge_margin(row_means, max_h, saturation_thresh)
    bottom = _scan_edge_margin(row_means[::-1], max_h, saturation_thresh)
    left = _scan_edge_margin(col_means, max_w, saturation_thresh)
    right = _scan_edge_margin(col_means[::-1], max_w, saturation_thresh)

    return top, bottom, left, right


def clear_scanner_bezel(
    image: np.ndarray,
    saturation_thresh: int = 240,
    safety_padding: int = 2
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Detect and blank out saturated scanner edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    top, bottom, left, right = detect_scanner_margins(gray, saturation_thresh=saturation_thresh)

    top = top + safety_padding if top > 0 else 0
    bottom = bottom + safety_padding if bottom > 0 else 0
    left = left + safety_padding if left > 0 else 0
    right = right + safety_padding if right > 0 else 0

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
    """Check if a grain touches or lies within boundary_margin_px of active image edges."""
    x, y, bw, bh = bbox
    img_h, img_w = image_shape[:2]
    top_m, bot_m, left_m, right_m = cleared_margins

    min_x = left_m + boundary_margin_px
    max_x = img_w - right_m - boundary_margin_px
    min_y = top_m + boundary_margin_px
    max_y = img_h - bot_m - boundary_margin_px

    return x <= min_x or (x + bw) >= max_x or y <= min_y or (y + bh) >= max_y

