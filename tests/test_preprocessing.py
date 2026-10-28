import pytest
import numpy as np
import cv2

from src.preprocessing import detect_scanner_margins, clear_scanner_bezel, is_boundary_grain, preprocess_image


def test_detect_scanner_margins_clean():
    # Uniform background with no saturated bezels
    img = np.full((200, 200), 50, dtype=np.uint8)
    top, bottom, left, right = detect_scanner_margins(img, saturation_thresh=240)
    assert top == 0 and bottom == 0 and left == 0 and right == 0


def test_detect_scanner_margins_with_bezel():
    img = np.full((200, 200), 50, dtype=np.uint8)
    # Add saturated margin on top 4 rows and bottom 6 rows
    img[:4, :] = 255
    img[-6:, :] = 255
    top, bottom, left, right = detect_scanner_margins(img, saturation_thresh=240, max_search_ratio=0.10)
    assert top == 4
    assert bottom == 6
    assert left == 0
    assert right == 0


def test_clear_scanner_bezel():
    img = np.full((100, 100, 3), 60, dtype=np.uint8)
    img[-5:, :] = 250 # saturated bottom
    cleaned, margins = clear_scanner_bezel(img, saturation_thresh=240, safety_padding=2)
    assert margins[1] == 7 # 5 + 2 padding
    # Check that cleared region is filled with background median (~60), not 250
    assert cleaned[-1, -1, 0] == 60


def test_is_boundary_grain():
    shape = (500, 500)
    # Grain well inside margins
    assert not is_boundary_grain((100, 100, 20, 40), shape, boundary_margin_px=4)
    # Grain touching left margin
    assert is_boundary_grain((2, 100, 20, 40), shape, boundary_margin_px=4)
    # Grain touching bottom margin
    assert is_boundary_grain((100, 480, 20, 40), shape, boundary_margin_px=4)


def test_preprocess_image():
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr[40:60, 40:60] = 200
    gray, smoothed = preprocess_image(bgr, gaussian_kernel=5, gaussian_sigma=0.8)
    assert gray.shape == (100, 100)
    assert smoothed.shape == (100, 100)
    assert smoothed[50, 50] > 0
