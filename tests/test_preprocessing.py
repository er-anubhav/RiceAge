"""Tests for preprocessing, border clearing, and margin detection."""

import pytest
import numpy as np

from src.preprocessing.border_clearing import (
    detect_scanner_margins,
    clear_scanner_bezel,
    is_boundary_grain
)
from src.preprocessing.denoise import preprocess_image


def test_detect_scanner_margins_clean_image():
    # Uniform dark image has zero margins
    img = np.full((200, 200), 50, dtype=np.uint8)
    top, bottom, left, right = detect_scanner_margins(img, saturation_thresh=240)
    assert top == 0
    assert bottom == 0
    assert left == 0
    assert right == 0


def test_detect_scanner_margins_with_bezel():
    img = np.full((200, 200), 50, dtype=np.uint8)
    # Add bottom margin of 10 rows and right margin of 8 cols
    img[-10:, :] = 255
    img[:, -8:] = 255

    top, bottom, left, right = detect_scanner_margins(img, saturation_thresh=240)
    assert top == 0
    assert bottom == 10
    assert left == 0
    assert right == 8


def test_clear_scanner_bezel():
    img = np.full((200, 200), 50, dtype=np.uint8)
    img[-10:, :] = 255
    cleaned, margins = clear_scanner_bezel(img, saturation_thresh=240, safety_padding=1)
    
    # Margin bottom should be detected and cleared with padding
    assert margins[1] == 11
    assert cleaned[-11:, :].max() == 0


def test_is_boundary_grain():
    shape = (200, 200)
    # Grain at center
    assert not is_boundary_grain((100, 100, 20, 20), shape, boundary_margin_px=5)
    # Grain touching left edge
    assert is_boundary_grain((2, 100, 20, 20), shape, boundary_margin_px=5)
    # Grain touching bottom edge
    assert is_boundary_grain((100, 195, 20, 20), shape, boundary_margin_px=5)


def test_preprocess_image_empty_and_noisy():
    # Empty image
    empty = np.zeros((100, 100, 3), dtype=np.uint8)
    gray, smoothed = preprocess_image(empty, gaussian_kernel=3)
    assert gray.shape == (100, 100)
    assert smoothed.shape == (100, 100)
    assert smoothed.max() == 0

    # Noisy image
    np.random.seed(42)
    noisy = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    gray, smoothed = preprocess_image(noisy, gaussian_kernel=5, gaussian_sigma=1.0)
    assert smoothed.std() < gray.std()
