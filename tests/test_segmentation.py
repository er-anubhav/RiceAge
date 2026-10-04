"""Tests for binarization, threshold comparison, and morphological cleanup."""

import pytest
import numpy as np
import cv2

from src.segmentation.binarize import (
    segment_otsu,
    segment_adaptive,
    segment_triangle,
    segment_yen,
    segment_li,
    compare_segmentation_methods
)
from src.segmentation.morphology import (
    fill_binary_holes,
    remove_small_objects,
    cleanup_mask
)


def test_segment_otsu_bimodal():
    # Synthetic image with dark background (50) and bright grain (200)
    img = np.full((100, 100), 50, dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, 200, -1)

    binary, thresh_val = segment_otsu(img)
    assert 50 <= thresh_val <= 200
    # Center should be foreground (255)
    assert binary[50, 50] == 255
    # Corner should be background (0)
    assert binary[5, 5] == 0


def test_compare_segmentation_methods():
    img = np.full((100, 100), 40, dtype=np.uint8)
    cv2.ellipse(img, (50, 50), (25, 10), 0, 0, 360, 220, -1)

    comparison = compare_segmentation_methods(img)
    assert "otsu" in comparison
    assert "triangle" in comparison
    assert "yen" in comparison
    assert "li" in comparison
    assert "adaptive_gaussian" in comparison

    otsu_res = comparison["otsu"]
    assert otsu_res["foreground_ratio"] > 0.05
    assert otsu_res["raw_connected_components"] >= 1


def test_fill_binary_holes():
    # Mask with a hollow ring
    mask = np.zeros((60, 60), dtype=np.uint8)
    cv2.circle(mask, (30, 30), 20, 255, -1)
    # Punch hole in center
    cv2.circle(mask, (30, 30), 5, 0, -1)
    assert mask[30, 30] == 0

    filled = fill_binary_holes(mask)
    assert filled[30, 30] == 255


def test_remove_small_objects():
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Large object (area ~314)
    cv2.circle(mask, (30, 30), 10, 255, -1)
    # Tiny speck (area ~4)
    cv2.circle(mask, (80, 80), 1, 255, -1)

    clean = remove_small_objects(mask, min_area_px=20)
    assert clean[30, 30] == 255
    assert clean[80, 80] == 0


def test_cleanup_mask_empty_and_full():
    empty = np.zeros((50, 50), dtype=np.uint8)
    assert cleanup_mask(empty).max() == 0

    full = np.full((50, 50), 255, dtype=np.uint8)
    assert cleanup_mask(full).min() == 255
