import pytest
import numpy as np
import cv2

from src.segmentation import (
    segment_otsu,
    fill_binary_holes,
    remove_small_objects,
    cleanup_mask,
    compare_segmentation_methods,
)


def test_segment_otsu():
    np.random.seed(42)
    # Realistic scanner image with dark background ~50 and bright foreground ~180
    img = np.random.normal(50, 5, (100, 100)).astype(np.uint8)
    fg = np.random.normal(180, 5, (40, 40)).astype(np.uint8)
    img[30:70, 30:70] = fg
    mask, thresh = segment_otsu(img)
    assert mask.shape == (100, 100)
    assert 55 < thresh < 150
    assert mask[50, 50] == 255
    assert mask[10, 10] == 0


def test_segment_otsu_trimodal_collapse_guard():
    # Simulate a trimodal image: pitch-black border margin (0), background noise (50), bright grains (180).
    # Uncontrolled Otsu splits 0 from 50 (threshold ~25), classifying 90% of the image as foreground.
    # The guardrail ensures thresholding clamps above the background mode.
    img = np.full((200, 200), 50, dtype=np.uint8) # Background: 50
    img[:, :40] = 0 # 20% pitch-black border margin (0)
    img[100:120, 100:120] = 180 # Grain: 180 (only 1% of area)
    
    mask, thresh = segment_otsu(img, max_fg_ratio=0.20)
    assert thresh >= 100.0 # Clamped above background noise
    assert mask[110, 110] == 255 # Grain is preserved
    assert mask[50, 50] == 0 # Background is rejected
    assert mask[50, 10] == 0 # Pitch-black border is rejected
    assert np.mean(mask > 0) < 0.05 # Foreground ratio is physically plausible


def test_fill_binary_holes():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    mask[45:55, 45:55] = 0 # internal hole
    assert mask[50, 50] == 0
    filled = fill_binary_holes(mask)
    assert filled[50, 50] == 255


def test_remove_small_objects():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 255 # Large object: 1600 px
    mask[80:83, 80:83] = 255 # Tiny dust: 9 px
    cleaned = remove_small_objects(mask, min_area_px=50)
    assert cleaned[30, 30] == 255
    assert cleaned[81, 81] == 0


def test_cleanup_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:70, 20:70] = 255
    mask[40:50, 40:50] = 0 # hole
    mask[85:87, 85:87] = 255 # dust
    clean = cleanup_mask(mask, kernel_size=3, min_area_px=40, fill_holes=True)
    assert clean[45, 45] == 255 # hole filled
    assert clean[86, 86] == 0 # dust removed
