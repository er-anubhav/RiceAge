"""Tests for per-grain morphology, geometry, and color metrics."""

import pytest
import math
import numpy as np
import cv2

from src.preprocessing.calibration import CalibrationResult
from src.measurement.morphology_metrics import extract_grain_morphology
from src.measurement.color_metrics import extract_grain_color


def test_extract_grain_morphology_known_circle():
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Circle with radius 15 at (50, 50)
    # Area = pi * r^2 = 706.85, Perimeter = 2 * pi * r = 94.25
    cv2.circle(mask, (50, 50), 15, 255, -1)

    morph = extract_grain_morphology(
        grain_id=1,
        grain_mask=mask,
        image_shape=(100, 100)
    )

    assert morph is not None
    assert morph.grain_id == 1
    assert abs(morph.centroid_x - 50.0) < 1.0
    assert abs(morph.centroid_y - 50.0) < 1.0
    # Circularity for circle should be close to 1.0
    assert morph.circularity > 0.85
    # Aspect ratio should be ~1.0
    assert pytest.approx(morph.aspect_ratio, abs=0.15) == 1.0


def test_extract_grain_morphology_calibrated():
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Ellipse with axes 20 (length 40) and 10 (width 20)
    cv2.ellipse(mask, (50, 50), (20, 10), 0, 0, 360, 255, -1)

    calib = CalibrationResult(
        is_verified=True,
        effective_dpi=254.0,  # 10 pixels per mm -> 0.1 mm/px
        pixel_scale_mm=0.10,
        pixels_per_mm=10.0,
        source_info="Test calibration"
    )

    morph = extract_grain_morphology(
        grain_id=2,
        grain_mask=mask,
        image_shape=(100, 100),
        calibration=calib
    )

    assert morph is not None
    assert morph.length_mm is not None
    assert morph.width_mm is not None
    # 40 px * 0.1 mm/px = ~4.0 mm
    assert pytest.approx(morph.length_mm, abs=0.5) == 4.0
    # 20 px * 0.1 mm/px = ~2.0 mm
    assert pytest.approx(morph.width_mm, abs=0.5) == 2.0


def test_extract_grain_color():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[:, :] = [180, 160, 140]  # R=180, G=160, B=140
    # Add small specular glint
    rgb[48:52, 48:52] = [255, 255, 255]

    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 15, 255, -1)

    color = extract_grain_color(grain_id=1, rgb_image=rgb, grain_mask=mask, specular_thresh=240)
    assert color is not None
    assert color.mean_r > 175
    assert color.mean_g > 155
    assert color.mean_b > 135
    assert color.highlight_ratio > 0.0
