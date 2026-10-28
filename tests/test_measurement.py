import pytest
import numpy as np
import cv2

from src.measurement import extract_grain_morphology, extract_grain_color, measure_all_grains
from src.preprocessing import CalibrationResult


def test_extract_grain_morphology():
    # Synthetic ellipse of known major/minor axes
    mask = np.zeros((100, 100), dtype=np.uint8)
    # axes: half-length = 30, half-width = 10 -> length ~ 60, width ~ 20
    cv2.ellipse(mask, (50, 50), (30, 10), 0, 0, 360, 255, -1)

    calib = CalibrationResult(
        is_verified=True,
        effective_dpi=300.0,
        pixel_scale_mm=0.084667,
        pixels_per_mm=11.81,
        source_info="Test calibration"
    )

    morph = extract_grain_morphology(
        grain_id=1,
        grain_mask=mask,
        image_shape=(100, 100),
        calibration=calib,
        separation_status="isolated_single"
    )

    assert morph is not None
    assert morph.grain_id == 1
    assert pytest.approx(morph.length_px, abs=4.0) == 60.0
    assert pytest.approx(morph.width_px, abs=3.0) == 20.0
    assert pytest.approx(morph.aspect_ratio, abs=0.5) == 3.0
    assert morph.length_mm is not None
    assert pytest.approx(morph.length_mm, abs=0.5) == 60.0 * 0.084667


def test_extract_grain_color():
    rgb = np.full((50, 50, 3), [180, 190, 200], dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[15:35, 15:35] = 255

    color = extract_grain_color(grain_id=1, rgb_image=rgb, grain_mask=mask)
    assert color is not None
    assert pytest.approx(color.mean_r, abs=1.0) == 180.0
    assert pytest.approx(color.mean_g, abs=1.0) == 190.0
    assert pytest.approx(color.mean_b, abs=1.0) == 200.0
