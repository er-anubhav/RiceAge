"""Tests for image calibration and metadata resolution."""

import pytest
import os
import numpy as np
from PIL import Image

from src.preprocessing.calibration import (
    inspect_metadata,
    resolve_calibration,
    CalibrationResult
)


@pytest.fixture
def dummy_1200dpi_image(tmp_path):
    """Create a temporary image with 1200 DPI metadata."""
    img_path = str(tmp_path / "test_1200dpi.jpg")
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    im = Image.fromarray(arr)
    im.save(img_path, dpi=(1200, 1200))
    return img_path


@pytest.fixture
def dummy_no_dpi_image(tmp_path):
    """Create a temporary image with no DPI metadata."""
    img_path = str(tmp_path / "test_nodpi.png")
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    im = Image.fromarray(arr)
    im.save(img_path)
    return img_path


def test_inspect_metadata_missing_file():
    with pytest.raises(FileNotFoundError):
        inspect_metadata("non_existent_path.jpg")


def test_inspect_metadata_valid(dummy_1200dpi_image):
    meta = inspect_metadata(dummy_1200dpi_image)
    assert meta["width_px"] == 100
    assert meta["height_px"] == 100
    assert meta["dpi"] == (1200, 1200)


def test_resolve_calibration_uncalibrated(dummy_no_dpi_image):
    res = resolve_calibration(dummy_no_dpi_image)
    assert not res.is_verified
    assert res.effective_dpi is None
    assert res.pixel_scale_mm is None


def test_resolve_calibration_assumed_dpi(dummy_no_dpi_image):
    res = resolve_calibration(dummy_no_dpi_image, assumed_dpi=600.0)
    assert not res.is_verified
    assert res.effective_dpi == 600.0
    assert pytest.approx(res.pixel_scale_mm, rel=1e-3) == 25.4 / 600.0


def test_resolve_calibration_derived(tmp_path):
    # Simulate raw 1200 DPI image (12000 x 9000)
    raw_path = str(tmp_path / "raw.jpg")
    im_raw = Image.fromarray(np.zeros((9000, 12000, 3), dtype=np.uint8))
    im_raw.save(raw_path, dpi=(1200, 1200))

    # Simulate downsampled image (2000 x 1500) -> 6.0x scale
    down_path = str(tmp_path / "down.jpg")
    im_down = Image.fromarray(np.zeros((1500, 2000, 3), dtype=np.uint8))
    im_down.save(down_path)

    res = resolve_calibration(down_path, raw_reference_path=raw_path)
    assert res.is_verified
    assert pytest.approx(res.effective_dpi, rel=1e-2) == 200.0
    assert pytest.approx(res.scale_factor_to_raw, rel=1e-2) == 6.0
