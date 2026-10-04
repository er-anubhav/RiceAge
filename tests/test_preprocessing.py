import pytest
import os
import numpy as np
from PIL import Image

from src.preprocessing.border_clearing import detect_scanner_margins, clear_scanner_bezel, is_boundary_grain
from src.preprocessing.denoise import preprocess_image
from src.preprocessing.calibration import inspect_metadata, resolve_calibration, CalibrationResult


@pytest.fixture
def dummy_1200dpi_image(tmp_path):
    img_path = str(tmp_path / "test_1200dpi.jpg")
    im = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    im.save(img_path, dpi=(1200, 1200))
    return img_path


@pytest.fixture
def dummy_no_dpi_image(tmp_path):
    img_path = str(tmp_path / "test_nodpi.png")
    im = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
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


def test_resolve_calibration_derived(dummy_no_dpi_image, tmp_path):
    raw_path = str(tmp_path / "raw.jpg")
    im = Image.fromarray(np.zeros((10200, 14028, 3), dtype=np.uint8))
    im.save(raw_path, dpi=(1200, 1200))

    scaled_path = str(tmp_path / "scaled.jpg")
    im_scaled = Image.fromarray(np.zeros((1489, 2048, 3), dtype=np.uint8))
    im_scaled.save(scaled_path)

    res = resolve_calibration(scaled_path, raw_reference_path=raw_path)
    assert res.is_verified
    assert pytest.approx(res.effective_dpi, rel=0.01) == 175.18
    assert res.pixel_scale_mm is not None


def test_detect_scanner_margins_clean_image():
    img = np.full((200, 200), 50, dtype=np.uint8)
    assert detect_scanner_margins(img, saturation_thresh=240) == (0, 0, 0, 0)


def test_detect_scanner_margins_with_bezel():
    img = np.full((200, 200), 50, dtype=np.uint8)
    img[-10:, :] = 255
    img[:, -8:] = 255
    assert detect_scanner_margins(img, saturation_thresh=240) == (0, 10, 0, 8)


def test_clear_scanner_bezel():
    img = np.full((200, 200), 50, dtype=np.uint8)
    img[-10:, :] = 255
    cleaned, margins = clear_scanner_bezel(img, saturation_thresh=240, safety_padding=1)
    assert margins[1] == 11
    assert cleaned[-11:, :].max() == 0


def test_is_boundary_grain():
    shape = (200, 200)
    assert not is_boundary_grain((100, 100, 20, 20), shape, boundary_margin_px=5)
    assert is_boundary_grain((2, 100, 20, 20), shape, boundary_margin_px=5)
    assert is_boundary_grain((100, 195, 20, 20), shape, boundary_margin_px=5)


def test_preprocess_image_empty_and_noisy():
    empty = np.zeros((100, 100, 3), dtype=np.uint8)
    gray, smoothed = preprocess_image(empty, gaussian_kernel=3)
    assert gray.shape == (100, 100) and smoothed.max() == 0

    np.random.seed(42)
    noisy = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    gray, smoothed = preprocess_image(noisy, gaussian_kernel=5, gaussian_sigma=1.0)
    assert smoothed.std() < gray.std()
