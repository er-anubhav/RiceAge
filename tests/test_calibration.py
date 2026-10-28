import pytest
import os
import numpy as np
from PIL import Image

from src.preprocessing import inspect_metadata, resolve_calibration, CalibrationResult


def test_inspect_metadata_valid(tmp_path):
    img_path = tmp_path / "test_img.jpg"
    img = Image.new("RGB", (200, 300), color=(100, 100, 100))
    img.save(img_path, dpi=(300, 300))

    meta = inspect_metadata(str(img_path))
    assert meta["width_px"] == 200
    assert meta["height_px"] == 300
    assert meta["mode"] == "RGB"
    assert meta["dpi"] == (300, 300)


def test_inspect_metadata_missing():
    with pytest.raises(FileNotFoundError):
        inspect_metadata("non_existent_file.jpg")


def test_resolve_calibration_assumed():
    calib = resolve_calibration("data/samples/sample_1489x2048.jpg", assumed_dpi=600.0)
    assert not calib.is_verified
    assert calib.effective_dpi == 600.0
    assert pytest.approx(calib.pixel_scale_mm, 0.001) == 25.4 / 600.0


def test_resolve_calibration_reference_derivation():
    calib = resolve_calibration(
        image_path="data/samples/sample_1489x2048.jpg",
        raw_reference_path="data/raw/laserjet_1200dpi_sample.jpg"
    )
    assert calib.is_verified
    assert pytest.approx(calib.effective_dpi, 0.1) == 175.18
    assert pytest.approx(calib.pixel_scale_mm, 0.0001) == 0.14499
