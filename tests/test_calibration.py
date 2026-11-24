import cv2
from src.measurement import measure_all_grains
from src.hardware.dual_platen import DualPlatenManager
from src.dual_view.registration import DualViewRegistration

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


def test_physical_grain_dimension_consistency_300_vs_600_dpi(tmp_path):
    """
    Regression test: Verifies that the same physical grain (e.g. 6.5mm x 2.1mm)
    produces consistent physical length and width when scanned at 300 DPI vs 600 DPI.
    """
    target_len_mm = 6.5
    target_wid_mm = 2.1

    # 300 DPI representation
    scale_300 = 25.4 / 300.0
    len_px_300 = int(round(target_len_mm / scale_300))
    wid_px_300 = int(round(target_wid_mm / scale_300))

    mask_300 = np.zeros((300, 300), dtype=np.int32)
    cv2.ellipse(mask_300, (150, 150), (len_px_300 // 2, wid_px_300 // 2), 0, 0, 360, 1, -1)
    rgb_300 = np.full((300, 300, 3), 200, dtype=np.uint8)

    calib_300 = CalibrationResult(
        is_verified=True,
        effective_dpi=300.0,
        pixel_scale_mm=scale_300,
        pixels_per_mm=300.0 / 25.4,
        source_info="300 DPI Calibration"
    )

    df_300, _ = measure_all_grains(
        instance_mask=mask_300,
        image_rgb=rgb_300,
        grain_diagnostics={1: {"status": "single_grain"}},
        calibration=calib_300,
        min_area_px=10
    )

    # 600 DPI representation (2x pixels in each dimension)
    scale_600 = 25.4 / 600.0
    len_px_600 = int(round(target_len_mm / scale_600))
    wid_px_600 = int(round(target_wid_mm / scale_600))

    mask_600 = np.zeros((600, 600), dtype=np.int32)
    cv2.ellipse(mask_600, (300, 300), (len_px_600 // 2, wid_px_600 // 2), 0, 0, 360, 1, -1)
    rgb_600 = np.full((600, 600, 3), 200, dtype=np.uint8)

    calib_600 = CalibrationResult(
        is_verified=True,
        effective_dpi=600.0,
        pixel_scale_mm=scale_600,
        pixels_per_mm=600.0 / 25.4,
        source_info="600 DPI Calibration"
    )

    df_600, _ = measure_all_grains(
        instance_mask=mask_600,
        image_rgb=rgb_600,
        grain_diagnostics={1: {"status": "single_grain"}},
        calibration=calib_600,
        min_area_px=10
    )

    assert len(df_300) == 1
    assert len(df_600) == 1

    l300 = df_300["length_mm"].iloc[0]
    w300 = df_300["width_mm"].iloc[0]
    l600 = df_600["length_mm"].iloc[0]
    w600 = df_600["width_mm"].iloc[0]

    # Verify physical dimensions match closely across 300 and 600 DPI
    assert pytest.approx(l300, abs=0.15) == l600
    assert pytest.approx(w300, abs=0.10) == w600
    assert pytest.approx(l300, abs=0.20) == target_len_mm
    assert pytest.approx(w300, abs=0.15) == target_wid_mm


def test_metadata_dpi_derivation_300_and_600(tmp_path):
    """Verifies that direct 300 DPI and 600 DPI metadata in JFIF/EXIF is correctly parsed."""
    # 300 DPI image
    p300 = str(tmp_path / "scan_300.jpg")
    img300 = Image.new("RGB", (500, 500), color=(128, 128, 128))
    img300.save(p300, dpi=(300, 300))

    calib300 = resolve_calibration(p300)
    assert calib300.is_verified is True
    assert calib300.effective_dpi == 300.0
    assert pytest.approx(calib300.pixel_scale_mm, 0.0001) == 25.4 / 300.0

    # 600 DPI image
    p600 = str(tmp_path / "scan_600.jpg")
    img600 = Image.new("RGB", (1000, 1000), color=(128, 128, 128))
    img600.save(p600, dpi=(600, 600))

    calib600 = resolve_calibration(p600)
    assert calib600.is_verified is True
    assert calib600.effective_dpi == 600.0
    assert pytest.approx(calib600.pixel_scale_mm, 0.0001) == 25.4 / 600.0


def test_reject_unsupported_or_unknown_dpi(tmp_path):
    """Verifies that unsupported or unknown DPIs are rejected rather than silently assuming 300 DPI."""
    dummy_img = str(tmp_path / "test.jpg")
    Image.new("RGB", (100, 100)).save(dummy_img)

    # resolve_calibration must reject unsupported assumed_dpi
    with pytest.raises(ValueError, match="Unsupported acquisition DPI"):
        resolve_calibration(dummy_img, assumed_dpi=150.0)

    with pytest.raises(ValueError, match="Unsupported acquisition DPI"):
        resolve_calibration(dummy_img, assumed_dpi=-5.0)

    # DualPlatenManager must reject unsupported scan_resolution
    with pytest.raises(ValueError, match="Unsupported scanner acquisition DPI"):
        DualPlatenManager(scan_resolution=400)

    with pytest.raises(ValueError, match="Unsupported scanner acquisition DPI"):
        DualPlatenManager(scan_resolution={"bottom": 300, "top": 150})


def test_independent_scanner_dpi_calibration():
    """Verifies top and bottom scanner calibrations are handled independently."""
    mgr = DualPlatenManager(scan_resolution={"bottom": 300, "top": 600})
    assert mgr.get_scanner_dpi("bottom") == 300
    assert mgr.get_scanner_dpi("top") == 600

    # Check homography scaling for heterogeneous DPI
    reg = DualViewRegistration(homography_matrix=None, dpi_bottom=300, dpi_top=600)
    assert reg.is_healthy is True
    assert reg.dpi_bottom == 300
    assert reg.dpi_top == 600

    # Homography for 300 DPI bottom to 600 DPI top
    # Scale ratio = 600 / 300 = 2.0
    assert pytest.approx(reg.scale_ratio, 0.01) == 2.0

    # Project a point from bottom platen
    success, xt, yt = reg.project_bottom_to_top(100.0, 100.0)
    assert success is True
    # At 300->600, x_t ≈ -100 * 2.0 + 4917.872 = 4717.872
    assert xt > 4000.0
