"""Integration tests for the complete RiceQualityPipeline."""

import pytest
import os
import numpy as np
import cv2

from configs.default_config import PipelineConfig
from src.pipeline import RiceQualityPipeline


@pytest.fixture
def synthetic_tray_image(tmp_path):
    """Create a synthetic image with isolated grains, touching grains, and borders."""
    img = np.full((300, 300, 3), 50, dtype=np.uint8)

    # 3 Isolated grains (ellipses)
    cv2.ellipse(img, (70, 70), (20, 8), 25, 0, 360, (190, 185, 180), -1)
    cv2.ellipse(img, (200, 80), (22, 9), 70, 0, 360, (200, 195, 190), -1)
    cv2.ellipse(img, (80, 220), (18, 8), 120, 0, 360, (185, 180, 175), -1)

    # 1 Touching pair
    cv2.ellipse(img, (180, 210), (20, 8), 0, 0, 360, (195, 190, 185), -1)
    cv2.ellipse(img, (208, 210), (20, 8), 0, 0, 360, (195, 190, 185), -1)

    # 1 Border touching grain
    cv2.ellipse(img, (295, 150), (18, 8), 90, 0, 360, (190, 185, 180), -1)

    # Saturated scanner edge (bottom 5 rows)
    img[-5:, :] = (255, 255, 255)

    img_path = str(tmp_path / "synthetic_sample.jpg")
    cv2.imwrite(img_path, img)
    return img_path


def test_pipeline_integration_synthetic(synthetic_tray_image, tmp_path):
    cfg = PipelineConfig()
    cfg.export.output_dir = str(tmp_path / "outputs")
    cfg.export.visualizations_dir = str(tmp_path / "outputs/visualizations")
    cfg.export.crops_dir = str(tmp_path / "outputs/crops")
    cfg.export.masks_dir = str(tmp_path / "outputs/masks")
    cfg.export.reports_dir = str(tmp_path / "outputs/reports")
    cfg.separation.cluster_area_threshold_px = 350

    pipeline = RiceQualityPipeline(config=cfg)
    results = pipeline.run(synthetic_tray_image, generate_outputs=True)

    summary = results["summary"]
    df = results["measurements_df"]

    # At least 5 grains detected (3 isolated + 2 from touching pair + 1 boundary)
    assert summary["total_grains_detected"] >= 5
    assert os.path.exists(os.path.join(cfg.export.visualizations_dir, "original.jpg"))
    assert os.path.exists(os.path.join(cfg.export.visualizations_dir, "segmentation.jpg"))
    assert os.path.exists(os.path.join(cfg.export.visualizations_dir, "instances.jpg"))
    assert os.path.exists(os.path.join(cfg.export.visualizations_dir, "measurements.jpg"))
    assert os.path.exists(os.path.join(cfg.export.output_dir, "grain_measurements.csv"))
    assert os.path.exists(os.path.join(cfg.export.output_dir, "summary.json"))
    assert os.path.exists(os.path.join(cfg.export.reports_dir, "M0_BASELINE_REPORT.md"))


def test_pipeline_integration_real_sample():
    """Verify execution on real sample image if available."""
    sample_path = "data/samples/sample_1489x2048.jpg"
    raw_path = "data/raw/laserjet_1200dpi_sample.jpg"
    if not os.path.exists(sample_path):
        pytest.skip("Real sample not present in data/samples")

    pipeline = RiceQualityPipeline()
    results = pipeline.run(sample_path, raw_reference_path=raw_path, generate_outputs=False)
    summary = results["summary"]
    
    assert summary["total_grains_detected"] > 600
    assert summary["calibration"]["is_verified"] is True
    assert summary["separation"]["clusters_successfully_split"] > 50
