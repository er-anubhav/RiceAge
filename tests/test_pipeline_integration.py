import pytest
import os
import numpy as np
import cv2

from configs.default_config import PipelineConfig
from src.pipeline import RiceQualityPipeline


def test_pipeline_on_synthetic_fixture(tmp_path):
    # Create synthetic test image with 3 simulated rice grains of realistic dimensions
    img = np.full((1489, 2048, 3), 55, dtype=np.uint8) # scanner background
    cv2.ellipse(img, (500, 500), (20, 7), 30, 0, 360, (180, 180, 180), -1)
    cv2.ellipse(img, (900, 500), (20, 7), -45, 0, 360, (190, 190, 190), -1)
    cv2.ellipse(img, (700, 900), (20, 7), 0, 0, 360, (185, 185, 185), -1)

    img_path = str(tmp_path / "synthetic_sample.jpg")
    cv2.imwrite(img_path, img)

    cfg = PipelineConfig()
    cfg.preprocessing.boundary_margin_px = 2
    cfg.export.output_dir = str(tmp_path / "outputs")
    pipeline = RiceQualityPipeline(config=cfg)

    res = pipeline.run(img_path, generate_outputs=False)
    assert res["summary"]["total_grains_detected"] == 3
    assert len(res["measurements_df"]) == 3
    assert res["instance_mask"].max() == 3


def test_pipeline_baseline_sample_execution():
    # End-to-end regression test on standard sample_1489x2048.jpg
    cfg = PipelineConfig()
    pipeline = RiceQualityPipeline(config=cfg)

    res = pipeline.run("data/samples/sample_1489x2048.jpg", generate_outputs=False)
    summary = res["summary"]
    # With hybrid separation, 734 grains are detected and all 80 clusters are split (80/80)
    assert summary["total_grains_detected"] == 734
    assert summary["separation"]["clusters_detected"] == 80
    assert summary["separation"]["clusters_successfully_split"] == 80
    assert summary["separation"]["clusters_unresolved"] == 0
