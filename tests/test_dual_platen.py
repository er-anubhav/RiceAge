import os
import pytest
import numpy as np
import cv2
import pandas as pd

from configs.default_config import PipelineConfig
from src.pipeline import RiceQualityPipeline
from src.hardware.dual_platen import (
    create_distortion_free_composite,
    DualPlatenManager
)


def test_distortion_free_composite(tmp_path):
    # Platen 1: 300x200
    img1 = np.full((300, 200, 3), 50, dtype=np.uint8)
    # Platen 2: 320x220 (different height and width)
    img2 = np.full((320, 220, 3), 60, dtype=np.uint8)

    out_file = str(tmp_path / "composite.jpg")
    composite = create_distortion_free_composite(img1, img2, output_path=out_file)

    assert os.path.exists(out_file)
    # Target height should be max(300, 320) = 320
    assert composite.shape[0] == 320
    # Width should be 200 + 8 (divider) + 220 = 428
    assert composite.shape[1] == 200 + 8 + 220


def test_dual_platen_manager_processing(tmp_path):
    # Create two synthetic platen scans with ellipses
    p1_path = str(tmp_path / "platen1.jpg")
    p2_path = str(tmp_path / "platen2.jpg")
    comp_path = str(tmp_path / "composite_out.jpg")

    img1 = np.full((600, 800, 3), 50, dtype=np.uint8)
    cv2.ellipse(img1, (300, 300), (20, 7), 0, 0, 360, (200, 200, 200), -1)
    cv2.ellipse(img1, (500, 300), (20, 7), 0, 0, 360, (200, 200, 200), -1)
    cv2.imwrite(p1_path, img1)

    img2 = np.full((600, 800, 3), 50, dtype=np.uint8)
    cv2.ellipse(img2, (400, 300), (20, 7), 0, 0, 360, (200, 200, 200), -1)
    cv2.imwrite(p2_path, img2)

    cfg = PipelineConfig()
    cfg.segmentation.engine = "classic"
    pipeline = RiceQualityPipeline(config=cfg)

    mgr = DualPlatenManager(pipeline=pipeline)
    res = mgr.run_dual_analysis(p1_path, p2_path, sample_id="TEST-001", composite_output_path=comp_path)

    assert res["status"] == "Complete"
    assert res["total_grains"] == 3
    assert "bottom" in res["platens"]
    assert "top" in res["platens"]
    assert res["platens"]["bottom"]["grains_detected"] == 2
    assert res["platens"]["top"]["grains_detected"] == 1
    assert "mass_basis" in res
    assert "count_basis" in res
    assert os.path.exists(comp_path)
