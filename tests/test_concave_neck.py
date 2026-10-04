"""Unit tests for concave neck detection, contour cutline pairing, and hybrid separation."""

import pytest
import numpy as np
import cv2

from src.separation.concave_neck import (
    detect_concave_points,
    is_cutline_valid,
    pair_concave_points,
    split_cluster_concave_neck,
)
from src.separation.hybrid_separation import benchmark_and_separate_hybrid


def test_detect_concave_points_smooth_convex():
    # Circle has no concavities
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, 255, -1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = contours[0]

    concave = detect_concave_points(c, min_defect_depth_px=2.0)
    assert len(concave) == 0


def test_detect_concave_points_dumbbell_shape():
    # Dumbbell: two circles connected with a thin neck
    img = np.zeros((120, 120), dtype=np.uint8)
    # Left bulb
    cv2.circle(img, (40, 60), 25, 255, -1)
    # Right bulb
    cv2.circle(img, (80, 60), 25, 255, -1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)

    concave = detect_concave_points(c, min_defect_depth_px=2.0)
    # Should detect 2 opposing pinch points at top and bottom of the neck
    assert len(concave) == 2
    pts = [d["point"] for d in concave]
    # The pinch points should be near x=60
    assert abs(pts[0][0] - 60) <= 5
    assert abs(pts[1][0] - 60) <= 5


def test_is_cutline_valid():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)

    # Valid line within rectangle
    assert is_cutline_valid((30, 30), (70, 70), mask)

    # Cutline passing through background (punch a hole in middle)
    cv2.circle(mask, (50, 50), 10, 0, -1)
    assert not is_cutline_valid((30, 30), (70, 70), mask)


def test_pair_concave_points():
    mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(mask, (40, 60), 25, 255, -1)
    cv2.circle(mask, (80, 60), 25, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)

    concave = detect_concave_points(c, min_defect_depth_px=2.0)
    pairs = pair_concave_points(concave, mask, max_neck_width_px=60.0)
    assert len(pairs) == 1
    p1, p2 = pairs[0]
    assert abs(p1[0] - p2[0]) <= 5


def test_split_cluster_concave_neck():
    mask = np.zeros((120, 120), dtype=np.uint8)
    # Two touching ellipses
    cv2.ellipse(mask, (45, 60), (22, 10), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (75, 60), (22, 10), 0, 0, 360, 255, -1)

    labeled, num_pieces, cuts = split_cluster_concave_neck(mask, min_defect_depth_px=2.0)
    assert num_pieces == 2
    assert len(cuts) == 1


def test_benchmark_and_separate_hybrid_synthetic():
    mask = np.zeros((200, 200), dtype=np.uint8)
    # 1 isolated ellipse
    cv2.ellipse(mask, (50, 50), (20, 8), 0, 0, 360, 255, -1)
    # 1 touching pair (cluster)
    cv2.ellipse(mask, (120, 120), (20, 9), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (150, 120), (20, 9), 0, 0, 360, 255, -1)

    mask_ws, mask_cnc, mask_hyb, bench_df, summary_b, rois = benchmark_and_separate_hybrid(
        mask,
        cluster_area_thresh_px=600,
        solidity_thresh=0.92,
        min_valid_area_px=20,
        median_grain_area_px=500.0
    )

    assert summary_b["total_clusters_evaluated"] == 1
    assert summary_b["method_c_hybrid_resolved"] == 1
    assert len(bench_df) == 1
