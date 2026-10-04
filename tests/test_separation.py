import pytest
import numpy as np
import cv2

from src.separation import (
    extract_connected_components,
    summarize_connected_components,
    compute_distance_transform,
    detect_distance_peaks,
    separate_touching_grains,
    detect_concave_points,
    is_cutline_valid,
    pair_concave_points,
    split_cluster_concave_neck,
    benchmark_and_separate_hybrid,
)


def test_connected_components():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (30, 30), 10, 255, -1)
    cv2.circle(mask, (70, 70), 10, 255, -1)
    num_labels, labels, stats, centroids = extract_connected_components(mask)
    assert num_labels == 3
    summary = summarize_connected_components(stats, min_area=10)
    assert summary["valid_components"] == 2


def test_distance_transform_peaks():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 20, 255, -1)
    dist = compute_distance_transform(mask)
    assert dist[50, 50] == pytest.approx(20.0, abs=1.0)
    peaks = detect_distance_peaks(dist, min_distance=5)
    assert len(peaks) == 1
    assert abs(peaks[0][0] - 50) <= 1
    assert abs(peaks[0][1] - 50) <= 1


def test_touching_grains_watershed_separation():
    mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.ellipse(mask, (45, 60), (22, 10), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (75, 60), (22, 10), 0, 0, 360, 255, -1)
    num_cc, _, _, _ = cv2.connectedComponentsWithStats(mask)
    assert num_cc == 2

    inst_mask, diags, stats = separate_touching_grains(
        mask, min_peak_distance_px=6, cluster_area_thresh_px=300, solidity_thresh=0.95
    )
    unique_ids = np.unique(inst_mask)
    grains = unique_ids[unique_ids > 0]
    assert len(grains) == 2
    assert stats["clusters_successfully_split"] == 1


def test_isolated_single_grain_not_oversegmented():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.ellipse(mask, (50, 50), (20, 9), 30, 0, 360, 255, -1)
    inst_mask, diags, stats = separate_touching_grains(
        mask, min_peak_distance_px=6, cluster_area_thresh_px=600, solidity_thresh=0.90
    )
    unique_ids = np.unique(inst_mask)
    grains = unique_ids[unique_ids > 0]
    assert len(grains) == 1
    assert diags[grains[0]]["status"] == "isolated_single"


def test_empty_mask_separation():
    empty = np.zeros((50, 50), dtype=np.uint8)
    inst_mask, diags, stats = separate_touching_grains(empty)
    assert inst_mask.max() == 0
    assert len(diags) == 0
    assert stats["total_detected_grains"] == 0


def test_detect_concave_points_smooth_convex():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, 255, -1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert len(detect_concave_points(contours[0], min_defect_depth_px=2.0)) == 0


def test_detect_concave_points_dumbbell_shape():
    img = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(img, (40, 60), 25, 255, -1)
    cv2.circle(img, (80, 60), 25, 255, -1)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    concave = detect_concave_points(c, min_defect_depth_px=2.0)
    assert len(concave) == 2
    pts = [d["point"] for d in concave]
    assert abs(pts[0][0] - 60) <= 5
    assert abs(pts[1][0] - 60) <= 5


def test_is_cutline_valid():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)
    assert is_cutline_valid((30, 30), (70, 70), mask)
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
    assert abs(pairs[0][0][0] - pairs[0][1][0]) <= 5


def test_split_cluster_concave_neck():
    mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.ellipse(mask, (45, 60), (22, 10), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (75, 60), (22, 10), 0, 0, 360, 255, -1)
    labeled, num_pieces, cuts = split_cluster_concave_neck(mask, min_defect_depth_px=2.0)
    assert num_pieces == 2
    assert len(cuts) == 1


def test_benchmark_and_separate_hybrid_synthetic():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.ellipse(mask, (50, 50), (20, 8), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (120, 120), (20, 9), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (150, 120), (20, 9), 0, 0, 360, 255, -1)

    _, _, _, bench_df, summary_b, _ = benchmark_and_separate_hybrid(
        mask, cluster_area_thresh_px=600, solidity_thresh=0.92, min_valid_area_px=20, median_grain_area_px=500.0
    )
    assert summary_b["total_clusters_evaluated"] == 1
    assert summary_b["method_c_hybrid_resolved"] == 1
    assert len(bench_df) == 1
