import pytest
import numpy as np
import cv2

from src.separation import (
    evaluate_component_geometry,
    detect_concave_points,
    split_cluster_concave_neck,
    separate_touching_grains,
    ComponentCategory,
)


def create_synthetic_ellipse(center=(50, 50), axes=(30, 15), angle=0, shape=(100, 100)):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    return mask


def test_evaluate_component_geometry_single():
    single_mask = create_synthetic_ellipse(center=(50, 50), axes=(25, 10), angle=30)
    geom = evaluate_component_geometry(
        single_mask,
        solidity_thresh=0.90,
        single_area_range=(300, 1000),
        pair_area_range=(1000, 2000)
    )
    assert geom["category"] == ComponentCategory.SINGLE_GRAIN
    assert geom["solidity"] > 0.92


def test_detect_concave_points_and_split():
    # Create two touching overlapping circles forming a dumbbell
    mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(mask, (55, 60), 25, 255, -1)
    cv2.circle(mask, (95, 60), 25, 255, -1)
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    assert len(cnts) == 1
    
    concave_pts = detect_concave_points(cnts[0], min_defect_depth_px=2.0)
    assert len(concave_pts) >= 2 # top and bottom neck indentations

    labeled, count, pairs = split_cluster_concave_neck(mask, min_defect_depth_px=2.0, max_neck_width_px=60.0)
    assert count == 2 # successfully split into 2 grains
    assert len(pairs) == 1


def test_separate_touching_grains_isolated_preserved():
    # Two well-separated grains
    mask = np.zeros((150, 250), dtype=np.uint8)
    cv2.ellipse(mask, (60, 75), (25, 10), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (180, 75), (25, 10), 45, 0, 360, 255, -1)

    inst_mask, diags, summary = separate_touching_grains(
        mask,
        cluster_area_thresh_px=1200,
        enable_concave_fallback=True
    )
    assert summary["total_detected_grains"] == 2
    assert summary["isolated_single_grains"] == 2
    assert summary["clusters_detected"] == 0


def test_separate_touching_grains_concave_gating_rejects_oversized_debris():
    # Large non-grain debris object (area exceeds 2.5 * cluster_area_thresh_px)
    mask = np.zeros((100, 200), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 25, 255, -1)
    cv2.circle(mask, (90, 50), 25, 255, -1)
    cv2.circle(mask, (130, 50), 25, 255, -1)

    # Set min_peak_distance_px high so watershed doesn't split it
    inst_mask, diags, summary = separate_touching_grains(
        mask,
        min_peak_distance_px=200,
        cluster_area_thresh_px=650,
        enable_concave_fallback=True,
        concave_max_cluster_area_factor=2.5
    )
    # Gating rejects concave splitting because area (~5400px) > 2.5 * 650
    assert summary["clusters_split_by_concave"] == 0
    assert summary["total_detected_grains"] == 1


def test_separate_touching_grains_concave_gating_accepts_touching_pair():
    # Touching rice pair forming dumbbell within valid aspect ratio, solidity, and area bounds
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (38, 50), 14, 255, -1)
    cv2.circle(mask, (62, 50), 14, 255, -1)

    # If min_peak_distance_px is larger than peak separation (24px), watershed produces <= 1 peak,
    # and concave-neck fallback successfully takes over and splits the pair.
    inst_mask, diags, summary = separate_touching_grains(
        mask,
        min_peak_distance_px=30,
        cluster_area_thresh_px=650,
        enable_concave_fallback=True,
        concave_min_cluster_solidity=0.72,
        concave_max_cluster_aspect_ratio=3.8
    )
    assert summary["clusters_split_by_concave"] == 1
    assert summary["total_detected_grains"] == 2



