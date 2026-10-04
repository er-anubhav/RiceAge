"""Tests for touching-grain separation, distance transform, and watershed."""

import pytest
import numpy as np
import cv2

from src.separation.connected_components import extract_connected_components, summarize_connected_components
from src.separation.distance_transform import compute_distance_transform, detect_distance_peaks
from src.separation.cluster_classifier import evaluate_component_geometry, ComponentCategory
from src.separation.watershed import separate_touching_grains


def test_connected_components():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (30, 30), 10, 255, -1)
    cv2.circle(mask, (70, 70), 10, 255, -1)

    num_labels, labels, stats, centroids = extract_connected_components(mask)
    assert num_labels == 3  # bg + 2 objects
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
    # Create two touching ellipses simulating rice grains
    mask = np.zeros((120, 120), dtype=np.uint8)
    # Grain 1: center at (45, 60), axes (22, 10), horizontal
    cv2.ellipse(mask, (45, 60), (22, 10), 0, 0, 360, 255, -1)
    # Grain 2: center at (75, 60), axes (22, 10), horizontal (touching at x=60)
    cv2.ellipse(mask, (75, 60), (22, 10), 0, 0, 360, 255, -1)

    # In raw CC, they form 1 single connected component!
    num_cc, _, _, _ = cv2.connectedComponentsWithStats(mask)
    assert num_cc == 2  # bg + 1 cluster

    # Now separate using marker-controlled watershed
    inst_mask, diags, stats = separate_touching_grains(
        mask,
        min_peak_distance_px=6,
        cluster_area_thresh_px=300,
        solidity_thresh=0.95
    )

    # Should separate into 2 grains!
    unique_ids = np.unique(inst_mask)
    grains = unique_ids[unique_ids > 0]
    assert len(grains) == 2
    assert stats["clusters_successfully_split"] == 1


def test_isolated_single_grain_not_oversegmented():
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Single clean grain
    cv2.ellipse(mask, (50, 50), (20, 9), 30, 0, 360, 255, -1)

    inst_mask, diags, stats = separate_touching_grains(
        mask,
        min_peak_distance_px=6,
        cluster_area_thresh_px=600,
        solidity_thresh=0.90
    )

    unique_ids = np.unique(inst_mask)
    grains = unique_ids[unique_ids > 0]
    # Must remain exactly 1 grain!
    assert len(grains) == 1
    assert diags[grains[0]]["status"] == "isolated_single"


def test_empty_mask_separation():
    empty = np.zeros((50, 50), dtype=np.uint8)
    inst_mask, diags, stats = separate_touching_grains(empty)
    assert inst_mask.max() == 0
    assert len(diags) == 0
    assert stats["total_detected_grains"] == 0
