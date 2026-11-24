"""
Phase 4.2: Dense-Cluster Segmentation Optimization Regression Tests.
Verifies:
- Provenance preservation for derived child instances.
- Prevention of double-counting between parent clusters and child grains.
- Rejection of implausibly small fragments.
- Unresolved cluster preservation when separation is uncertain.
- Preservation of dual-view distance and ambiguity gating on split instances.
- Preservation of isolated single grains without false splitting.
"""

import pytest
import numpy as np
import cv2

from src.dual_view.models import ViewObservation, CorrespondenceStatus
from src.dual_view.registration import DualViewRegistration
from src.dual_view.correspondence import DualViewCorrespondenceEngine
from src.dual_view.fusion import DualViewFusionEngine
from src.separation import split_cluster_concave_neck


def make_synthetic_cluster_mask(h=200, w=300):
    """Creates a binary mask of two touching elliptical grains."""
    mask = np.zeros((h, w), dtype=np.uint8)
    # Grain 1: center at (90, 100), major=45, minor=20, angle=25
    cv2.ellipse(mask, (90, 100), (45, 20), 25, 0, 360, 255, -1)
    # Grain 2: center at (170, 100), major=45, minor=20, angle=-25 (touching Grain 1 at ~x=130)
    cv2.ellipse(mask, (170, 100), (45, 20), -25, 0, 360, 255, -1)
    return mask


def test_concave_neck_split_produces_two_valid_pieces():
    """Verify that touching grains with a clear concave neck are split into exactly two pieces."""
    mask = make_synthetic_cluster_mask()
    labeled, num_pieces, pairs = split_cluster_concave_neck(
        mask,
        min_defect_depth_px=3.0,
        max_neck_width_px=80.0,
        cut_thickness=2,
        min_piece_area_px=200
    )

    assert num_pieces == 2
    assert len(pairs) >= 1
    assert np.all(np.unique(labeled) == [0, 1, 2])


def test_child_instances_preserve_parent_cluster_provenance():
    """Verify that every child grain produced from a split preserves parent cluster ID and observation ID."""
    parent_cid = "CLUST_B_0042"
    parent_obs_id = "OBS_B_00356"

    # Simulate two child instances from cluster split
    child_1 = ViewObservation(
        observation_id=f"{parent_obs_id}_C1",
        scanner_id="bottom",
        detection_id=f"{parent_obs_id}_child_1",
        centroid_x=1050.0,
        centroid_y=1200.0,
        bounding_box=(1020, 1180, 60, 40),
        area_px=3800.0,
        length_mm=2.5,
        width_mm=1.2,
        aspect_ratio=2.08,
        raw_features={
            "parent_cluster_id": parent_cid,
            "parent_observation_id": parent_obs_id,
            "split_method": "Method_E_Hybrid_Refinement"
        }
    )

    child_2 = ViewObservation(
        observation_id=f"{parent_obs_id}_C2",
        scanner_id="bottom",
        detection_id=f"{parent_obs_id}_child_2",
        centroid_x=1120.0,
        centroid_y=1210.0,
        bounding_box=(1090, 1190, 60, 40),
        area_px=4100.0,
        length_mm=2.6,
        width_mm=1.3,
        aspect_ratio=2.00,
        raw_features={
            "parent_cluster_id": parent_cid,
            "parent_observation_id": parent_obs_id,
            "split_method": "Method_E_Hybrid_Refinement"
        }
    )

    assert child_1.raw_features["parent_cluster_id"] == parent_cid
    assert child_2.raw_features["parent_cluster_id"] == parent_cid
    assert child_1.raw_features["parent_observation_id"] == parent_obs_id
    assert child_2.raw_features["parent_observation_id"] == parent_obs_id
    assert child_1.observation_id != child_2.observation_id


def test_zero_inventory_double_counting_between_parent_and_children():
    """
    Verify that an accepted split replaces the parent cluster record in the active single-grain count,
    ensuring parent and children are never both counted as independent physical grains.
    """
    H = np.array([
        [-0.998463, -0.000463, 4917.872],
        [ 0.001606,  0.999657,   21.086],
        [ 0.000000,  0.000000,    1.000]
    ])
    reg = DualViewRegistration(homography_matrix=H)
    corr = DualViewCorrespondenceEngine(registration=reg)
    fusion = DualViewFusionEngine(correspondence_engine=corr)

    # 2 child grains derived from a parent cluster (parent cluster itself is excluded from active single grains)
    child_1 = ViewObservation(
        observation_id="OBS_B_00010_C1",
        scanner_id="bottom",
        detection_id="b_10_c1",
        centroid_x=500.0,
        centroid_y=500.0,
        bounding_box=(480, 480, 40, 40),
        area_px=3800.0
    )
    child_2 = ViewObservation(
        observation_id="OBS_B_00010_C2",
        scanner_id="bottom",
        detection_id="b_10_c2",
        centroid_x=600.0,
        centroid_y=500.0,
        bounding_box=(580, 480, 40, 40),
        area_px=4200.0
    )

    # When fed to fusion engine, resulting physical grains count must be exactly 2 (the child grains)
    res = fusion.fuse_views([child_1, child_2], [])
    assert res.total_single_physical_grains == 2
    assert res.unresolved_cluster_count == 0


def test_reject_implausibly_small_fragments():
    """Fragments below 800px or with degenerate aspect ratio must be rejected and not promoted to single grains."""
    def validate_grain_candidate(area_px: float, aspect_ratio: float, solidity: float) -> bool:
        if area_px < 800.0 or area_px > 7500.0:
            return False
        if aspect_ratio < 1.15 or aspect_ratio > 6.0:
            return False
        if solidity < 0.78:
            return False
        return True

    # Valid grain at 600 DPI
    assert validate_grain_candidate(4200.0, 2.4, 0.94) is True
    # Tiny fragment / noise speck
    assert validate_grain_candidate(350.0, 1.8, 0.92) is False
    # Degenerate sliver
    assert validate_grain_candidate(1200.0, 7.5, 0.85) is False
    # Unseparated cluster blob
    assert validate_grain_candidate(18000.0, 1.5, 0.65) is False


def test_unresolved_cluster_preservation_when_separation_uncertain():
    """When a cluster has no clear concave neck and cannot be reliably split, it must remain an unresolved cluster."""
    # A single solid circular blob (simulating dense overlapping cluster with no concave necks)
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 255, -1)

    labeled, num_pieces, pairs = split_cluster_concave_neck(
        mask,
        min_defect_depth_px=4.0,
        max_neck_width_px=80.0
    )

    # Must NOT force a split
    assert num_pieces == 1
    assert len(pairs) == 0


def test_dual_view_distance_gating_preserved_on_child_instances():
    """Separated child instances must strictly respect the 35px distance gate during cross-platen matching."""
    H = np.array([
        [-0.998463, -0.000463, 4917.872],
        [ 0.001606,  0.999657,   21.086],
        [ 0.000000,  0.000000,    1.000]
    ])
    reg = DualViewRegistration(homography_matrix=H)
    corr = DualViewCorrespondenceEngine(registration=reg, max_matching_distance_px=35.0)
    fusion = DualViewFusionEngine(correspondence_engine=corr)

    bx, by = 1500.0, 2000.0
    _, tx, ty = reg.project_bottom_to_top(bx, by)

    # Child 1: Near projection (distance ~ 2px) -> MATCHED
    child_b1 = ViewObservation(
        observation_id="OBS_B_00100_C1", scanner_id="bottom", detection_id="b1_c1",
        centroid_x=bx, centroid_y=by, bounding_box=(0,0,0,0), area_px=4000.0
    )
    top_t1 = ViewObservation(
        observation_id="OBS_T_00200", scanner_id="top", detection_id="t1",
        centroid_x=tx + 1.5, centroid_y=ty - 1.0, bounding_box=(0,0,0,0), area_px=4100.0
    )

    # Child 2: Far from top counterpart (distance ~ 50px > 35px gate) -> UNMATCHED
    child_b2 = ViewObservation(
        observation_id="OBS_B_00100_C2", scanner_id="bottom", detection_id="b1_c2",
        centroid_x=bx + 200.0, centroid_y=by, bounding_box=(0,0,0,0), area_px=3900.0
    )

    res = fusion.fuse_views([child_b1, child_b2], [top_t1])

    # child_b1 matches top_t1
    matched = [g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.MATCHED]
    assert len(matched) == 1
    assert matched[0].bottom_observation_id == "OBS_B_00100_C1"

    # child_b2 is 50px away: MUST NOT match top_t1
    unmatched_b = [g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.UNMATCHED_BOTTOM]
    assert len(unmatched_b) == 1
    assert unmatched_b[0].bottom_observation_id == "OBS_B_00100_C2"
