"""
RiceAge Phase 4 Dual-View Integration Prototype Regression Tests.
Verifies all 10 core regression invariants, plus reproduction of the Phase 3.2 b_859 bug:
 1. A grain observed by both scanners produces exactly one physical grain record.
 2. Total physical grain count does not double after adding second view.
 3. A bottom detection cannot be assigned to two top detections.
 4. A top detection cannot be assigned to two bottom detections.
 5. Missing candidates do not trigger forced assignments.
 6. Ambiguous matches do not enter automatic fusion.
 7. Registration failure triggers safe fallback.
 8. Existing single-view behavior remains operational.
 9. Sample-level aggregation counts each physical grain once.
10. Unmatched observations remain traceable.
11. Regression test: Phase 3.2 dropped detection (b_859 / t_981) duplicate assignment prevented.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.dual_view.models import (
    ViewObservation,
    PhysicalGrain,
    CorrespondenceStatus,
    DualViewFusionResult
)
from src.dual_view.registration import DualViewRegistration
from src.dual_view.correspondence import DualViewCorrespondenceEngine
from src.dual_view.fusion import DualViewFusionEngine, dataframe_to_view_observations
from src.hardware.dual_platen import DualPlatenManager
from configs.default_config import PipelineConfig


@pytest.fixture
def calibrated_registration():
    """Provides a valid calibrated registration instance."""
    H = np.array([
        [-0.998463, -0.000463, 4917.872],
        [ 0.001606,  0.999657,   21.086],
        [ 0.000000,  0.000000,    1.000]
    ], dtype=np.float64)
    return DualViewRegistration(homography_matrix=H)


@pytest.fixture
def correspondence_engine(calibrated_registration):
    return DualViewCorrespondenceEngine(registration=calibrated_registration, max_matching_distance_px=35.0, ambiguity_margin_px=8.0)


@pytest.fixture
def fusion_engine(correspondence_engine):
    # Preserves Phase 4/4.1 experimental canonical fusion feature flag for backward compatibility testing
    return DualViewFusionEngine(
        correspondence_engine=correspondence_engine,
        pixel_scale_mm=0.08467,
        architecture_mode="experimental_canonical_fusion"
    )


def make_obs(
    scanner_id: str,
    idx: int,
    cx: float,
    cy: float,
    area: float = 600.0,
    dclass: str = "Good / Head Rice",
    ar: float = 2.4,
    dconf: float = 0.95,
    sep_status: str = "single_grain"
):
    prefix = scanner_id[0].upper()
    return ViewObservation(
        observation_id=f"OBS_{prefix}_{idx:05d}",
        scanner_id=scanner_id,
        detection_id=f"{scanner_id[0]}_{idx}",
        centroid_x=cx,
        centroid_y=cy,
        bounding_box=(int(cx - 15), int(cy - 6), 30, 12),
        area_px=area,
        length_mm=round(30 * 0.08467, 2),
        width_mm=round(12 * 0.08467, 2),
        aspect_ratio=ar,
        defect_class=dclass,
        defect_confidence=dconf,
        raw_features={"separation_status": sep_status}
    )


# Test 1: Single physical grain produced for dual-view observation
def test_grain_observed_by_both_scanners_produces_single_physical_grain(calibrated_registration, fusion_engine):
    bx, by = 500.0, 500.0
    _, tx_exp, ty_exp = calibrated_registration.project_bottom_to_top(bx, by)

    b_obs = [make_obs("bottom", 1, bx, by, area=620.0, dclass="Good / Head Rice")]
    t_obs = [make_obs("top", 1, tx_exp + 1.2, ty_exp - 0.8, area=610.0, dclass="Good / Head Rice")]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.total_physical_grains == 1
    assert res.matched_count == 1
    g = res.physical_grains[0]
    assert g.correspondence_status == CorrespondenceStatus.MATCHED
    assert g.bottom_observation_id == "OBS_B_00001"
    assert g.top_observation_id == "OBS_T_00001"
    assert g.quality_attributes["is_fused"] is True


# Test 2: Total physical grain count does NOT double after adding second view
def test_total_physical_grain_count_does_not_double(calibrated_registration, fusion_engine):
    N = 5
    b_obs = []
    t_obs = []

    for i in range(N):
        bx = 400.0 + i * 150.0
        by = 600.0 + i * 100.0
        _, tx, ty = calibrated_registration.project_bottom_to_top(bx, by)
        b_obs.append(make_obs("bottom", i, bx, by, area=650.0))
        t_obs.append(make_obs("top", i, tx, ty, area=640.0))

    res = fusion_engine.fuse_views(b_obs, t_obs)

    # Must be 5 physical grains, NOT 10!
    assert res.total_physical_grains == N
    assert res.matched_count == N
    assert res.unmatched_bottom_count == 0
    assert res.unmatched_top_count == 0
    # Mass must be approximately 5 grains worth, NOT 10!
    single_grain_mass = res.physical_grains[0].quality_attributes["estimated_mass_g"]
    assert abs(res.estimated_mass_g - (N * single_grain_mass)) < 1e-4


# Test 3: Bottom detection cannot map to two top detections
def test_bottom_cannot_map_to_two_top_detections(calibrated_registration, fusion_engine):
    bx, by = 600.0, 700.0
    _, tx, ty = calibrated_registration.project_bottom_to_top(bx, by)

    b_obs = [make_obs("bottom", 1, bx, by, area=600.0)]
    # Two top detections: one close (dist ~3px), one further (dist ~25px)
    t_obs = [
        make_obs("top", 1, tx + 2.0, ty + 2.0, area=605.0),
        make_obs("top", 2, tx + 25.0, ty, area=610.0)
    ]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.total_physical_grains == 2
    assert res.matched_count == 1
    assert res.unmatched_top_count == 1

    matched = [g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.MATCHED]
    assert len(matched) == 1
    assert matched[0].top_observation_id == "OBS_T_00001"


# Test 4: Top detection cannot map to two bottom detections
def test_top_cannot_map_to_two_bottom_detections(calibrated_registration, fusion_engine):
    bx1, by1 = 800.0, 800.0
    bx2, by2 = 820.0, 800.0  # 20px away
    _, tx1, ty1 = calibrated_registration.project_bottom_to_top(bx1, by1)

    b_obs = [
        make_obs("bottom", 1, bx1, by1, area=600.0),
        make_obs("bottom", 2, bx2, by2, area=600.0)
    ]
    # Single top detection right at projection of b_1
    t_obs = [make_obs("top", 1, tx1, ty1, area=600.0)]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.total_physical_grains == 2
    assert res.matched_count == 1
    assert res.unmatched_bottom_count == 1

    matched = [g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.MATCHED]
    assert matched[0].bottom_observation_id == "OBS_B_00001"


# Test 5: Missing candidates do NOT trigger forced assignments
def test_missing_candidates_do_not_trigger_forced_assignments(fusion_engine):
    b_obs = [make_obs("bottom", 1, 500.0, 500.0)]
    # Far-away top observation (over 1000px away)
    t_obs = [make_obs("top", 1, 2000.0, 2000.0)]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.matched_count == 0
    assert res.unmatched_bottom_count == 1
    assert res.unmatched_top_count == 1
    assert res.total_physical_grains == 2


# Test 6: Ambiguous matches do not enter automatic fusion
def test_ambiguous_matches_do_not_enter_automatic_fusion(calibrated_registration, fusion_engine):
    bx, by = 1000.0, 1000.0
    _, tx, ty = calibrated_registration.project_bottom_to_top(bx, by)

    b_obs = [make_obs("bottom", 1, bx, by, area=600.0)]
    # Two top detections equidistant (e.g. dist ~15px and dist ~16px, diff < 8px)
    t_obs = [
        make_obs("top", 1, tx + 15.0, ty, area=600.0),
        make_obs("top", 2, tx - 15.0, ty, area=600.0)
    ]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.matched_count == 0
    assert res.ambiguous_count > 0
    # Ambiguous records must NOT have automatic fusion applied
    for g in res.physical_grains:
        if g.correspondence_status == CorrespondenceStatus.AMBIGUOUS:
            assert g.quality_attributes["is_fused"] is False


# Test 7: Registration failure triggers safe fail-closed fallback
def test_registration_failure_triggers_safe_fallback():
    # Corrupted matrix with positive determinant (+1.0 -> no reflection)
    H_bad = np.array([
        [ 1.0, 0.0, 100.0],
        [ 0.0, 1.0, 100.0],
        [ 0.0, 0.0,   1.0]
    ])
    bad_reg = DualViewRegistration(homography_matrix=H_bad)
    assert bad_reg.is_healthy is False

    engine = DualViewCorrespondenceEngine(registration=bad_reg)
    fusion = DualViewFusionEngine(correspondence_engine=engine, architecture_mode="experimental_canonical_fusion")

    b_obs = [make_obs("bottom", 1, 500.0, 500.0)]
    t_obs = [make_obs("top", 1, 500.0, 500.0)]

    res = fusion.fuse_views(b_obs, t_obs)

    assert res.registration_failed_count == 2
    assert res.matched_count == 0
    for g in res.physical_grains:
        assert g.correspondence_status == CorrespondenceStatus.REGISTRATION_FAILED
        assert g.quality_attributes["is_fused"] is False


# Test 8: Existing single-view / legacy behavior remains operational
def test_existing_single_view_behavior_remains_operational(tmp_path):
    p1 = str(tmp_path / "p1.jpg")
    p2 = str(tmp_path / "p2.jpg")

    import cv2
    img1 = np.full((400, 400, 3), 40, dtype=np.uint8)
    cv2.ellipse(img1, (200, 200), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.imwrite(p1, img1)

    img2 = np.full((400, 400, 3), 40, dtype=np.uint8)
    cv2.ellipse(img2, (200, 200), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.imwrite(p2, img2)

    from src.pipeline import RiceQualityPipeline
    cfg = PipelineConfig()
    cfg.segmentation.engine = "classic"
    pipeline = RiceQualityPipeline(config=cfg)

    # Legacy mode: use_dual_view_fusion=False
    mgr_legacy = DualPlatenManager(pipeline=pipeline, use_dual_view_fusion=False)
    res_legacy = mgr_legacy.run_dual_analysis(p1, p2, sample_id="TEST-LEGACY")

    assert res_legacy["status"] == "Complete"
    assert "is_dual_view_prototype" not in res_legacy or res_legacy.get("is_dual_view_prototype") is not True

    # Prototype mode: use_dual_view_fusion=True
    mgr_proto = DualPlatenManager(pipeline=pipeline, use_dual_view_fusion=True)
    res_proto = mgr_proto.run_dual_analysis(p1, p2, sample_id="TEST-PROTO")

    assert res_proto["status"] == "Complete"
    assert res_proto["is_dual_view_prototype"] is True
    assert "physical_grains" in res_proto
    assert "dual_view_diagnostics" in res_proto


# Test 9: Sample-level aggregation counts each physical grain once
def test_sample_level_aggregation_counts_each_physical_grain_once(calibrated_registration, fusion_engine):
    b_obs = [
        make_obs("bottom", 1, 500.0, 500.0, area=600.0, dclass="Good / Head Rice"),
        make_obs("bottom", 2, 700.0, 500.0, area=650.0, dclass="Chalky Grains"),
        make_obs("bottom", 3, 900.0, 500.0, area=580.0, dclass="Big Brokens (25-75%)")
    ]
    # Top has counterpart for grain 1 only
    _, tx1, ty1 = calibrated_registration.project_bottom_to_top(500.0, 500.0)
    t_obs = [
        make_obs("top", 1, tx1, ty1, area=610.0, dclass="Good / Head Rice"),
        make_obs("top", 99, 100.0, 100.0, area=500.0, dclass="Damaged Grains")  # Unmatched top
    ]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    # 1 matched + 2 unmatched bottom + 1 unmatched top = 4 physical grains
    assert res.total_physical_grains == 4
    cb = res.aggregation_summary["count_basis"]
    assert cb["total_physical_grains"] == 4
    assert cb["whole_count"] == 1
    assert cb["chalky_count"] == 1
    assert cb["broken_count"] == 1
    assert cb["damaged_count"] == 1
    total_cat_sum = (cb["whole_count"] + cb["chalky_count"] + cb["broken_count"] +
                     cb["damaged_count"] + cb["discoloured_count"] + cb["small_broken_count"] +
                     cb["foreign_matter_count"] + cb["unresolved_count"])
    assert total_cat_sum == 4


# Test 10: Unmatched observations remain traceable
def test_unmatched_observations_remain_traceable(fusion_engine):
    b_obs = [make_obs("bottom", 42, 500.0, 600.0, area=612.0)]
    t_obs = [make_obs("top", 99, 2000.0, 2000.0, area=580.0)]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.total_physical_grains == 2
    b_grain = next(g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.UNMATCHED_BOTTOM)
    assert b_grain.bottom_obs is not None
    assert b_grain.bottom_obs.detection_id == "b_42"
    assert b_grain.bottom_obs.centroid_x == 500.0

    t_grain = next(g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.UNMATCHED_TOP)
    assert t_grain.top_obs is not None
    assert t_grain.top_obs.detection_id == "t_99"
    assert t_grain.top_obs.centroid_x == 2000.0


# Test 11: Phase 3.2 Regression - Dropped sub-400px detection cannot cause distant b_859 double-assignment
def test_reproduce_phase3_2_dropped_detection_b859_bug(calibrated_registration, fusion_engine):
    """
    In Phase 3.2, grain CLEAN_GRAIN_140 was dropped from detection because area was 390px (<400px threshold).
    A naive argmin() found b_859 199.1px away, which was already the true detection for CLEAN_GRAIN_138.
    This caused b_859 to be double-assigned and generated a false positive mismatch.
    This regression test proves that our architecture prevents distant hijacking:
    Distance gating strictly rejects any candidate > 35px, and bipartite matching guarantees
    b_859 can NEVER be assigned to two records.
    """
    # Detection b_859 at (518.4, 31.4)
    b_859 = make_obs("bottom", 859, 518.44, 31.40, area=770.0)
    b_obs = [b_859]

    # Two queries:
    # 1. True top observation t_981 at (4400.2, 53.3) -> distance to b_859 proj is ~2.0px (VALID MATCH)
    _, tx_859, ty_859 = calibrated_registration.project_bottom_to_top(518.44, 31.40)
    t_981 = make_obs("top", 981, tx_859 + 1.5, ty_859 - 1.0, area=760.0)

    # 2. Second top observation t_distant at (4265.9, 200.4) -> distance to b_859 proj is ~199px away!
    t_distant = make_obs("top", 999, 4265.94, 200.39, area=390.0)

    t_obs = [t_981, t_distant]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    # b_859 must match t_981
    matched = [g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.MATCHED]
    assert len(matched) == 1
    assert matched[0].bottom_observation_id == "OBS_B_00859"
    assert matched[0].top_observation_id == "OBS_T_00981"

    # t_distant is 199px away: must NEVER be forced or hijack b_859!
    unmatched_t = [g for g in res.physical_grains if g.correspondence_status == CorrespondenceStatus.UNMATCHED_TOP]
    assert len(unmatched_t) == 1
    assert unmatched_t[0].top_observation_id == "OBS_T_00999"

    # Total physical grains = 1 matched + 1 unmatched top = 2
    assert res.total_physical_grains == 2


# ============================================================================
# Phase 4.1 Prototype Hardening Regression Tests
# ============================================================================

# Test 12: Every scanner observation has exactly one disposition
def test_every_scanner_observation_has_exactly_one_disposition(calibrated_registration, fusion_engine):
    """
    Every input observation (bottom and top) must have exactly one documented disposition
    in the final accounting, with no omissions or orphan observations.
    """
    bx1, by1 = 500.0, 500.0
    _, tx1, ty1 = calibrated_registration.project_bottom_to_top(bx1, by1)
    bx2, by2 = 800.0, 800.0
    _, tx2, ty2 = calibrated_registration.project_bottom_to_top(bx2, by2)

    b_obs = [
        make_obs("bottom", 1, bx1, by1, area=600.0),
        make_obs("bottom", 2, 200.0, 200.0, area=600.0),
        make_obs("bottom", 3, bx2, by2, area=600.0),
    ]
    t_obs = [
        make_obs("top", 1, tx1, ty1, area=600.0),
        make_obs("top", 2, 4000.0, 4000.0, area=600.0),
        make_obs("top", 3, tx2 + 10.0, ty2, area=600.0),
        make_obs("top", 4, tx2 - 10.0, ty2, area=600.0),
    ]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    accounted_b = set()
    accounted_t = set()

    for g in res.physical_grains:
        if g.bottom_observation_id:
            assert g.bottom_observation_id not in accounted_b, f"Duplicate bottom observation {g.bottom_observation_id}"
            accounted_b.add(g.bottom_observation_id)
        if g.top_observation_id:
            assert g.top_observation_id not in accounted_t, f"Duplicate top observation {g.top_observation_id}"
            accounted_t.add(g.top_observation_id)

    input_b_ids = {o.observation_id for o in b_obs}
    input_t_ids = {o.observation_id for o in t_obs}

    assert accounted_b == input_b_ids
    assert accounted_t == input_t_ids


# Test 13: Diagnostic categories do not double-count observations
def test_diagnostic_categories_do_not_double_count_observations(calibrated_registration, fusion_engine):
    """
    Ensure all diagnostic categories (MATCHED, UNMATCHED, AMBIGUOUS, UNRESOLVED_CLUSTER)
    form a strict partition of the input observation set.
    """
    b1 = make_obs("bottom", 1, 1000.0, 1000.0, area=600.0)
    _, tx1, ty1 = calibrated_registration.project_bottom_to_top(1000.0, 1000.0)
    t1 = make_obs("top", 1, tx1, ty1, area=600.0)

    b2 = make_obs("bottom", 2, 1200.0, 1000.0, area=600.0)
    t2 = make_obs("top", 2, 3000.0, 3000.0, area=600.0)

    b_clust = make_obs("bottom", 3, 1500.0, 1500.0, area=12000.0, sep_status="under_segmented_cluster")
    t_clust = make_obs("top", 3, 3500.0, 3500.0, area=12000.0, sep_status="under_segmented_cluster")

    b_obs = [b1, b2, b_clust]
    t_obs = [t1, t2, t_clust]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.matched_count == 1
    assert res.unmatched_bottom_count == 1
    assert res.unmatched_top_count == 1
    assert res.unresolved_cluster_count == 2
    assert len(b_obs) == 3
    assert len(t_obs) == 3


# Test 14: Canonical grain totals reconcile independently
def test_canonical_grain_totals_reconcile_independently(calibrated_registration, fusion_engine):
    """
    Reconcile:
    N_single_physical = N_matched + N_unmatched_b + N_unmatched_t + N_ambiguous
    N_total_inventory = N_single_physical + N_unresolved_clusters
    """
    b1 = make_obs("bottom", 1, 1000.0, 1000.0)
    _, tx1, ty1 = calibrated_registration.project_bottom_to_top(1000.0, 1000.0)
    t1 = make_obs("top", 1, tx1, ty1)

    b2 = make_obs("bottom", 2, 200.0, 200.0)
    t2 = make_obs("top", 2, 4500.0, 4500.0)

    b_c = make_obs("bottom", 3, 500.0, 500.0, area=10000.0, sep_status="under_segmented_cluster")

    b_obs = [b1, b2, b_c]
    t_obs = [t1, t2]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    expected_single = res.matched_count + res.unmatched_bottom_count + res.unmatched_top_count + res.ambiguous_count
    assert res.total_single_physical_grains == expected_single
    assert res.total_inventory_records == expected_single + res.unresolved_cluster_count


# Test 15: Unresolved clusters are not fabricated into individual grains
def test_unresolved_clusters_not_fabricated_into_individual_grains(fusion_engine):
    """
    Clusters must never be artificially decomposed into speculative grains without actual segmentation.
    They remain segregated as cluster inventory records.
    """
    cluster_obs = make_obs("bottom", 999, 1000.0, 1000.0, area=15000.0, sep_status="under_segmented_cluster")
    res = fusion_engine.fuse_views([cluster_obs], [])

    assert res.total_single_physical_grains == 0
    assert res.unresolved_cluster_count == 1
    assert res.total_inventory_records == 1
    cluster_grain = res.physical_grains[0]
    assert cluster_grain.correspondence_status == CorrespondenceStatus.SEGMENTATION_UNRESOLVED
    assert cluster_grain.quality_attributes["separation_status"] == "under_segmented_cluster"
    assert cluster_grain.quality_attributes["is_cluster"] is True


# Test 16: Feature flag disabled with two scanner inputs does not double-count
def test_feature_flag_disabled_does_not_double_count(tmp_path):
    """
    Regression proof: when dual-view fusion is disabled, the system selects safe single-view
    fallback and NEVER executes pd.concat across opposing platens of the same sample.
    """
    p1 = str(tmp_path / "p1.jpg")
    p2 = str(tmp_path / "p2.jpg")

    import cv2
    img1 = np.full((500, 500, 3), 40, dtype=np.uint8)
    cv2.ellipse(img1, (200, 200), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.ellipse(img1, (300, 200), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.imwrite(p1, img1)

    img2 = np.full((500, 500, 3), 40, dtype=np.uint8)
    cv2.ellipse(img2, (250, 250), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.imwrite(p2, img2)

    from src.pipeline import RiceQualityPipeline
    cfg = PipelineConfig()
    cfg.segmentation.engine = "classic"
    cfg.dual_view.single_view_fallback_scanner = "bottom"
    pipeline = RiceQualityPipeline(config=cfg)

    mgr = DualPlatenManager(pipeline=pipeline, use_dual_view_fusion=False)
    res = mgr.run_dual_analysis(p1, p2, sample_id="TEST-FALLBACK")

    assert res["status"] == "Complete"
    # Bottom has 2 grains, top has 1 grain. Safe single-view fallback selects 2.
    assert res["total_grains"] == 2
    assert res["is_sample_doubling_prevented"] is True
    assert res["fallback_mode"] == "SAFE_SINGLE_VIEW_SELECTION"


# Test 17: Mass is marked as NOT_PHYSICALLY_MEASURED and statutory_acceptance_usable is False
def test_mass_marked_unvalidated_and_not_physically_measured(calibrated_registration, fusion_engine):
    b_obs = [make_obs("bottom", 1, 500.0, 500.0, area=600.0)]
    t_obs = []

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert res.mass_measurement_status == "NOT_PHYSICALLY_MEASURED"
    assert res.is_physically_measured is False
    assert res.statutory_acceptance_usable is False
    assert "unvalidated" in res.statutory_mass_disclaimer.lower() or "uncalibrated" in res.statutory_mass_disclaimer.lower()


# Test 18: Unvalidated mass cannot drive statutory acceptance or PASS/FAIL disposition
def test_unvalidated_mass_cannot_drive_statutory_acceptance(calibrated_registration, fusion_engine):
    """
    Ensure the prototype's grading and commercial disposition are based exclusively
    on whole-grain count percentage and not on optical mass proxies.
    """
    b_obs = [
        make_obs("bottom", 1, 500.0, 500.0, dclass="Good / Head Rice"),
        make_obs("bottom", 2, 600.0, 500.0, dclass="Big Brokens (25-75%)"),
        make_obs("bottom", 3, 700.0, 500.0, dclass="Big Brokens (25-75%)"),
        make_obs("bottom", 4, 800.0, 500.0, dclass="Big Brokens (25-75%)"),
    ]
    res = fusion_engine.fuse_views(b_obs, [])

    assert res.statutory_acceptance_usable is False
    cb = res.aggregation_summary["count_basis"]
    assert cb["whole_pct"] == 25.0


# Test 19: Per-view defect predictions remain available after fusion
def test_per_view_defect_predictions_remain_available(calibrated_registration, fusion_engine):
    bx, by = 500.0, 500.0
    _, tx, ty = calibrated_registration.project_bottom_to_top(bx, by)

    b_obs = [make_obs("bottom", 1, bx, by, dclass="Good / Head Rice", dconf=0.92)]
    t_obs = [make_obs("top", 1, tx, ty, dclass="Chalky Grains", dconf=0.88)]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    assert len(res.physical_grains) == 1
    g = res.physical_grains[0]

    assert g.correspondence_status == CorrespondenceStatus.MATCHED
    assert g.bottom_predicted_class == "Good / Head Rice"
    assert g.bottom_confidence == 0.92
    assert g.top_predicted_class == "Chalky Grains"
    assert g.top_confidence == 0.88
    assert g.quality_attributes["defect_class"] == "Chalky Grains"


# Test 20: Fusion rule and version are recorded in quality_attributes
def test_fusion_rule_and_version_recorded_in_quality_attributes(calibrated_registration, fusion_engine):
    bx, by = 500.0, 500.0
    _, tx, ty = calibrated_registration.project_bottom_to_top(bx, by)

    b_obs = [make_obs("bottom", 1, bx, by, dclass="Good / Head Rice")]
    t_obs = [make_obs("top", 1, tx, ty, dclass="Damaged Grains")]

    res = fusion_engine.fuse_views(b_obs, t_obs)

    g = res.physical_grains[0]
    qa = g.quality_attributes

    assert "fusion_rule" in qa
    assert qa["fusion_rule"] == "surface_defect_precedence"
    assert "fusion_rule_version" in qa
    assert "fusion_status" in qa
    assert qa["fusion_status"] == "AUTOMATIC"
    assert "defect_precedence_is_standard_approved" in qa
    assert qa["defect_precedence_is_standard_approved"] is False
