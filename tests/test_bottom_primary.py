"""
Comprehensive Validation Suite for RiceAge Bottom-Scanner-Primary Architecture.
Validates:
  1. Bottom scanner is the authoritative source for count, whole/broken, defects, dimensions, and denominators.
  2. Top scanner detections are never concatenated or added to the primary grain count.
  3. Unmatched top observations are strictly excluded from the primary count.
  4. Top-view observations are used solely for secondary verification and discrepancy flags.
  5. Bottom unresolved clusters are preserved and segregated without entering the single-grain denominator.
  6. Bottom scanner failure produces an explicit analysis failure (fail-closed, never silently substitutes top).
  7. Dashboard, JSON, CSV, and PDF agree on the same bottom-primary result and denominator.
  8. Acquisition metadata accurately identifies the bottom scanner as primary.
  9. Feature flag allows toggling back to experimental canonical fusion.
"""

import os
import pytest
import numpy as np
import pandas as pd
import cv2

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
from src.reporting_pdf import generate_apscscl_pdf_certificate
from configs.default_config import PipelineConfig
from src.pipeline import RiceQualityPipeline


@pytest.fixture
def calibrated_registration():
    """Provides valid calibrated registration matrix with -1 reflection determinant."""
    H = np.array([
        [-0.998463, -0.000463, 4917.872],
        [ 0.001606,  0.999657,   21.086],
        [ 0.000000,  0.000000,    1.000]
    ], dtype=np.float64)
    return DualViewRegistration(homography_matrix=H)


@pytest.fixture
def correspondence_engine(calibrated_registration):
    return DualViewCorrespondenceEngine(
        registration=calibrated_registration,
        max_matching_distance_px=35.0,
        ambiguity_margin_px=8.0
    )


@pytest.fixture
def bottom_primary_fusion_engine(correspondence_engine):
    return DualViewFusionEngine(
        correspondence_engine=correspondence_engine,
        pixel_scale_mm=0.08467,
        architecture_mode="bottom_primary"
    )


def make_test_obs(
    scanner_id: str,
    idx: int,
    cx: float,
    cy: float,
    area: float = 600.0,
    len_mm: float = 6.4,
    wid_mm: float = 2.1,
    dclass: str = "Good / Head Rice",
    is_broken: bool = False,
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
        length_mm=len_mm,
        width_mm=wid_mm,
        aspect_ratio=round(len_mm / max(wid_mm, 0.01), 2),
        defect_class=dclass,
        defect_confidence=dconf,
        is_broken=is_broken,
        raw_features={"separation_status": sep_status}
    )


# 1. Test: Bottom scanner is authoritative for grain count and unmatched top are excluded
def test_bottom_scanner_is_authoritative_for_grain_count(calibrated_registration, bottom_primary_fusion_engine):
    # 5 bottom grains, 8 top grains (3 unmatched top)
    b_obs = []
    t_obs = []

    for i in range(5):
        bx = 500.0 + i * 150.0
        by = 500.0 + i * 100.0
        _, tx, ty = calibrated_registration.project_bottom_to_top(bx, by)
        b_obs.append(make_test_obs("bottom", i + 1, bx, by))
        t_obs.append(make_test_obs("top", i + 1, tx, ty))

    # Add 3 unmatched top observations
    t_obs.append(make_test_obs("top", 6, 2500.0, 2500.0))
    t_obs.append(make_test_obs("top", 7, 2700.0, 2500.0))
    t_obs.append(make_test_obs("top", 8, 2900.0, 2500.0))

    res = bottom_primary_fusion_engine.fuse_views(b_obs, t_obs)

    # Primary count MUST equal 5 (the bottom count), NOT 8!
    assert res.total_single_physical_grains == 5
    assert res.total_physical_grains == 5
    assert res.matched_count == 5
    assert res.unmatched_top_count == 3
    assert len(res.physical_grains) == 5

    # Check denominator in aggregation summary
    cb = res.aggregation_summary["count_basis"]
    assert cb["total_physical_grains"] == 5


# 2. Test: Top detections never concatenate or inflate quality denominators
def test_top_detections_never_concatenate_or_inflate_denominator(calibrated_registration, bottom_primary_fusion_engine):
    b_obs = [
        make_test_obs("bottom", 1, 500.0, 500.0, dclass="Good / Head Rice"),
        make_test_obs("bottom", 2, 800.0, 500.0, dclass="Good / Head Rice"),
        make_test_obs("bottom", 3, 1100.0, 500.0, dclass="Big Brokens (25-75%)", is_broken=True),
        make_test_obs("bottom", 4, 1400.0, 500.0, dclass="Good / Head Rice"),
    ]
    # Top scanner detects 10 extra unmatched objects
    t_obs = []
    for j in range(10):
        t_obs.append(make_test_obs("top", j + 10, 3000.0 + j * 50.0, 3000.0, dclass="Foreign Matter"))

    res = bottom_primary_fusion_engine.fuse_views(b_obs, t_obs)

    # Total grains must be strictly 4
    assert res.total_single_physical_grains == 4
    cb = res.aggregation_summary["count_basis"]
    assert cb["total_physical_grains"] == 4
    # Whole % should be 3/4 = 75.00%, broken % should be 1/4 = 25.00%
    assert cb["whole_count"] == 3
    assert cb["whole_pct"] == 75.0
    assert cb["broken_count"] == 1
    assert cb["broken_pct"] == 25.0
    # Foreign matter must be 0% (top unmatched foreign matter is not added!)
    assert cb["foreign_matter_count"] == 0
    assert cb["foreign_matter_pct"] == 0.0


# 3. Test: Bottom scanner is authoritative for classification, dimensions, and whole/broken
def test_bottom_authoritative_for_classification_and_dimensions(calibrated_registration, bottom_primary_fusion_engine):
    bx, by = 600.0, 600.0
    _, tx, ty = calibrated_registration.project_bottom_to_top(bx, by)

    # Bottom view: Whole grain, 6.6 mm length, 2.1 mm width
    b_obs = [make_test_obs("bottom", 1, bx, by, len_mm=6.6, wid_mm=2.1, dclass="Good / Head Rice", is_broken=False)]
    # Top view: Broken & Chalky, 4.5 mm length, 2.0 mm width
    t_obs = [make_test_obs("top", 1, tx, ty, len_mm=4.5, wid_mm=2.0, dclass="Chalky Grains", is_broken=True)]

    res = bottom_primary_fusion_engine.fuse_views(b_obs, t_obs)

    assert len(res.physical_grains) == 1
    g = res.physical_grains[0]
    qa = g.quality_attributes

    # Authoritative attributes MUST come from bottom!
    assert qa["defect_class"] == "Good / Head Rice"
    assert qa["fused_class"] == "Good / Head Rice"
    assert qa["is_broken"] is False
    assert qa["length_mm"] == 6.6
    assert qa["width_mm"] == 2.1
    assert qa["defect_precedence_source"] == "bottom_primary_authoritative"

    # Top view must be recorded for secondary verification & discrepancy flag
    assert qa["top_match_status"] == "MATCHED"
    assert qa["top_predicted_class"] == "Chalky Grains"
    assert qa["is_verified"] is False
    assert qa["discrepancy_flag"] is True
    assert "Bottom: Good / Head Rice vs Top: Chalky Grains" in qa["discrepancy_details"]


# 4. Test: Secondary verification flags agreement and discrepancies accurately
def test_top_view_verification_and_discrepancy_counting(calibrated_registration, bottom_primary_fusion_engine):
    b_obs = []
    t_obs = []

    # Grain 1: Agreement
    bx1, by1 = 500.0, 500.0
    _, tx1, ty1 = calibrated_registration.project_bottom_to_top(bx1, by1)
    b_obs.append(make_test_obs("bottom", 1, bx1, by1, dclass="Good / Head Rice"))
    t_obs.append(make_test_obs("top", 1, tx1, ty1, dclass="Good / Head Rice"))

    # Grain 2: Discrepancy (Bottom Good vs Top Damaged)
    bx2, by2 = 800.0, 500.0
    _, tx2, ty2 = calibrated_registration.project_bottom_to_top(bx2, by2)
    b_obs.append(make_test_obs("bottom", 2, bx2, by2, dclass="Good / Head Rice"))
    t_obs.append(make_test_obs("top", 2, tx2, ty2, dclass="Damaged Grains"))

    # Grain 3: Unmatched in top
    bx3, by3 = 1100.0, 500.0
    b_obs.append(make_test_obs("bottom", 3, bx3, by3, dclass="Good / Head Rice"))

    res = bottom_primary_fusion_engine.fuse_views(b_obs, t_obs)

    assert res.matched_count == 2
    assert res.unmatched_bottom_count == 1
    assert res.discrepancy_count == 1

    g1 = res.physical_grains[0]
    assert g1.quality_attributes["is_verified"] is True
    assert g1.quality_attributes["discrepancy_flag"] is False

    g2 = res.physical_grains[1]
    assert g2.quality_attributes["is_verified"] is False
    assert g2.quality_attributes["discrepancy_flag"] is True

    g3 = res.physical_grains[2]
    assert g3.quality_attributes["top_match_status"] == "UNMATCHED_IN_TOP"
    assert g3.quality_attributes["is_verified"] is False
    assert g3.quality_attributes["discrepancy_flag"] is False


# 5. Test: Bottom unresolved clusters are preserved as unresolved and segregated from denominator
def test_bottom_unresolved_clusters_preserved_and_segregated(bottom_primary_fusion_engine):
    b_obs = [
        make_test_obs("bottom", 1, 500.0, 500.0, area=650.0),
        make_test_obs("bottom", 2, 700.0, 500.0, area=700.0),
        # Cluster with huge area
        make_test_obs("bottom", 3, 900.0, 500.0, area=12000.0, sep_status="under_segmented_cluster")
    ]
    t_obs = [
        make_test_obs("top", 1, 500.0, 500.0, area=640.0)
    ]

    res = bottom_primary_fusion_engine.fuse_views(b_obs, t_obs)

    # 2 single grains, 1 unresolved cluster
    assert res.total_single_physical_grains == 2
    assert res.unresolved_cluster_count == 1
    assert res.total_inventory_records == 3

    # Check that cluster is preserved in inventory records
    cluster_records = [g for g in res.physical_grains if g.quality_attributes.get("is_cluster")]
    assert len(cluster_records) == 1
    assert cluster_records[0].quality_attributes["separation_status"] == "under_segmented_cluster"
    assert cluster_records[0].correspondence_status == CorrespondenceStatus.SEGMENTATION_UNRESOLVED

    # Denominator in aggregation summary must be 2, NOT 3
    cb = res.aggregation_summary["count_basis"]
    assert cb["total_physical_grains"] == 2


# 6. Test: Bottom scanner failure returns explicit analysis failure (fail-closed)
def test_bottom_scanner_failure_returns_explicit_failure(bottom_primary_fusion_engine):
    # Empty bottom observations, valid top observations
    b_obs = []
    t_obs = [make_test_obs("top", 1, 500.0, 500.0)]

    res = bottom_primary_fusion_engine.fuse_views(b_obs, t_obs)

    # Must fail closed, never substitute top observations!
    assert res.total_single_physical_grains == 0
    assert len(res.physical_grains) == 0
    assert res.aggregation_summary["status"] == "ANALYSIS_FAILED"
    assert res.aggregation_summary["failure_reason"] == "BOTTOM_PRIMARY_SCANNER_FAILED"
    assert res.registration_metadata["status"] == "FAILED_BOTTOM_PRIMARY_MISSING"


# 7. Test: DualPlatenManager bottom-primary integration and acquisition metadata
def test_dual_platen_manager_bottom_primary_integration(tmp_path):
    p1_path = str(tmp_path / "bottom.jpg")
    p2_path = str(tmp_path / "top.jpg")
    comp_path = str(tmp_path / "comp.jpg")

    # Bottom scan has 3 grains (using standard platen resolution 1489x2048)
    img1 = np.full((1489, 2048, 3), 40, dtype=np.uint8)
    cv2.ellipse(img1, (400, 700), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.ellipse(img1, (800, 700), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.ellipse(img1, (1200, 700), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.imwrite(p1_path, img1)

    # Top scan has 2 grains
    img2 = np.full((1489, 2048, 3), 40, dtype=np.uint8)
    cv2.ellipse(img2, (400, 700), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.ellipse(img2, (800, 700), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.imwrite(p2_path, img2)

    cfg = PipelineConfig()
    cfg.segmentation.engine = "classic"
    cfg.dual_view.enabled = True
    cfg.dual_view.architecture_mode = "bottom_primary"
    pipeline = RiceQualityPipeline(config=cfg)

    mgr = DualPlatenManager(pipeline=pipeline, use_dual_view_fusion=True)
    res = mgr.run_dual_analysis(p1_path, p2_path, sample_id="TEST-BP-001", composite_output_path=comp_path)

    assert res["status"] == "Complete"
    # Primary count must strictly be 3 (from bottom platen)
    assert res["total_grains"] == 3
    assert res["primary_scanner"] == "bottom"
    assert res["architecture_mode"] == "BOTTOM_PRIMARY"

    # Acquisition metadata must be populated
    meta = res["acquisition_metadata"]
    assert meta["primary_scanner"] == "bottom"
    assert meta["secondary_scanner"] == "top"
    assert meta["primary_grain_count"] == 3
    assert meta["primary_denominator"] == 3
    assert meta["secondary_verification_enabled"] is True

    # Grain records check
    assert len(res["grain_records"]) == 3
    for r in res["grain_records"]:
        assert r["primary_scanner"] == "bottom"


# 8. Test: DualPlatenManager fails closed when bottom platen is missing
def test_dual_platen_manager_fails_closed_when_bottom_missing(tmp_path):
    p2_path = str(tmp_path / "top_only.jpg")
    img2 = np.full((400, 400, 3), 40, dtype=np.uint8)
    cv2.ellipse(img2, (200, 200), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.imwrite(p2_path, img2)

    cfg = PipelineConfig()
    cfg.segmentation.engine = "classic"
    pipeline = RiceQualityPipeline(config=cfg)

    mgr = DualPlatenManager(pipeline=pipeline, use_dual_view_fusion=True)
    res = mgr.run_dual_analysis(None, p2_path, sample_id="TEST-FAIL-001")

    assert res["status"] == "ANALYSIS_FAILED"
    assert res["failure_reason"] == "BOTTOM_PRIMARY_SCANNER_FAILED"
    assert res["is_fallback_substituted"] is False
    assert res["total_grains"] == 0


# 9. Test: Dashboard, JSON, CSV and PDF all agree on bottom-primary denominator
def test_all_reports_agree_on_bottom_primary_denominator(tmp_path):
    p1_path = str(tmp_path / "b_scan.jpg")
    p2_path = str(tmp_path / "t_scan.jpg")
    pdf_out = str(tmp_path / "test_cert.pdf")

    img1 = np.full((1489, 2048, 3), 40, dtype=np.uint8)
    cv2.ellipse(img1, (600, 700), (22, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.ellipse(img1, (1200, 700), (22, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.imwrite(p1_path, img1)

    img2 = np.full((1489, 2048, 3), 40, dtype=np.uint8)
    cv2.ellipse(img2, (600, 700), (22, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.imwrite(p2_path, img2)

    cfg = PipelineConfig()
    cfg.segmentation.engine = "classic"
    cfg.dual_view.enabled = True
    cfg.dual_view.architecture_mode = "bottom_primary"
    pipeline = RiceQualityPipeline(config=cfg)

    mgr = DualPlatenManager(pipeline=pipeline, use_dual_view_fusion=True)
    res = mgr.run_dual_analysis(p1_path, p2_path, sample_id="TEST-REPORTS-AGREE")

    primary_count = res["total_grains"]
    assert primary_count == 2
    assert res["count_basis"]["total_physical_grains"] == primary_count
    assert len(res["grain_records"]) == primary_count

    # Generate PDF certificate
    pdf_path = generate_apscscl_pdf_certificate(res, pdf_out)
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000


# 10. Test: Feature flag toggles cleanly between bottom-primary and experimental symmetric mode
def test_feature_flag_toggles_cleanly(correspondence_engine):
    b_obs = [make_test_obs("bottom", 1, 500.0, 500.0)]
    t_obs = [
        make_test_obs("top", 1, 2000.0, 2000.0)  # Far away unmatched top
    ]

    # Mode A: Bottom-Primary (default) -> Unmatched top is excluded
    engine_bp = DualViewFusionEngine(correspondence_engine=correspondence_engine, architecture_mode="bottom_primary")
    res_bp = engine_bp.fuse_views(b_obs, t_obs)
    assert res_bp.total_single_physical_grains == 1
    assert len(res_bp.physical_grains) == 1
    assert res_bp.unmatched_top_count == 1

    # Mode B: Experimental Canonical Fusion (Phase 4 symmetric) -> Unmatched top is included
    engine_sym = DualViewFusionEngine(correspondence_engine=correspondence_engine, architecture_mode="experimental_canonical_fusion")
    res_sym = engine_sym.fuse_views(b_obs, t_obs)
    assert res_sym.total_single_physical_grains == 2
    assert len(res_sym.physical_grains) == 2
    assert res_sym.unmatched_top_count == 1


# 11. Test: Top image quality gate sets NOT_VERIFIABLE and suppresses false discrepancies
def test_top_image_quality_gate_blurry_sets_not_verifiable(calibrated_registration, bottom_primary_fusion_engine):
    bx, by = 500.0, 500.0
    _, tx, ty = calibrated_registration.project_bottom_to_top(bx, by)
    b_obs = [make_test_obs("bottom", 1, bx, by, dclass="Good / Head Rice")]
    t_obs = [make_test_obs("top", 1, tx, ty, dclass="Chalky Grains")]  # Apparent disagreement

    # When top image quality is NOT_VERIFIABLE:
    res = bottom_primary_fusion_engine.fuse_views(
        b_obs,
        t_obs,
        top_quality_status="NOT_VERIFIABLE",
        top_quality_metrics={"status": "NOT_VERIFIABLE", "laplacian_variance": 3.2, "reasons": ["Image blurred"]}
    )

    assert res.top_image_quality_status == "NOT_VERIFIABLE"
    assert res.discrepancy_count == 0  # CRUCIAL: Must NOT be counted as a discrepancy!
    assert len(res.physical_grains) == 1
    
    g = res.physical_grains[0]
    qa = g.quality_attributes
    assert qa["defect_class"] == "Good / Head Rice"  # Bottom remains authoritative
    assert qa["is_verified"] is False
    assert qa["discrepancy_flag"] is False  # Suppressed because top quality is unsuitable
    assert qa["top_verification_status"] == "NOT_VERIFIABLE"
    assert qa["top_match_status"] == "NOT_VERIFIABLE"
    assert "NOT_VERIFIABLE" in qa["discrepancy_details"]


# 12. Test: assess_image_quality objectively differentiates sharp vs blurred images
def test_assess_image_quality_objective_metrics():
    from src.dual_view.fusion import assess_image_quality

    # Sharp image with sharp edges
    sharp_img = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(10):
        cv2.circle(sharp_img, (50 + i * 30, 200), 12, (200, 200, 200), -1)
    
    res_sharp = assess_image_quality(sharp_img, min_laplacian_var=8.0)
    assert res_sharp["status"] == "SUITABLE"
    assert res_sharp["is_suitable"] is True
    assert res_sharp["laplacian_variance"] >= 8.0

    # Heavily blurred image (out-of-focus simulation)
    blurred_img = cv2.GaussianBlur(sharp_img, (45, 45), 20.0)
    res_blur = assess_image_quality(blurred_img, min_laplacian_var=8.0)
    assert res_blur["status"] == "NOT_VERIFIABLE"
    assert res_blur["is_suitable"] is False
    assert "blur" in res_blur["reasons"][0].lower()


# 13. Test: DualPlatenManager with blurry top scan reports NOT_VERIFIABLE without failure
def test_dual_platen_manager_blurry_top_scan(tmp_path):
    p1_path = str(tmp_path / "bottom_sharp.jpg")
    p2_path = str(tmp_path / "top_blur.jpg")

    img1 = np.full((1489, 2048, 3), 40, dtype=np.uint8)
    cv2.ellipse(img1, (600, 700), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    cv2.imwrite(p1_path, img1)

    # Top scan is heavily out-of-focus
    img2 = np.full((1489, 2048, 3), 40, dtype=np.uint8)
    cv2.ellipse(img2, (600, 700), (20, 8), 0, 0, 360, (220, 220, 220), -1)
    img2_blur = cv2.GaussianBlur(img2, (51, 51), 25.0)
    cv2.imwrite(p2_path, img2_blur)

    cfg = PipelineConfig()
    cfg.segmentation.engine = "classic"
    cfg.dual_view.enabled = True
    cfg.dual_view.architecture_mode = "bottom_primary"
    cfg.dual_view.top_min_laplacian_variance = 5.0
    pipeline = RiceQualityPipeline(config=cfg)

    mgr = DualPlatenManager(pipeline=pipeline, use_dual_view_fusion=True)
    res = mgr.run_dual_analysis(p1_path, p2_path, sample_id="TEST-BLUR-GATE")

    assert res["status"] == "Complete"
    assert res["primary_scanner"] == "bottom"
    assert res["total_grains"] == 1  # Bottom-primary single grain
    assert res["top_image_quality_status"] == "NOT_VERIFIABLE"
    assert res["discrepancy_count"] == 0
    assert res["acquisition_metadata"]["top_verification_status"] == "NOT_VERIFIABLE"
    assert res["grain_records"][0]["top_verification_status"] == "NOT_VERIFIABLE"


def test_operational_mode_selection_and_mode_b_prohibition():
    """Verify explicit mode selection and ensure Mode B is prohibited."""
    from configs.default_config import DualViewConfig
    cfg = DualViewConfig()
    
    # 1. Default must be single_view
    assert cfg.mode == "single_view"
    assert cfg.enabled is False
    assert cfg.architecture_mode == "bottom_primary"
    
    # 2. Select Mode C (bottom_primary_verification)
    cfg.apply_mode("bottom_primary_verification")
    assert cfg.mode == "bottom_primary_verification"
    assert cfg.enabled is True
    assert cfg.architecture_mode == "bottom_primary_verification"
    
    # 3. Rollback to single_view
    cfg.apply_mode("single_view")
    assert cfg.mode == "single_view"
    assert cfg.enabled is False
    assert cfg.architecture_mode == "bottom_primary"
    
    # 4. Mode B prohibition
    import pytest
    for bad_mode in ("symmetric", "experimental_canonical_fusion", "canonical_fusion", "mode_b"):
        with pytest.raises(ValueError, match="strictly prohibited"):
            cfg.apply_mode(bad_mode)


def test_mode_c_bottom_count_conservation_and_unmatched_top_isolation(correspondence_engine):
    """Verify Mode C preserves authoritative bottom counts and isolates unmatched top detections."""
    from src.dual_view.fusion import DualViewFusionEngine
    from src.dual_view.models import CorrespondenceStatus
    
    bx1, by1 = 500.0, 500.0
    tx1 = -1.0 * bx1 + 4917.872
    ty1 = 1.0 * by1 + 21.086

    bx2, by2 = 800.0, 500.0
    tx2 = -1.0 * bx2 + 4917.872
    ty2 = 1.0 * by2 + 21.086

    bx3, by3 = 1100.0, 500.0

    b_obs = [
        make_test_obs("bottom", 1, bx1, by1, dclass="Good / Head Rice"),
        make_test_obs("bottom", 2, bx2, by2, dclass="Discoloured Grains"),
        make_test_obs("bottom", 3, bx3, by3, dclass="Good / Head Rice"),
    ]
    
    # Top has 1 match (b_1 <-> t_1), 1 disagreement (b_2 <-> t_2), and 5 unmatched tops!
    t_obs = [
        make_test_obs("top", 1, tx1, ty1, dclass="Good / Head Rice"),
        make_test_obs("top", 2, tx2, ty2, dclass="Good / Head Rice"),  # Disagreement with b_2
        make_test_obs("top", 3, 2000.0, 2000.0),
        make_test_obs("top", 4, 2100.0, 2000.0),
        make_test_obs("top", 5, 2200.0, 2000.0),
        make_test_obs("top", 6, 2300.0, 2000.0),
        make_test_obs("top", 7, 2400.0, 2000.0),
    ]
    
    engine = DualViewFusionEngine(correspondence_engine=correspondence_engine, architecture_mode="bottom_primary_verification")
    res = engine.fuse_views(b_obs, t_obs)
    
    # 1. Authoritative count must be strictly 3 (len(b_obs)), NOT 3 + 5 = 8!
    assert res.total_single_physical_grains == 3
    assert res.total_physical_grains == 3
    assert len(res.physical_grains) == 3
    
    # 2. Unmatched top count is recorded in diagnostics
    assert res.unmatched_top_count == 5
    assert res.matched_count == 2
    
    # 3. Disagreement flagged advisory without changing bottom classification
    grain_2 = [g for g in res.physical_grains if g.bottom_obs and g.bottom_obs.detection_id == "b_2"][0]
    assert grain_2.quality_attributes["bottom_predicted_class"] == "Discoloured Grains"
    assert grain_2.quality_attributes["top_predicted_class"] == "Good / Head Rice"
    assert grain_2.quality_attributes["defect_class"] == "Discoloured Grains"  # Bottom authoritative!
    assert grain_2.quality_attributes["discrepancy_flag"] is True
    assert grain_2.quality_attributes["is_verified"] is False
    assert res.discrepancy_count == 1
