# Targeted QA and Regression Test Suite for RiceAge M3.5 Audit
# Verifies feature extraction robustness, multi-defect logic, denominator integrity,
# report consistency across JSON/CSV/HTML, and raw inference isolation.

import os
import json
import numpy as np
import pandas as pd
import pytest
import cv2

from src.quality_classification import (
    extract_quality_features_from_crop,
    BaselineForeignMatterRuleClassifier,
    BaselineChalkyRuleClassifier,
    BaselineDiscolouredRuleClassifier,
    BaselineDamagedRuleClassifier,
    SupervisedQualityHead,
    get_default_feature_sets
)
from src.quality_grading import (
    GrainClassification,
    SampleQualityMetrics,
    compute_sample_quality_metrics,
    evaluate_quality_grade,
    CODEX_CXS_198_1995,
    USDA_MILLED_RICE,
    IS_4333_AGMARK
)
from src.quality_reporting import (
    export_sample_quality_report,
    generate_html_report
)


# -------------------------------------------------------------------------
# 1. Feature Extraction on Degraded / Edge-Case Crops
# -------------------------------------------------------------------------

def test_feature_extraction_empty_and_zero_crops():
    """Verify feature extractor safely returns None for empty or sub-10px crops."""
    empty_rgb = np.zeros((50, 50, 3), dtype=np.uint8)
    empty_mask = np.zeros((50, 50), dtype=np.uint8)
    
    feats = extract_quality_features_from_crop(empty_rgb, empty_mask, grain_id="EMPTY_01")
    assert feats is None


def test_feature_extraction_low_contrast_and_noise():
    """Verify feature extractor is stable on noisy low-contrast grain crops."""
    np.random.seed(42)
    # Synthetic grain: ellipse on uniform gray background
    crop_rgb = np.full((100, 60, 3), 106, dtype=np.uint8)
    mask = np.zeros((100, 60), dtype=np.uint8)
    cv2.ellipse(mask, (30, 50), (20, 40), 0, 0, 360, 255, -1)
    
    # Low contrast grain: intensity 112 with Gaussian noise
    noise = np.random.normal(0, 3, (100, 60, 3)).astype(np.int16)
    grain_pixels = (mask > 0)[:, :, None]
    crop_rgb = np.where(grain_pixels, np.clip(112 + noise, 0, 255).astype(np.uint8), crop_rgb)

    feats = extract_quality_features_from_crop(crop_rgb, mask, grain_id="NOISY_01")
    assert feats is not None
    assert feats.area_px > 500
    assert feats.length_px > 50.0
    assert np.isfinite(feats.mean_b)
    assert np.isfinite(feats.chalky_ratio)


def test_feature_extraction_dark_borders():
    """Verify feature extraction with dark margin bezel artifacts in the crop."""
    crop_rgb = np.zeros((100, 60, 3), dtype=np.uint8)  # Black border
    crop_rgb[10:90, 10:50] = 106  # Scanner background
    mask = np.zeros((100, 60), dtype=np.uint8)
    cv2.ellipse(mask, (30, 50), (12, 30), 0, 0, 360, 255, -1)
    crop_rgb[mask > 0] = 160  # Grain body

    feats = extract_quality_features_from_crop(crop_rgb, mask, grain_id="BEZEL_BORDER_01")
    assert feats is not None
    assert feats.area_px > 200
    assert feats.solidity > 0.90


# -------------------------------------------------------------------------
# 2. Multi-Defect Rice & Denominator Accounting
# -------------------------------------------------------------------------

def test_multiple_defects_do_not_inflate_rice_denominator():
    """Verify grains with >=2 defects (e.g. Broken + Chalky + Damaged) do not double-count."""
    grains = [
        # Grain 1: Whole & clean
        GrainClassification(object_id="G1", is_rice=True, is_foreign_matter=False, morphology="Whole"),
        # Grain 2: Broken & Chalky (2 defects)
        GrainClassification(object_id="G2", is_rice=True, is_foreign_matter=False, morphology="Broken", is_chalky=True),
        # Grain 3: Broken & Damaged & Discoloured (3 defects)
        GrainClassification(object_id="G3", is_rice=True, is_foreign_matter=False, morphology="Broken", is_damaged=True, is_discoloured=True),
        # Grain 4: Whole & Chalky
        GrainClassification(object_id="G4", is_rice=True, is_foreign_matter=False, morphology="Whole", is_chalky=True),
        # Object 5: Foreign Matter (weed seed)
        GrainClassification(object_id="G5", is_rice=False, is_foreign_matter=True, morphology="N/A")
    ]

    metrics = compute_sample_quality_metrics(grains, sample_id="MULTI_TEST")

    # Invariants
    assert metrics.total_detected_objects == 5
    assert metrics.total_rice_grains == 4  # G5 excluded
    assert metrics.total_foreign_matter == 1
    assert metrics.whole_count + metrics.broken_count == metrics.total_rice_grains
    assert metrics.whole_count == 2
    assert metrics.broken_count == 2
    assert metrics.chalky_count == 2
    assert metrics.damaged_count == 1
    assert metrics.discoloured_count == 1
    # Note: In RiceAge ontology, Whole vs Broken is morphological classification,
    # while defect flags are chalky, damaged, discoloured. G3 has both damaged & discoloured.
    assert metrics.multi_defect_rice_count == 1  # G3

    # Exact percentages on total_rice_grains = 4
    assert metrics.whole_percentage == 50.0
    assert metrics.broken_percentage == 50.0
    assert metrics.chalky_percentage == 50.0
    assert metrics.damaged_percentage == 25.0
    assert metrics.discoloured_percentage == 25.0
    # Foreign matter percentage evaluated on total_detected_objects = 5
    assert metrics.foreign_matter_percentage == 20.0


# -------------------------------------------------------------------------
# 3. Grade Boundary Conditions
# -------------------------------------------------------------------------

def test_exact_grade_boundary_comparison():
    """Verify inclusive boundary comparison (<= limit passes, > limit fails)."""
    # Codex Grade 1: max_broken = 10.0%, max_chalky = 3.0%
    # Build sample with exactly 10.0% broken and 3.0% chalky out of 100 grains
    grains_exact = []
    for i in range(10):
        grains_exact.append(GrainClassification(object_id=f"B_{i}", is_rice=True, is_foreign_matter=False, morphology="Broken"))
    for i in range(90):
        # 3 chalky whole grains
        is_ch = (i < 3)
        grains_exact.append(GrainClassification(object_id=f"W_{i}", is_rice=True, is_foreign_matter=False, morphology="Whole", is_chalky=is_ch))

    metrics_exact = compute_sample_quality_metrics(grains_exact)
    assert metrics_exact.broken_percentage == 10.0
    assert metrics_exact.chalky_percentage == 3.0

    dec_exact = evaluate_quality_grade(metrics_exact, profile=CODEX_CXS_198_1995)
    # Should meet Grade_1 (fails Extra because Extra max broken is 5.0%)
    assert dec_exact.assigned_grade == "Grade_1"
    assert dec_exact.is_compliant is True

    # Now add 1 more broken grain (11 broken / 100 = 11.0% broken)
    grains_fail = list(grains_exact)
    grains_fail[10] = GrainClassification(object_id="B_extra", is_rice=True, is_foreign_matter=False, morphology="Broken")
    metrics_fail = compute_sample_quality_metrics(grains_fail)
    assert metrics_fail.broken_percentage == 11.0

    dec_fail = evaluate_quality_grade(metrics_fail, profile=CODEX_CXS_198_1995)
    # Should drop to Grade_2 (limit 15.0%)
    assert dec_fail.assigned_grade == "Grade_2"


def test_missing_non_optical_measurements_marks_incomplete():
    """Verify that require_full_criteria=True flags regulatory non-optical criteria as INCOMPLETE."""
    grains = [
        GrainClassification(object_id=f"W_{i}", is_rice=True, is_foreign_matter=False, morphology="Whole")
        for i in range(20)
    ]
    metrics = compute_sample_quality_metrics(grains)
    
    # Pure optical screening mode
    opt_dec = evaluate_quality_grade(metrics, profile=CODEX_CXS_198_1995, require_full_criteria=False)
    assert opt_dec.status == "DECIDED"
    assert opt_dec.assigned_grade == "Extra"

    # Full regulatory certification mode
    reg_dec = evaluate_quality_grade(metrics, profile=CODEX_CXS_198_1995, require_full_criteria=True)
    assert reg_dec.status == "INCOMPLETE"
    assert len(reg_dec.unsupported_criteria) > 0
    assert any("moisture" in c.lower() for c in reg_dec.unsupported_criteria)


# -------------------------------------------------------------------------
# 4. JSON / CSV / HTML Cross-Format Reconciliation
# -------------------------------------------------------------------------

def test_report_export_reconciles_across_all_three_formats(tmp_path):
    """Verify JSON, CSV, and HTML reports contain strictly reconciled metrics."""
    grains = [
        GrainClassification(object_id="G1", is_rice=True, is_foreign_matter=False, morphology="Whole"),
        GrainClassification(object_id="G2", is_rice=True, is_foreign_matter=False, morphology="Broken", is_chalky=True),
        GrainClassification(object_id="G3", is_rice=True, is_foreign_matter=False, morphology="Whole", is_discoloured=True),
        GrainClassification(object_id="G4", is_rice=False, is_foreign_matter=True, morphology="N/A")
    ]
    metrics = compute_sample_quality_metrics(grains, sample_id="RECON_TEST")
    grading = evaluate_quality_grade(metrics, profile=CODEX_CXS_198_1995)

    out_dir = str(tmp_path)
    res = export_sample_quality_report(
        sample_id="RECON_TEST",
        metrics=metrics,
        grading=grading,
        grains=grains,
        output_dir=out_dir
    )

    # 1. Check files created
    assert os.path.exists(res["json_path"])
    assert os.path.exists(res["csv_path"])
    assert os.path.exists(res["html_path"])

    # 2. Check JSON metrics
    with open(res["json_path"], "r") as f:
        jdata = json.load(f)
    assert jdata["quality_metrics"]["total_detected_objects"] == 4
    assert jdata["quality_metrics"]["total_rice_grains"] == 3
    assert jdata["quality_metrics"]["total_foreign_matter"] == 1
    assert jdata["quality_metrics"]["broken_count"] == 1
    assert jdata["quality_metrics"]["chalky_count"] == 1

    # 3. Check CSV grain rows
    df_csv = pd.read_csv(res["csv_path"])
    assert len(df_csv) == 4
    assert (df_csv["is_foreign_matter"] == 1).sum() == 1
    assert (df_csv["morphology"] == "Broken").sum() == 1

    # 4. Check HTML contents
    with open(res["html_path"], "r") as f:
        html_content = f.read()
    assert "RECON_TEST" in html_content
    assert str(metrics.total_rice_grains) in html_content
    assert grading.assigned_grade in html_content


# -------------------------------------------------------------------------
# 5. Isolation of Inference from Ground-Truth Annotations
# -------------------------------------------------------------------------

def test_inference_operates_without_ground_truth():
    """Verify that feature extraction and classifiers require zero ground-truth data."""
    # Synthetic scan region
    crop = np.full((80, 40, 3), 106, dtype=np.uint8)
    mask = np.zeros((80, 40), dtype=np.uint8)
    cv2.ellipse(mask, (20, 40), (14, 30), 0, 0, 360, 255, -1)
    crop[mask > 0] = [210, 210, 210]  # Chalky appearance

    import dataclasses
    feats = extract_quality_features_from_crop(crop, mask, grain_id="INFER_01")
    assert feats is not None
    f_dict = dataclasses.asdict(feats)
    # Ground truth is unassigned / unknown at raw inference time
    assert f_dict["gt_condition"] == "Unknown"
    assert f_dict["category"] == "unknown"

    # Verify classifier feature sets contain strictly photometric/morphological features
    feature_sets = get_default_feature_sets()
    for head_name, cols in feature_sets.items():
        assert "gt_condition" not in cols
        assert "category" not in cols

    # Rule classifiers evaluate purely on features
    df_feat = pd.DataFrame([f_dict])
    ch_rule = BaselineChalkyRuleClassifier(chalky_threshold=0.50)
    ch_pred = ch_rule.predict(df_feat)
    assert ch_pred[0] in [0, 1]

    fm_rule = BaselineForeignMatterRuleClassifier()
    fm_pred = fm_rule.predict(df_feat)
    assert fm_pred[0] == 0  # Rice shape, not foreign matter
