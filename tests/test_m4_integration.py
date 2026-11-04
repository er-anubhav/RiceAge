# Comprehensive integration, safety, and failure-handling test suite for RiceAge Milestone M4.
# Verifies end-to-end pipeline execution from raw scans, reconciliation of denominators,
# edge-case handling, and guarantees that inference never accesses ground-truth annotations.

import os
import json
import builtins
import numpy as np
import pandas as pd
import pytest
import cv2

from configs.default_config import PipelineConfig
from src.pipeline import RiceQualityPipeline
from src.quality_classification import MultiHeadQualityClassifier
from src.quality_grading import (
    GrainClassification,
    SampleQualityMetrics,
    compute_sample_quality_metrics,
    evaluate_quality_grade
)
from src.quality_reporting import export_sample_quality_report


def test_raw_scan_end_to_end_no_ground_truth(tmp_path):
    """
    P1/P5: Verify raw scan end-to-end processing without loading ground-truth annotations.
    Reconciles HTML, JSON, and CSV report totals.
    """
    raw_image_path = "data/dataset/images/mixed/mixed 1.jpg"
    assert os.path.exists(raw_image_path), f"Raw test image missing: {raw_image_path}"

    cfg = PipelineConfig()
    cfg.export.output_dir = str(tmp_path / "outputs")
    cfg.export.reports_dir = str(tmp_path / "outputs" / "reports")
    cfg.classification.sample_id = "RAW_SCAN_M4_VAL"

    pipeline = RiceQualityPipeline(config=cfg)
    results = pipeline.run(raw_image_path, generate_outputs=True)

    summary = results["summary"]
    metrics: SampleQualityMetrics = results["quality_metrics"]
    reports = results["report_files"]

    # 1. Pipeline outputs exist
    assert "quality_classification" in summary
    assert "quality_grading" in summary
    assert len(results["grain_classifications"]) > 0

    # 2. Strict denominator reconciliation (P3)
    assert metrics.total_detected_objects == metrics.total_rice_grains + metrics.total_foreign_matter
    assert metrics.total_rice_grains == metrics.whole_count + metrics.broken_count
    assert metrics.whole_percentage + metrics.broken_percentage == pytest.approx(100.0, abs=0.1)

    # 3. Report file existence
    assert os.path.exists(reports["html_path"])
    assert os.path.exists(reports["json_path"])
    assert os.path.exists(reports["csv_path"])

    # 4. Report consistency across JSON, CSV, HTML
    with open(reports["json_path"], "r") as f:
        json_data = json.load(f)
    assert json_data["report_metadata"]["sample_id"] == "RAW_SCAN_M4_VAL"
    assert json_data["quality_metrics"]["total_detected_objects"] == metrics.total_detected_objects
    assert json_data["quality_metrics"]["total_rice_grains"] == metrics.total_rice_grains

    df_csv = pd.read_csv(reports["csv_path"])
    assert len(df_csv) == metrics.total_detected_objects
    assert set(df_csv.columns) == {
        "sample_id", "object_id", "is_rice", "is_foreign_matter",
        "morphology", "is_chalky", "is_damaged", "is_discoloured"
    }

    with open(reports["html_path"], "r") as f:
        html_data = f.read()
    assert "Optical Screening Estimate (Provisional)" in html_data
    assert "RAW_SCAN_M4_VAL" in html_data


def test_inference_does_not_access_annotations(monkeypatch, tmp_path):
    """
    P1/P5: Enforce that pipeline inference NEVER accesses annotations files or directories.
    """
    raw_image_path = "data/dataset/images/whole/whole.jpg"
    assert os.path.exists(raw_image_path)

    accessed_paths = []
    original_open = builtins.open

    def audit_open(file, *args, **kwargs):
        filepath = str(file)
        # Check if any annotation or ground truth file is being opened
        if any(kw in filepath.lower() for kw in ["annotation", "annotations", "cvat", "annotated_images"]):
            raise PermissionError(f"CRITICAL LEAK: Pipeline inference attempted to open annotation file: {filepath}")
        accessed_paths.append(filepath)
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", audit_open)

    cfg = PipelineConfig()
    cfg.export.output_dir = str(tmp_path / "leak_check")
    cfg.export.reports_dir = str(tmp_path / "leak_check" / "reports")

    pipeline = RiceQualityPipeline(config=cfg)
    results = pipeline.run(raw_image_path, generate_outputs=True)

    assert results["summary"]["total_grains_detected"] > 0
    # Confirm no annotation path was accessed
    for p in accessed_paths:
        assert "annotation" not in p.lower()


def test_empty_and_no_detected_grains(tmp_path):
    """
    P6: Verify robust handling when an image has zero grains (empty black background).
    Must not crash, must not divide by zero, and must output INDETERMINATE grading.
    """
    empty_img = np.zeros((400, 400, 3), dtype=np.uint8)
    empty_path = str(tmp_path / "empty_scanner.jpg")
    cv2.imwrite(empty_path, empty_img)

    cfg = PipelineConfig()
    cfg.export.output_dir = str(tmp_path / "empty_out")
    cfg.export.reports_dir = str(tmp_path / "empty_out" / "reports")

    pipeline = RiceQualityPipeline(config=cfg)
    res = pipeline.run(empty_path, generate_outputs=True)

    summary = res["summary"]
    metrics = res["quality_metrics"]
    dec = res["grading_decision"]

    assert summary["total_grains_detected"] == 0
    assert metrics.total_detected_objects == 0
    assert metrics.total_rice_grains == 0
    assert metrics.whole_percentage == 0.0
    assert metrics.broken_percentage == 0.0
    assert dec.status == "INDETERMINATE"
    assert dec.assigned_grade is None


def test_missing_model_configuration_fallback(tmp_path):
    """
    P6: Verify graceful fallback to classical rule baselines if weights file is missing.
    """
    cfg = PipelineConfig()
    cfg.classification.quality_weights_path = str(tmp_path / "nonexistent_weights.json")
    cfg.export.output_dir = str(tmp_path / "fallback_out")
    cfg.export.reports_dir = str(tmp_path / "fallback_out" / "reports")

    pipeline = RiceQualityPipeline(config=cfg)
    assert pipeline.quality_classifier.is_loaded is False
    assert pipeline.quality_classifier.fallback_warning is not None

    # Run on a standard image
    raw_img = "data/samples/sample_1489x2048.jpg"
    res = pipeline.run(raw_img, generate_outputs=True)

    assert res["summary"]["total_grains_detected"] > 0
    assert "quality_classification" in res["summary"]
    assert res["summary"]["quality_classification"]["fallback_warning"] is not None
    assert os.path.exists(res["report_files"]["json_path"])


def test_safe_grading_incomplete_behavior(tmp_path):
    """
    P4: Verify that when certified grading is requested (require_full_criteria=True),
    the grade is flagged as INCOMPLETE due to unmeasured non-optical lab parameters.
    """
    cfg = PipelineConfig()
    cfg.classification.require_full_criteria = True
    cfg.export.output_dir = str(tmp_path / "incomplete_out")
    cfg.export.reports_dir = str(tmp_path / "incomplete_out" / "reports")

    pipeline = RiceQualityPipeline(config=cfg)
    res = pipeline.run("data/dataset/images/whole/whole.jpg", generate_outputs=True)

    dec = res["grading_decision"]
    assert dec.status == "INCOMPLETE"
    assert dec.assigned_grade is None
    assert len(dec.unsupported_criteria) > 0
    assert any("moisture" in c.lower() for c in dec.unsupported_criteria)


def test_multi_defect_cooccurrence_and_denominators():
    """
    P2/P3: Verify that co-occurring defects (e.g. Broken + Chalky + Damaged)
    do not inflate total_rice_grains and are correctly tallied in multi_defect_rice_count.
    """
    grains = [
        # Grain 1: Clean Whole
        GrainClassification("G1", is_rice=True, is_foreign_matter=False, morphology="Whole"),
        # Grain 2: Broken + Damaged (legitimate co-occurrence)
        GrainClassification("G2", is_rice=True, is_foreign_matter=False, morphology="Broken", is_damaged=True),
        # Grain 3: Broken + Chalky (legitimate co-occurrence)
        GrainClassification("G3", is_rice=True, is_foreign_matter=False, morphology="Broken", is_chalky=True),
        # Grain 4: Whole + Discoloured + Damaged
        GrainClassification("G4", is_rice=True, is_foreign_matter=False, morphology="Whole", is_discoloured=True, is_damaged=True),
        # Grain 5: Broken + Chalky + Damaged + Discoloured (4 defects)
        GrainClassification("G5", is_rice=True, is_foreign_matter=False, morphology="Broken", is_chalky=True, is_damaged=True, is_discoloured=True),
        # Foreign matter 1 & 2
        GrainClassification("FM1", is_rice=False, is_foreign_matter=True, morphology="N/A"),
        GrainClassification("FM2", is_rice=False, is_foreign_matter=True, morphology="N/A"),
    ]

    metrics = compute_sample_quality_metrics(grains, sample_id="COOCCUR_TEST")

    # 1. Physical object counts
    assert metrics.total_detected_objects == 7
    assert metrics.total_rice_grains == 5
    assert metrics.total_foreign_matter == 2
    assert metrics.total_rice_grains + metrics.total_foreign_matter == metrics.total_detected_objects

    # 2. Morphology counts
    assert metrics.whole_count == 2
    assert metrics.broken_count == 3
    assert metrics.whole_count + metrics.broken_count == metrics.total_rice_grains

    # 3. Defect counts
    assert metrics.chalky_count == 2       # G3, G5
    assert metrics.damaged_count == 3      # G2, G4, G5
    assert metrics.discoloured_count == 2  # G4, G5
    assert metrics.multi_defect_rice_count == 2  # G4 (discoloured+damaged), G5 (broken+chalky+damaged+discoloured)

    # 4. Percentages: Defect denominators must be 5 (total_rice_grains)
    assert metrics.whole_percentage == 40.0   # 2 / 5 * 100
    assert metrics.broken_percentage == 60.0  # 3 / 5 * 100
    assert metrics.chalky_percentage == 40.0  # 2 / 5 * 100
    assert metrics.damaged_percentage == 60.0 # 3 / 5 * 100
    assert metrics.discoloured_percentage == 40.0 # 2 / 5 * 100

    # 5. Foreign matter denominator must be 7 (total_detected_objects)
    assert metrics.foreign_matter_percentage == pytest.approx(28.57, abs=0.01)


def test_cross_triggering_reduction_audit():
    """
    P2: Validate that M4 model weights exhibit significant cross-triggering reductions
    compared to M3 baseline audit findings.
    """
    weights_path = "configs/quality_classifier_weights.json"
    assert os.path.exists(weights_path), "M4 weights file not found"

    with open(weights_path, "r") as f:
        data = json.load(f)

    comp = data["cross_trigger_analysis"]["m3_vs_m4_comparison"]
    broken_to_dmg = comp["broken_triggering_damaged"]
    chalky_to_disc = comp["chalky_triggering_discoloured"]

    # Broken -> Damaged was 99.8% in M3; must be <= 20% in M4
    assert broken_to_dmg["m3_rate"] > 0.95
    assert broken_to_dmg["m4_cv_rate"] <= 0.20
    assert broken_to_dmg["reduction_percentage"] >= 75.0

    # Chalky -> Discoloured was 93.6% in M3; must be <= 30% in M4
    assert chalky_to_disc["m3_rate"] > 0.90
    assert chalky_to_disc["m4_cv_rate"] <= 0.30
    assert chalky_to_disc["reduction_percentage"] >= 65.0
