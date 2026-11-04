# Automated tests for RiceAge M3 quality reporting pipeline (src/quality_reporting.py).

import os
import json
import pytest
import pandas as pd

from src.quality_grading import (
    GrainClassification,
    compute_sample_quality_metrics,
    evaluate_quality_grade
)
from src.quality_reporting import generate_html_report, export_sample_quality_report


def test_html_report_rendering():
    """Verify that generate_html_report outputs well-formed HTML with critical elements."""
    grains = [
        GrainClassification("G1", is_rice=True, is_foreign_matter=False, morphology="Whole"),
        GrainClassification("G2", is_rice=True, is_foreign_matter=False, morphology="Broken", is_chalky=True),
        GrainClassification("FM1", is_rice=False, is_foreign_matter=True, morphology="N/A")
    ]
    metrics = compute_sample_quality_metrics(grains, sample_id="TEST_SAMPLE_01")
    grading = evaluate_quality_grade(metrics, profile_id="CODEX_CXS_198_1995")

    html = generate_html_report("TEST_SAMPLE_01", metrics, grading, grains)
    assert "<!DOCTYPE html>" in html
    assert "TEST_SAMPLE_01" in html
    assert "Whole Rice (Head Rice)" in html
    assert "Chalky Kernels" in html
    assert "Foreign Matter" in html
    assert "Regulatory Certification & Laboratory Limitations Notice" in html


def test_export_sample_quality_report(tmp_path):
    """Verify JSON, CSV, and HTML file exports and schema integrity."""
    grains = [
        GrainClassification("G1", is_rice=True, is_foreign_matter=False, morphology="Whole"),
        GrainClassification("G2", is_rice=True, is_foreign_matter=False, morphology="Broken", is_chalky=True),
        GrainClassification("FM1", is_rice=False, is_foreign_matter=True, morphology="N/A")
    ]
    metrics = compute_sample_quality_metrics(grains, sample_id="EXPORT_TEST")
    grading = evaluate_quality_grade(metrics, profile_id="CODEX_CXS_198_1995")

    out_dir = str(tmp_path / "reports")
    paths = export_sample_quality_report("EXPORT_TEST", metrics, grading, grains, output_dir=out_dir)

    # 1. JSON verification
    assert os.path.exists(paths["json_path"])
    with open(paths["json_path"]) as f:
        data = json.load(f)
    assert data["report_metadata"]["sample_id"] == "EXPORT_TEST"
    assert data["quality_metrics"]["total_detected_objects"] == 3
    assert data["quality_metrics"]["total_rice_grains"] == 2
    assert data["quality_metrics"]["total_foreign_matter"] == 1
    assert len(data["grain_classifications"]) == 3

    # 2. CSV verification
    assert os.path.exists(paths["csv_path"])
    df = pd.read_csv(paths["csv_path"])
    assert len(df) == 3
    assert list(df.columns) == [
        "sample_id", "object_id", "is_rice", "is_foreign_matter",
        "morphology", "is_chalky", "is_damaged", "is_discoloured"
    ]

    # 3. HTML verification
    assert os.path.exists(paths["html_path"])
    with open(paths["html_path"]) as f:
        html = f.read()
    assert "EXPORT_TEST" in html
    assert "Codex Standard for Rice" in html
