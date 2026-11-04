# Unit tests for RiceAge M3 Quality Grading Engine (src/quality_grading.py).
# Validates percentages, overlapping defect handling, grade boundaries, and indeterminate states.

import pytest
from src.quality_grading import (
    GrainClassification,
    SampleQualityMetrics,
    GradeThreshold,
    GradingProfile,
    compute_sample_quality_metrics,
    evaluate_quality_grade,
    CODEX_CXS_198_1995,
    USDA_MILLED_RICE,
    IS_4333_AGMARK
)


def test_empty_sample_metrics():
    """Verify that an empty sample returns zero counts without division by zero."""
    metrics = compute_sample_quality_metrics([], sample_id="EMPTY_SAMPLE")
    assert metrics.total_detected_objects == 0
    assert metrics.total_rice_grains == 0
    assert metrics.total_foreign_matter == 0
    assert metrics.whole_percentage == 0.0
    assert metrics.broken_percentage == 0.0
    assert metrics.foreign_matter_percentage == 0.0

    decision = evaluate_quality_grade(metrics, profile_id="CODEX_CXS_198_1995")
    assert decision.status == "INDETERMINATE"
    assert decision.assigned_grade is None
    assert decision.is_compliant is False


def test_pure_whole_clean_sample():
    """Verify that 100% clean whole rice grades as Extra / US No. 1 / Grade I."""
    grains = [
        GrainClassification(
            object_id=f"G_{i}",
            is_rice=True,
            is_foreign_matter=False,
            morphology="Whole",
            is_chalky=False,
            is_damaged=False,
            is_discoloured=False
        )
        for i in range(100)
    ]
    metrics = compute_sample_quality_metrics(grains, sample_id="CLEAN_WHOLE")
    assert metrics.total_detected_objects == 100
    assert metrics.total_rice_grains == 100
    assert metrics.total_foreign_matter == 0
    assert metrics.whole_count == 100
    assert metrics.broken_count == 0
    assert metrics.whole_percentage == 100.0
    assert metrics.broken_percentage == 0.0
    assert metrics.chalky_percentage == 0.0
    assert metrics.foreign_matter_percentage == 0.0

    # Codex Extra Grade check
    codex_decision = evaluate_quality_grade(metrics, profile_id="CODEX_CXS_198_1995")
    assert codex_decision.assigned_grade == "Extra"
    assert codex_decision.is_compliant is True
    assert codex_decision.status == "DECIDED"

    # USDA No 1 check
    usda_decision = evaluate_quality_grade(metrics, profile_id="USDA_MILLED_RICE")
    assert usda_decision.assigned_grade == "US_No_1"

    # IS 4333 Grade I check
    is_decision = evaluate_quality_grade(metrics, profile_id="IS_4333_AGMARK")
    assert is_decision.assigned_grade == "Grade_I"


def test_coexisting_defects_no_double_counting():
    """
    Critical test: Verify that a grain with multiple defects (Broken + Chalky + Damaged)
    is counted exactly once in total_rice_grains and does not inflate the denominator.
    """
    grains = [
        # Grain 1: Sound whole
        GrainClassification("G1", is_rice=True, is_foreign_matter=False, morphology="Whole"),
        # Grain 2: Sound whole
        GrainClassification("G2", is_rice=True, is_foreign_matter=False, morphology="Whole"),
        # Grain 3: Broken + Chalky + Damaged (3 simultaneous conditions)
        GrainClassification("G3", is_rice=True, is_foreign_matter=False, morphology="Broken", is_chalky=True, is_damaged=True),
        # Grain 4: Whole + Discoloured
        GrainClassification("G4", is_rice=True, is_foreign_matter=False, morphology="Whole", is_discoloured=True),
        # Object 5: Foreign matter
        GrainClassification("FM1", is_rice=False, is_foreign_matter=True, morphology="N/A")
    ]

    metrics = compute_sample_quality_metrics(grains, sample_id="MULTI_DEFECT")

    # Object counts
    assert metrics.total_detected_objects == 5
    assert metrics.total_rice_grains == 4  # G1, G2, G3, G4
    assert metrics.total_foreign_matter == 1  # FM1

    # Morphology counts sum to total_rice_grains
    assert metrics.whole_count == 3
    assert metrics.broken_count == 1
    assert metrics.whole_count + metrics.broken_count == metrics.total_rice_grains

    # Coexisting counts
    assert metrics.chalky_count == 1  # G3
    assert metrics.damaged_count == 1  # G3
    assert metrics.discoloured_count == 1  # G4
    assert metrics.multi_defect_rice_count == 1  # G3 has broken, chalky, damaged

    # Percentages: Denominator is 4 rice grains
    assert metrics.whole_percentage == 75.0   # 3 / 4
    assert metrics.broken_percentage == 25.0  # 1 / 4
    assert metrics.chalky_percentage == 25.0  # 1 / 4
    assert metrics.damaged_percentage == 25.0 # 1 / 4
    assert metrics.discoloured_percentage == 25.0 # 1 / 4

    # Foreign matter denominator is 5 total objects
    assert metrics.foreign_matter_percentage == 20.0  # 1 / 5


def test_grade_threshold_boundary_codex():
    """
    Verify precision at grade boundaries:
    Codex Extra: Broken <= 5.0%
    Codex Grade 1: Broken <= 10.0%
    """
    # Sample with exactly 5.0% broken (5 broken, 95 whole)
    grains_exact_5 = [
        GrainClassification(f"G_{i}", is_rice=True, is_foreign_matter=False, morphology="Broken" if i < 5 else "Whole")
        for i in range(100)
    ]
    m_exact_5 = compute_sample_quality_metrics(grains_exact_5)
    assert m_exact_5.broken_percentage == 5.0
    dec_exact = evaluate_quality_grade(m_exact_5, profile_id="CODEX_CXS_198_1995")
    assert dec_exact.assigned_grade == "Extra"

    # Sample with 5.1% broken (e.g. 51 broken out of 1000)
    grains_5_1 = [
        GrainClassification(f"G_{i}", is_rice=True, is_foreign_matter=False, morphology="Broken" if i < 51 else "Whole")
        for i in range(1000)
    ]
    m_5_1 = compute_sample_quality_metrics(grains_5_1)
    assert m_5_1.broken_percentage == 5.1
    dec_5_1 = evaluate_quality_grade(m_5_1, profile_id="CODEX_CXS_198_1995")
    assert dec_5_1.assigned_grade == "Grade_1"  # Demoted from Extra to Grade 1


def test_substandard_sample():
    """Verify that a sample exceeding all tier limits is assigned Substandard / Below FAQ."""
    # 40% broken rice exceeds Codex Grade 3 (max 25%)
    grains_bad = [
        GrainClassification(f"G_{i}", is_rice=True, is_foreign_matter=False, morphology="Broken" if i < 40 else "Whole")
        for i in range(100)
    ]
    m_bad = compute_sample_quality_metrics(grains_bad)
    dec_bad = evaluate_quality_grade(m_bad, profile_id="CODEX_CXS_198_1995")
    assert dec_bad.assigned_grade == "Substandard / Below FAQ"
    assert dec_bad.is_compliant is False
    assert "broken_rice" in dec_bad.failed_criteria["Grade_3"]


def test_incomplete_status_when_full_criteria_required():
    """
    Verify that if require_full_criteria is True, status is marked INCOMPLETE
    due to unmeasured laboratory criteria (moisture, odor).
    """
    grains = [
        GrainClassification(f"G_{i}", is_rice=True, is_foreign_matter=False, morphology="Whole")
        for i in range(50)
    ]
    m = compute_sample_quality_metrics(grains)
    dec = evaluate_quality_grade(m, profile_id="CODEX_CXS_198_1995", require_full_criteria=True)
    assert dec.status == "INCOMPLETE"
    assert len(dec.unsupported_criteria) > 0
    assert any("moisture_content" in c for c in dec.unsupported_criteria)
