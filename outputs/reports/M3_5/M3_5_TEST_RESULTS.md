# RiceAge M3.5 — Test Execution & Regression Audit Report

**Audit Date:** 2026-10-12  
**Auditor Role:** Senior Software Test Engineer & Quality Assurance Lead  
**Test Framework:** Pytest 9.1.1, Python 3.13.13, Linux x86_64  
**Command:** `.venv/bin/pytest -v`

---

## 1. Test Suite Summary

```
============================== 61 passed in 5.84s ==============================
```

- **Total Test Cases:** 61
- **Passed:** 61 (100%)
- **Failed:** 0 (0%)
- **Skipped:** 0 (0%)
- **Pre-Existing Tests:** 53
- **Newly Added Audit Tests:** 8 (in `tests/test_m3_5_audit.py`)

> [!NOTE]
> A passing test suite verifies deterministic software mechanics (correct error handling, parameter constraints, denominator accounting, and boundary handling). As emphasized throughout this audit, passing unit tests do **not** equate to real-world computer vision accuracy on unseen physical grains or altered illumination.

---

## 2. Test Suite Breakdown by Module

| Test Module | Scope / Component Under Test | Tests Executed | Passed | Failed |
| :--- | :--- | :---: | :---: | :---: |
| `tests/test_calibration.py` | DPI metadata resolution & scale derivation | 4 | 4 | 0 |
| `tests/test_classification.py` | M0 baseline rule classification & anchor length | 2 | 2 | 0 |
| `tests/test_dataset.py` | Dataset validation, filtering & instance masks | 4 | 4 | 0 |
| `tests/test_evaluation.py` | Pairwise IoU & instance evaluation matrices | 2 | 2 | 0 |
| `tests/test_evaluation_audit.py` | Hungarian matching strictness & split leakage | 5 | 5 | 0 |
| `tests/test_measurement.py` | Grain morphology and color extraction | 2 | 2 | 0 |
| `tests/test_pipeline_integration.py`| Baseline pipeline on synthetic & sample fixtures | 2 | 2 | 0 |
| `tests/test_preprocessing.py` | Margin detection, bezel clearing, boundary flags | 5 | 5 | 0 |
| `tests/test_quality_classification.py`| M3 feature extraction & supervised head fitting | 3 | 3 | 0 |
| `tests/test_quality_grading.py` | Sample metrics, denominator rules, Codex/USDA | 6 | 6 | 0 |
| `tests/test_quality_reporting.py` | HTML rendering & tripartite file export | 2 | 2 | 0 |
| `tests/test_segmentation.py` | Otsu binarization, trimodal guard, cleanup | 5 | 5 | 0 |
| `tests/test_separation.py` | Watershed clustering & concave neck-cutting | 5 | 5 | 0 |
| `tests/test_wb_classification.py` | M2 Whole/Broken rules, calibration & locking | 6 | 6 | 0 |
| **`tests/test_m3_5_audit.py` (NEW)** | **Targeted M3.5 audit & edge-case regression** | **8** | **8** | **0** |
| **Total** | | **61** | **61** | **0** |

---

## 3. Detailed Audit of Newly Added Regression Tests

The eight tests added in [tests/test_m3_5_audit.py](file:///home/anubhavtripathi/Documents/RiceAge/tests/test_m3_5_audit.py) directly target the edge-case failure modes discovered during the M3.5 audit:

### 3.1 `test_feature_extraction_empty_and_zero_crops`
- **Objective:** Verify that feature extraction does not crash or raise uncaught division-by-zero exceptions on corrupted 0-area masks or empty image crops.
- **Verification:** Safely returns `None` and rejects components with area $< 10\text{ px}$.

### 3.2 `test_feature_extraction_low_contrast_and_noise`
- **Objective:** Verify feature stability when an ellipse is placed on low-contrast background with additive Gaussian noise ($\sigma = 3$).
- **Verification:** Length, area, solidity, and photometric values remain finite and within valid physical ranges.

### 3.3 `test_feature_extraction_dark_borders`
- **Objective:** Verify that scanner glass margins or dark bezel artifacts within a crop do not skew aspect ratio or solidity.
- **Verification:** Solidity remains $> 0.90$ with zero NaN/Inf outputs.

### 3.4 `test_multiple_defects_do_not_inflate_rice_denominator`
- **Objective:** Verify multi-defect rice grain accounting. When physical grains possess multiple coexisting defects (e.g. Broken + Damaged + Discoloured), ensure they are not double-counted in the sample denominator.
- **Verification:**
  $$\text{total\_rice\_grains} = \text{whole\_count} + \text{broken\_count}$$
  Foreign matter is strictly excluded from the grain denominator. Multi-defect counter increments without inflating grain count.

### 3.5 `test_exact_grade_boundary_comparison`
- **Objective:** Verify threshold boundary comparisons (inclusive $\le$ vs exclusive $>$).
- **Verification:** Sample with exactly 10.0% broken grains passes Codex Grade 1 ($\le 10.0\%$), while sample with 11.0% broken drops to Grade 2 ($\le 15.0\%$).

### 3.6 `test_missing_non_optical_measurements_marks_incomplete`
- **Objective:** Verify that unverified regulatory factors (moisture content, olfactory defects) prevent unconditional legal certification.
- **Verification:** `require_full_criteria=True` produces status `INCOMPLETE`, while optical screening mode produces status `DECIDED` with mandatory audit caveats.

### 3.7 `test_report_export_reconciles_across_all_three_formats`
- **Objective:** Verify cross-format reconciliation across HTML, JSON, and CSV exports.
- **Verification:** Object counts, broken counts, defect percentages, and sample IDs match identically across all three generated files.

### 3.8 `test_inference_operates_without_ground_truth`
- **Objective:** Verify strict isolation of the feature extractor and classification models from ground-truth label access at inference time.
- **Verification:** Extracted features contain zero ground-truth label references. Classifiers predict strictly on numeric morphological and photometric vectors.
