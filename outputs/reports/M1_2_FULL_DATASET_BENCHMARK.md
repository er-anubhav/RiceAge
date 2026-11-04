# RiceAge M1.2: Full Dataset Regression Benchmark & Evaluation Validation

**Document Version:** 1.0.0  
**Benchmark Date:** 2026-10-10  
**Lead Evaluator:** Senior Computer Vision Engineer, Google DeepMind Antigravity Team  
**Evaluation Target:** RiceAge Classical Computer Vision Engine (`src/pipeline.py`, `src/segmentation.py`, `src/separation.py`, `src/evaluation.py`)  
**Dataset Path:** `data/dataset/` (119 scanner images, 2,529 ground-truth instances)  
**Artifact Dependencies:**
- Baseline Artifacts: `outputs/evaluation_full_hybrid/`, `outputs/evaluation_full_watershed/`
- Current M1.2 Artifacts: `outputs/evaluation_m1_2_hybrid/`, `outputs/evaluation_m1_2_watershed/`
- Structured Per-Image Data: `outputs/reports/M1_2_PER_IMAGE_RESULTS.csv`
- Structured Per-Class Data: `outputs/reports/M1_2_PER_CLASS_RESULTS.csv`
- Regression Analysis: `outputs/reports/M1_2_REGRESSION_CASES.md`

---

## 1. Executive Summary

Milestone M1.2 executes the first complete, rigorous regression benchmark across the full 119-image annotated scanner dataset following the integration of the Otsu foreground sanity guard (`max_fg_ratio = 0.20`) and background noise-floor clamping (`noise_floor = np.percentile(bg, 99.9)`).

### Core Benchmark Outcomes

1. **100% Processing Reliability Across Full Dataset:**
   All **119 images (100.0%)** and **2,529 ground-truth grain instances** across all six condition categories were ingested, preprocessed, segmented, separated, measured, classified, and matched with **0 fatal runtime crashes, 0 skipped images, and 0 memory faults**.
2. **Resolution of Otsu Trimodal Collapse:**
   The scanner edge artifact on `Damaged 6.jpg` was completely resolved without manual cropping or image exclusion. Predicted false positives dropped from 108 (or 235 unclamped) to **2**, and True Positives jumped from 0 to **19 / 19 (100% recall)** with **90.48% precision** ($F_1 = 0.9500$).
3. **Zero Regression Across All 118 Other Scans:**
   A pairwise regression comparison between the preserved baseline and current code confirmed that **exactly zero images suffered metric regression** ($0.00\%$ regression rate). All 118 other scans produced identical detection, count, and matching values.
4. **Substantial Global Metric Improvements:**
   - **Micro Precision:** Increased from **78.21%** to **81.15%** (+2.94% absolute gain).
   - **Micro Recall:** Increased from **94.07%** to **94.82%** (+0.75% absolute gain).
   - **Micro F1 Score:** Increased from **85.41%** to **87.45%** (+2.04% absolute gain).
   - **Macro Precision:** Increased from **87.07%** to **87.83%** (+0.76% absolute gain).
   - **Macro Recall:** Increased from **94.60%** to **95.44%** (+0.84% absolute gain).
   - **Mean Absolute Count Error:** Decreased from **5.61** to **4.87 grains / image** (13.2% error reduction).
5. **Separation Engine Trade-off Clarified:**
   - On **rice kernels**, Hybrid separation successfully recovered **+5 touching pairs** (under-segmented merges reduced from 5 to 0) with 0 false splits.
   - On **non-grain foreign matter**, concave-neck cutlines split irregular debris into fragments (+26 false positive pieces), giving Pure Watershed higher precision on foreign matter (47.87% vs 45.96%) but lower recall on rice kernels (95.24% vs 95.44%).

---

## 2. Baseline vs. Current M1.2 Comprehensive Comparison

The table below directly benchmarks the preserved baseline against the current M1.2 pipeline under identical Hungarian bipartite containment matching ($\tau_{\text{containment}} = 0.50$):

| Evaluation Dimension | Metric | Preserved Baseline (M1.0) | Current M1.2 Benchmark | Absolute Delta | Relative Change |
|---|---|---|---|---|---|
| **Coverage** | Total Images Evaluated | 119 / 119 | **119 / 119** | 0 | 100.0% coverage |
| | Ground-Truth Grains | 2,529 | **2,529** | 0 | Invariant |
| | Predicted Grains (Hybrid) | 3,042 | **2,955** | -87 | -2.86% (Noise eliminated) |
| | Predicted Grains (Watershed) | 3,004 | **2,918** | -86 | -2.86% (Noise eliminated) |
| **Instance Counting** | Mean Absolute Count Error | 5.61 grains/image | **4.87 grains/image** | **-0.74** | **-13.19% error** |
| | Mean Relative Count Error | 29.07% | **25.22%** | **-3.85%** | **-13.24% error** |
| **Micro-Averaged Metrics** | True Positives (Micro) | 2,379 | **2,398** | **+19** | **+0.80% TP** |
| *(Global Dataset Sum)* | False Positives (Micro) | 663 | **557** | **-106** | **-15.99% FP** |
| | False Negatives (Micro) | 150 | **131** | **-19** | **-12.67% FN** |
| | **Micro Precision** | **78.21%** | **81.15%** | **+2.94%** | **+3.76% rel** |
| | **Micro Recall** | **94.07%** | **94.82%** | **+0.75%** | **+0.80% rel** |
| | **Micro F1 Score** | **85.41%** | **87.45%** | **+2.04%** | **+2.39% rel** |
| **Macro-Averaged Metrics** | **Macro Precision** | **87.07%** | **87.83%** | **+0.76%** | **+0.87% rel** |
| *(Per-Image Mean)* | **Macro Recall** | **94.60%** | **95.44%** | **+0.84%** | **+0.89% rel** |
| | **Macro F1 Score** | **89.03%** | **89.83%** | **+0.80%** | **+0.90% rel** |
| **Geometry** | Mean Matched IoU | 0.3059 | **0.3082** | +0.0023 | Bounded by 4-pt OBB |
| **Clustering Diagnostics** | Under-Segmented Merges | 0 | **0** | 0 | 0 merged pairs in Hybrid |
| | Over-Segmented Splits | 18 | **19** | +1 | 1 grain in Damaged 6 |

---

## 3. Dataset Coverage & Category Breakdown

The dataset contains **119 images** across six folders. Exactly one identical duplicate scan exists: `whole 4.jpg` and `whole 5.jpg` (MD5: `b7b43a9b9a674d8bb8568c4a1eb6f8cb`, correlation 1.0000).

### Per-Category Performance Breakdown (M1.2 Hybrid Separation)

| Category | Images | GT Grains | Pred Grains | Abs Error | Micro Prec | Micro Rec | Micro F1 | Macro Prec | Macro Rec | Macro F1 | Mean IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Whole** | 20 | 395 | 395 | **0.00** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.2837 |
| **Broken** | 20 | 398 | 398 | **0.00** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.3048 |
| **Chalky** | 20 | 462 | 402 | 3.00 | **1.0000** | 0.8701 | 0.9306 | **1.0000** | 0.8728 | 0.9314 | 0.3815 |
| **Damaged** | 19 | 371 | 378 | **0.37** | **0.9815** | **1.0000** | **0.9907** | **0.9821** | **1.0000** | **0.9908** | 0.5301 |
| **Mixed** | 20 | 534 | 592 | 4.60 | 0.8074 | 0.8951 | 0.8490 | 0.8336 | 0.8914 | 0.8556 | 0.1869 |
| **Foreign Matter**| 20 | 369 | 790 | 21.05 | 0.4481 | 0.9593 | 0.6109 | 0.4596 | 0.9647 | 0.6168 | 0.1730 |
| **Total / Summary**| **119** | **2,529** | **2,955** | **4.87** | **0.8115** | **0.9482** | **0.8745** | **0.8783** | **0.9544** | **0.8983** | **0.3082** |

---

## 4. Evaluation Methodology & Metric Integrity

### A. Instance Counting & Localization (Containment Matching)
- **Objective Function:** Minimum weight bipartite matching using Hungarian optimization (`scipy.optimize.linear_sum_assignment`):
  $$\min \sum (1.0 - \text{Containment}(j, k)) \quad \text{where} \quad \text{Containment}(j, k) = \frac{|M_{\text{pred}}^{(j)} \cap M_{\text{gt}}^{(k)}|}{|M_{\text{pred}}^{(j)}|}$$
- **Duplicate Penalization:** Verified via [`test_duplicate_predictions_penalized_as_false_positives`](file:///home/anubhavtripathi/Documents/RiceAge/tests/test_evaluation_audit.py#L65-L113). If multiple predictions fall within a single ground-truth bounding box, exactly one is matched as True Positive and the remaining predictions are assigned to False Positives.
- **Threshold Gating:** Pairs with containment $<0.50$ are rejected from True Positives.

### B. Mask Segmentation IoU Limitations
- Over **91.0% (2,301 / 2,529)** of dataset annotations in CVAT XML 1.1 are coarse **4-point oriented bounding boxes (OBBs)**, not tight grain silhouettes.
- For a physical rice kernel modeled as an ellipse of area $\pi a b$ inside an annotator-drawn bounding box of area $8 a b$, the theoretical upper bound on pixel IoU is:
  $$\text{IoU}_{\text{max}} = \frac{\pi a b}{8 a b} \approx \mathbf{0.39}$$
- When evaluated under strict pixel IoU $\ge 0.50$, the pipeline achieves only **1.30% precision and recall** on Whole rice.
- **Important Constraint:** Containment scores evaluate *region-of-interest localization*, not segmentation boundary accuracy. Future instance segmentation models (M5) cannot use CVAT XML polygons as mask ground truth without contour contraction.

### C. Classification Evaluation (Supported vs. Unsupported Domain)
The RiceAge rule-based classifier (`src/classification.py`) evaluates grains solely on geometric dimensions (length ratio and aspect ratio):
- `WHOLE_RICE`: Length $\ge 0.75 \times L_{\text{ref}}$ (or length ratio $\ge 0.75$ and aspect ratio $\ge 2.0$).
- `BROKEN_RICE`: Length $< 0.75 \times L_{\text{ref}}$ or aspect ratio $< 2.0$.

#### Supported 2x2 Confusion Matrix (`Whole` vs `Broken` Ground Truth)
Evaluating exclusively on the 1,009 matched instances with ground-truth labels `Whole` or `Broken`:

| | Predicted `WHOLE_RICE` | Predicted `BROKEN_RICE` | Total Matched | Class Recall |
|---|---|---|---|---|
| **Ground Truth `Whole`** | **455** | 56 | 511 | **89.04%** |
| **Ground Truth `Broken`** | 89 | **409** | 498 | **82.13%** |
| **Total Predicted** | 544 | 465 | 1,009 | **Accuracy: 85.63%** |

- **Whole Precision:** $455 / 544 = \mathbf{83.64\%}$
- **Broken Precision:** $409 / 465 = \mathbf{87.96\%}$
- **Macro Classification F1:** $\mathbf{85.58\%}$

#### Unsupported Defect Categories
The dataset contains 1,520 instances labeled with optical defect conditions:
- `Chalky`: 423
- `Damaged`: 451
- `Foreign Matter`: 461
- `Discoloured`: 177

Because the classical CV pipeline classifies whole chalky or damaged kernels as `WHOLE_RICE` based on their physical length, evaluating the classical pipeline directly on a 6-class confusion matrix is an unfair category mismatch. Defect classification requires spectral/color defect heads (Milestone M3/M5).

---

## 5. Pure Watershed vs. Hybrid Separation Comparison

Both separation engines were benchmarked across the complete 119 images under identical preprocessing, parameters, and containment matching ($\tau = 0.50$):

| Comparison Metric | Pure Watershed Baseline | Hybrid Separation (Current) | Delta (Hybrid vs Watershed) |
|---|---|---|---|
| **Total Images Evaluated** | 119 | 119 | 0 |
| **Total Predicted Grains** | 2,918 | 2,955 | +37 |
| **Micro True Positives** | 2,393 | **2,398** | **+5 (+0.21%)** |
| **Micro False Positives** | **525** | 557 | +32 (+6.10%) |
| **Micro False Negatives** | 136 | **131** | **-5 (-3.68%)** |
| **Micro Precision** | **82.01%** | 81.15% | -0.86% (due to foreign matter) |
| **Micro Recall** | 94.62% | **94.82%** | **+0.20%** |
| **Micro F1 Score** | 87.86% | **87.45%** | -0.41% |
| **Macro Precision** | **88.28%** | 87.83% | -0.45% |
| **Macro Recall** | 95.24% | **95.44%** | **+0.20%** |
| **Macro F1 Score** | **90.06%** | 89.83% | -0.23% |
| **Under-Segmented Merges** | 5 | **0** | **-5 (100% merges resolved)** |
| **Over-Segmented Splits** | **17** | 19 | +2 |

### Engineering Trade-Off Analysis
1. **On Rice Kernels (Chalky, Damaged, Whole, Broken):**
   Hybrid separation is **strictly superior**. It resolved 5 residual touching grain pairs that watershed failed to separate (`chalky 13.jpg`, `chalky 19.jpg`, `Damaged 4.jpg`, `Damaged 9.jpg`, `Damaged 12.jpg`), reducing under-segmented merges from 5 to 0 with **0 false splits on single rice grains**.
2. **On Non-Grain Debris (Foreign Matter):**
   Irregular non-grain debris (twigs, weed husks, chaff) exhibits natural concave boundary notches. The concave-neck cutlines split these irregular debris particles, creating **+26 false positive pieces** across 12 foreign matter images (`foreign 2.jpg`, `foreign 20.jpg`, etc.).
3. **Architectural Recommendation:**
   Do not discard Hybrid separation. Instead, introduce a pre-split aspect ratio and area guardrail ($A > 400\text{ px}, \text{aspect ratio} \in [1.5, 4.0]$) before triggering concave-neck cutlines to prevent fragmenting irregular foreign matter debris.

---

## 6. Validation of the Otsu Segmentation Fix

### A. Guardrail Mechanics
In `src/segmentation.py`:
```python
def segment_otsu(gray_image: np.ndarray, max_fg_ratio: float = 0.20) -> Tuple[np.ndarray, float]:
    thresh_val, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_ratio = float(np.mean(binary > 0))
    if fg_ratio > max_fg_ratio:
        valid_pixels = gray_image[gray_image > 20]
        if len(valid_pixels) > 1000:
            alt_thresh, _ = cv2.threshold(valid_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            bg_pix = valid_pixels[valid_pixels < 130]
            noise_floor = float(np.percentile(bg_pix, 99.9)) if len(bg_pix) > 100 else 100.0
            alt_thresh = max(alt_thresh, noise_floor, 100.0)
            alt_binary = (gray_image > alt_thresh).astype(np.uint8) * 255
            if float(np.mean(alt_binary > 0)) <= max_fg_ratio:
                return alt_binary, float(alt_thresh)
...
```

### B. Validation Across Platter Densities
- **Sparse Platter Validation (20 Grains on A4 Platter):**
  Evaluated on all 40 scans in `whole` and `broken`. Rice kernels occupy $<0.3\%$ of platter area. The guardrail remains dormant (`fg_ratio \approx 0.003 < 0.20`), and standard Otsu functions flawlessly ($100\%$ precision and recall).
- **Dense Platter Validation (734 Grains on Calibration Platter):**
  Evaluated on `sample_1489x2048.jpg`. Kernels occupy $\approx 12\%$ of platter area. The guardrail remains dormant ($12\% < 20\%$), preserving all 734 kernels with 0 threshold interference.
- **Trimodal Platter Validation (`Damaged 6.jpg`):**
  Pitch-black margin triggered `fg_ratio > 0.20`. The guardrail activated deterministically, recomputing Otsu on non-margin pixels and clamping the threshold to $100.0$. All 19 real grains were preserved (100% recall) and 106 false positive specks were suppressed.

---

## 7. Automated Test Suite Health

The automated test suite in `tests/` contains **34 unit, regression, and audit tests**. All 34 tests execute and pass cleanly:

```bash
$ .venv/bin/pytest tests/ -v
============================== test session starts ==============================
platform linux -- Python 3.13.13, pytest-9.1.1, pluggy-1.6.0 -- .venv/bin/python3
rootdir: /home/anubhavtripathi/Documents/RiceAge, configfile: pytest.ini
collected 34 items

tests/test_calibration.py::test_inspect_metadata_valid PASSED            [  2%]
tests/test_calibration.py::test_inspect_metadata_missing PASSED          [  5%]
tests/test_calibration.py::test_resolve_calibration_assumed PASSED       [  8%]
tests/test_calibration.py::test_resolve_calibration_reference_derivation PASSED [ 11%]
tests/test_classification.py::test_classify_grains_rule_based_whole_and_broken PASSED [ 14%]
tests/test_classification.py::test_classify_grains_with_reference_anchor PASSED [ 17%]
tests/test_dataset.py::test_validate_dataset PASSED                      [ 20%]
tests/test_dataset.py::test_load_dataset_and_filtering PASSED            [ 23%]
tests/test_dataset.py::test_dataset_split_no_leakage PASSED              [ 26%]
tests/test_dataset.py::test_rasterize_instance_mask PASSED               [ 29%]
tests/test_evaluation.py::test_compute_pairwise_overlap_matrices PASSED  [ 32%]
tests/test_evaluation.py::test_evaluate_image_instances PASSED           [ 35%]
tests/test_evaluation_audit.py::test_hungarian_one_to_one_matching_strictness PASSED [ 38%]
tests/test_evaluation_audit.py::test_duplicate_predictions_penalized_as_false_positives PASSED [ 41%]
tests/test_evaluation_audit.py::test_sub_threshold_overlap_rejection PASSED [ 44%]
tests/test_evaluation_audit.py::test_iou_vs_containment_semantics_on_bounding_boxes PASSED [ 47%]
tests/test_evaluation_audit.py::test_dataset_split_no_duplicate_leakage PASSED [ 50%]
tests/test_measurement.py::test_extract_grain_morphology PASSED          [ 52%]
tests/test_measurement.py::test_extract_grain_color PASSED               [ 55%]
tests/test_pipeline_integration.py::test_pipeline_on_synthetic_fixture PASSED [ 58%]
tests/test_pipeline_integration.py::test_pipeline_baseline_sample_execution PASSED [ 61%]
tests/test_preprocessing.py::test_detect_scanner_margins_clean PASSED    [ 64%]
tests/test_preprocessing.py::test_detect_scanner_margins_with_bezel PASSED [ 67%]
tests/test_preprocessing.py::test_clear_scanner_bezel PASSED             [ 70%]
tests/test_preprocessing.py::test_is_boundary_grain PASSED               [ 73%]
tests/test_preprocessing.py::test_preprocess_image PASSED                [ 76%]
tests/test_segmentation.py::test_segment_otsu PASSED                     [ 79%]
tests/test_segmentation.py::test_segment_otsu_trimodal_collapse_guard PASSED [ 82%]
tests/test_segmentation.py::test_fill_binary_holes PASSED                [ 85%]
tests/test_segmentation.py::test_remove_small_objects PASSED             [ 88%]
tests/test_segmentation.py::test_cleanup_mask PASSED                     [ 91%]
tests/test_separation.py::test_evaluate_component_geometry_single PASSED [ 94%]
tests/test_separation.py::test_detect_concave_points_and_split PASSED    [ 97%]
tests/test_separation.py::test_separate_touching_grains_isolated_preserved PASSED [100%]

============================== 34 passed in 7.82s ==============================
```

---

## 8. Limitations & Recommended Next Steps

1. **Benchmark Nature (Diagnostic Evaluation):**
   Because the dataset is evaluated in full without a separate held-out split, this benchmark serves as a **reproducible diagnostic regression baseline**, not an unbiased test-set generalization estimate.
2. **Debris Over-Segmentation Mitigation:**
   Constrain concave-neck cutlines in `src/separation.py` to trigger only on components with grain-like area ($>400\text{ px}$) and aspect ratio ($\ge 1.5$) to prevent splitting irregular chaff and husks.
3. **Physical Tray Validation (Milestone M1):**
   Proceed to **Milestone M1 (Physical Reality Tray Validation)** using hand-counted physical grain trays (100-grain and 250-grain samples) and micrometer-measured kernels to anchor physical millimeter accuracy.
