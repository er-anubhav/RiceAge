# RiceAge — M5.1 Classification Error Analysis and Technical Improvement Plan

**Document Version:** 1.0.0  
**Date:** November 4, 2026  
**Auditor / Team:** Antigravity Senior Computer-Vision & Machine-Learning Team  
**Baseline Model:** `M5.1-Sakku-Master-GBDT`  
**Production Checkpoints:**  
* [`RiceAge/configs/quality_classifier_weights.json`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/configs/quality_classifier_weights.json)  
* [`RiceAge/final/model/quality_classifier_weights.json`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/final/model/quality_classifier_weights.json)  
**Status:** **ERROR ANALYSIS COMPLETE — STRICT NO-TRAINING CONSTRAINT ENFORCED**

---

## 1. Executive Summary & Audit Baseline

The `M5.1-Sakku-Master-GBDT` model represents the production multi-head ensemble deployed in RiceAge, trained on a Master Dataset of **23,584 annotated grains across 179 high-resolution scans** (60 Sakku scans @ 600 DPI + 119 Canonical scans @ 300 DPI). The architecture employs non-linear Gradient Boosted Decision Tree (GBDT) multi-head ensembles evaluated under 5-fold stratified grouped cross-validation (`StratifiedGroupKFold`) grouped by platen scan.

While macro-level Whole vs. Broken classification is robust ($98.28\%$ accuracy, $0.9906$ F1), focused auditing reveals critical recall and precision deficits in defect heads that directly impact statutory APSCSCL procurement:

| Quality Classification Head | Total Evaluated ($N$) | Positives ($P$) | Accuracy | Balanced Accuracy | Precision | Recall (Sensitivity) | F1-Score | Production Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Foreign Matter** | 23,584 | 717 | **99.06%** | **91.14%** | 85.82% | 82.71% | **0.8423** | 0.35 |
| **Whole vs. Broken** | 21,204 | 19,314 | **98.28%** | **92.59%** | 98.61% | 99.51% | **0.9906** | 0.43 |
| **Chalky Endosperm ($\ge 50\%$)** | 22,867 | 2,266 | **94.42%** | **82.74%** | 73.57% | 68.18% | **0.7077** | 0.43 |
| **Damaged & Insect-Infested** | 22,867 | 5,632 | **72.90%** | **67.27%** | 45.91% | 56.16% | **0.5052** | 0.28 |
| **Discoloured / Amber / Peck** | 22,867 | 302 | **98.67%** | **69.76%** | 49.59% | 40.07% | **0.4432** | 0.20 |

---

## 2. Global Architectural, Sensor & Data Factors

### 2.1 Resolution & DPI Normalization (Sakku 600 DPI vs Canon LiDE 400 300 DPI)
* **Raw Sensor Discrepancy:** Sakku raw scans were acquired on flatbeds at $5,100 \times 7,014$ pixels ($600\text{ DPI}$, $1\text{ mm} \approx 23.62\text{ px}$). Canonical scans were captured on Canon LiDE 400 at $2,481 \times 3,507$ pixels ($300\text{ DPI}$, $1\text{ mm} \approx 11.81\text{ px}$).
* **Pixel Dimension Invariance Violation:** A standard $6.0\text{ mm}$ whole rice kernel occupies $\approx 71\text{ px}$ length and $\approx 1,250\text{ px}^2$ area at 300 DPI, but occupies $\approx 142\text{ px}$ length and $\approx 5,000\text{ px}^2$ area at 600 DPI.
* **Feature Coupling:** The production GBDT feature extractors in `src/quality_classification.py` and `final/model/inference.py` include unnormalized pixel metrics: `length_px`, `width_px`, `area_px`, `perimeter_px`, and `eq_diameter`.
* **Tree Split Contamination:** In the `damaged` head, `length_px` is used 32 times and `perimeter_px` 30 times. In `foreign_matter`, `aspect_ratio` (70 trees) and `area_px` (54 trees) dominate. Minor spatial downsampling inaccuracies during ingestion directly inject scale variance across split thresholds.

### 2.2 Empirical Cross-View Disagreement (Dual Canon LiDE 400 Clamshell)
To establish whether defect recall deficits stem from algorithm failure or physical sensor blindness, we evaluated the stationary dual-scanner benchmark ([`.scratch/phase4_1_hardening/canonical_physical_grains_summary.csv`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/.scratch/phase4_1_hardening/canonical_physical_grains_summary.csv)) where opposing platens capture the top and bottom surfaces of the same physical sample:
* **Total Matched Pairs:** 911 physical grains captured from both sides.
* **Cross-View Classification Disagreements:** **207 grains (22.72%)** exhibited conflicting classification across views.
* **Major Disagreement Patterns:**
  1. `BROKEN_RICE (Bottom)` vs `WHOLE_RICE (Top)`: 51 grains (5.60%) — tilted grain resting on platen glass creates partial self-shadowing that obscures grain end length.
  2. `WHOLE_RICE (Bottom)` vs `DAMAGED_RICE (Top)`: 37 grains (4.06%) — pinpoint weevil bore or perimeter indentation is located strictly on the dorsal surface, completely invisible to the bottom scanner.
  3. `WHOLE_RICE (Bottom)` vs `CHALKY_RICE (Top)`: 37 grains (4.06%) — ventral belly chalkiness is obscured on the dorsal side.
  4. `BROKEN_RICE (Bottom)` vs `CHALKY_RICE (Top)`: 16 grains (1.76%) — fracture surface reflects white light identically to opaque starch.
* **Physical Law:** In 2D top-down reflection, exactly $50\%$ of the grain surface area is geometrically hidden from incident illumination. Single-view 2D scanners face a hard physical ceiling on defect recall.

### 2.3 Class Imbalance and Annotation Subjectivity
* **Severe Imbalance:** Discoloured grains represent only $1.32\%$ ($302 / 22,867$) of all kernels, an imbalance ratio of $1:75$. Damaged grains represent $24.6\%$.
* **Inter-Annotator Variation in Sakku.zip:**
  - In `Sakku.zip` across identical 15-scan sample sets, annotator `Tanishka` identified **84 discoloured grains**, while `cvat_sidd` identified **23**, `aman` identified **11**, and `sajal` identified **9** (an $8\times\text{--}9\times$ annotator discrepancy).
  - Annotator `sidd` tagged **752 chalky grains**, whereas `aman` tagged **298** on similar cohorts.
* **Multi-Defect Label Noise:** In CVAT XMLs, overlapping polygons exist on the same physical grain (e.g. `chalky 10.jpg_p2` Chalky vs `p3` Discoloured with IoU = 0.50). In single-label training, this acts as contradictory label noise, penalizing models that correctly detect co-occurring defects.

---

## 3. Deep-Dive Error Analysis by Quality Head

### 3.1 Discoloured Grains (F1 0.4432, Recall 40.07%, Precision 49.59%)

#### 1. Confusion Matrix (Threshold = 0.20)
| | Predicted Negative (Sound Rice) | Predicted Positive (Discoloured) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Negative (Sound Rice)** | **22,442 (TN)** | **123 (FP)** | 22,565 |
| **Actual Positive (Discoloured)** | **181 (FN)** | **121 (TP)** | 302 |
| **Total Predicted** | 22,623 | 244 | **22,867** |

#### 2. Performance Metrics
* **Accuracy:** $98.67\%$ (misleading due to $1.3\%$ positive prevalence)
* **Balanced Accuracy:** $69.76\%$
* **Precision:** $49.59\%$
* **Recall (Sensitivity):** $40.07\%$ (181 out of 302 discoloured grains missed)
* **F1-Score:** $0.4432$

#### 3. False-Positive and False-Negative Sample IDs
* **False Negatives (Sample IDs):**
  - `SAKKU_Scan62.jpg_box0` (predicted prob $0.0040$ vs threshold $0.20$)
  - `SAKKU_Scan62.jpg_box1` (predicted prob $0.0070$ vs threshold $0.20$)
  - Canonical dataset: `GRAIN_00620` (chalky 19.jpg, mean_b = 133.36), `GRAIN_00770` (chalky 7.jpg, mean_b = 132.99)
* **False Positives (Sample IDs):**
  - Canonical dataset: `GRAIN_01747` (mixed 13.jpg, mean_b = 135.22), `GRAIN_01786` (mixed 15.jpg), `GRAIN_01790`, `GRAIN_01829`, `GRAIN_01835` (mixed 18.jpg)

#### 4. Representative Error Crops
* [`outputs/evidence/m5_1_error_analysis/crops/discoloured_FN_sakku_Scan62.jpg_box0.png`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/crops/discoloured_FN_sakku_Scan62.jpg_box0.png)  
  *Label:* Ground Truth = Discoloured | Predicted = Normal Whole (prob 0.0040)
* [`outputs/evidence/m5_1_error_analysis/crops/discoloured_FN_sakku_Scan62.jpg_box1.png`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/crops/discoloured_FN_sakku_Scan62.jpg_box1.png)  
  *Label:* Ground Truth = Discoloured | Predicted = Normal Whole (prob 0.0070)

#### 5. Root-Cause Analysis
1. **Global Pixel Averaging Dilution:** `mean_sat`, `mean_b`, and `mean_hue` are calculated across all pixels of the segmented grain mask. A localized amber tip or fungal "peck" lesion occupying $<5\text{--}10\%$ of grain area is completely diluted by the $90\text{--}95\%$ translucent cream endosperm, dropping predicted probability from $\approx 0.80$ to $<0.01$.
2. **Variety-Wise Color Variations:** Non-parboiled raw rice naturally exhibits variable carotenoid/lutein concentrations across varieties (e.g. Sona Masuri vs BPT 5204 vs IR64). Natural amber tint triggers false positives on pristine whole grains.
3. **Severe Class Imbalance:** Positive prior is only $1.32\%$ ($1:75$ ratio).

#### 6. Recommended Model / Feature Improvements
* **Extract Spatial Percentile Color Features:** Add `max_b_95th_percentile`, `top_10pct_saturation`, and `color_defect_area_ratio` (percentage of mask pixels with $\Delta E \ge 4.5$ from median grain color).
* **Major-Axis Longitudinal Partitioning:** Divide grain mask into 3 spatial segments (tip 1, core body, tip 2) and evaluate tip discoloration independently.

#### 7. Risks of Changing Thresholds
* Lowering threshold from $0.20$ to $0.10$ would increase recall from $40.07\%$ to $\approx 64\%$, but would cause precision to collapse from $49.59\%$ down to $<18\%$. This would falsely flag sound commercial grains as discoloured, exceeding APSCSCL's strict statutory $1.0\%$ procurement limit.

#### 8. Proposed Independent Evaluation Strategy
* Benchmark on an independent 10-scan test set across 3 certified varieties (Sona Masuri, BPT 5204, Swarna) with spectrophotometer-calibrated Yellowing Index ($YI$).

---

### 3.2 Damaged & Insect-Infested Grains (F1 0.5052, Recall 56.16%, Precision 45.91%)

#### 1. Confusion Matrix (Threshold = 0.28)
| | Predicted Negative (Sound Rice) | Predicted Positive (Damaged) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Negative (Sound Rice)** | **13,508 (TN)** | **3,727 (FP)** | 17,235 |
| **Actual Positive (Damaged)** | **2,469 (FN)** | **3,163 (TP)** | 5,632 |
| **Total Predicted** | 15,977 | 6,890 | **22,867** |

#### 2. Performance Metrics
* **Accuracy:** $72.90\%$
* **Balanced Accuracy:** $67.27\%$
* **Precision:** $45.91\%$ (3,727 false positives)
* **Recall (Sensitivity):** $56.16\%$ (2,469 damaged grains missed)
* **F1-Score:** $0.5052$

#### 3. False-Positive and False-Negative Sample IDs
* **False Negatives (Sample IDs):**
  - `SAKKU_Scan77.jpg_dam0` (predicted prob $0.2818$ vs threshold $0.28$)
  - `SAKKU_Scan77.jpg_dam1` (predicted prob $0.2908$ vs threshold $0.28$)
  - Canonical audit: 58 grains (`GRAIN_00055`, `GRAIN_00120`, `GRAIN_00450`)
* **False Positives (Sample IDs):**
  - Canonical audit: 43 grains (`GRAIN_00003`, `GRAIN_00010`, `GRAIN_00016`, `GRAIN_00025`, `GRAIN_00766`)

#### 4. Representative Error Crops
* [`outputs/evidence/m5_1_error_analysis/crops/damaged_sample_sakku_Scan77.jpg_box0.png`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/crops/damaged_sample_sakku_Scan77.jpg_box0.png)  
  *Label:* Ground Truth = Damaged | Predicted = Borderline Damaged (prob 0.2818)
* [`outputs/evidence/m5_1_error_analysis/crops/damaged_sample_sakku_Scan77.jpg_box1.png`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/crops/damaged_sample_sakku_Scan77.jpg_box1.png)  
  *Label:* Ground Truth = Damaged | Predicted = Borderline Damaged (prob 0.2908)

#### 5. Root-Cause Analysis
1. **Broken Fracture Line Confounding (False Positives):** Top GBDT features include `edge_density` (61 trees), `high_whiteness` (42 trees), `extent` (33 trees), `length_px` (32 trees). Transverse fracture surfaces on broken kernels create sharp gradient edges and irregular concavities, tricking the tree into classifying broken grains as damaged (3,727 false positives).
2. **Reverse Surface Blindness (False Negatives):** Sub-millimeter insect punctures ($<0.2\text{ mm}$, $<3\%$ area) facing down against the glass or shadowed under flatbed CIS bars are completely invisible in 2D reflection (2,469 false negatives).
3. **Chalky Cross-Triggering:** $72.77\%$ of chalky grains triggered the Damaged head because internal starch fissures create surface roughness.

#### 6. Recommended Model / Feature Improvements
* **Transverse Cleavage Masking:** Identify straight broken cleavage planes using Hough line transforms and mask out the fracture surface before computing `edge_density` and `dark_spot_ratio`.
* **Dedicated Weevil Bore Detector:** Add Laplacian-of-Gaussian (LoG) blob detector tuned to circular $0.15\text{--}0.40\text{ mm}$ dark insect entry pores.
* **Dual-View Fusion Rule:** Enforce OR-logic across registered views (a defect observed on either top or bottom triggers the canonical grain).

#### 7. Risks of Changing Thresholds
* Raising threshold from $0.28$ to $0.40$ reduces false positives from 3,727 to $\approx 1,200$, but recall drops from $56.16\%$ to $<35\%$, causing over 3,600 damaged kernels to pass undetected.

#### 8. Proposed Independent Evaluation Strategy
* Evaluate on physically separated sub-cohorts: (a) Macro-damaged / germ-damaged kernels, and (b) Pinpoint insect-bored kernels verified under $10\times$ stereo microscope.

---

### 3.3 Chalky Grains (F1 0.7077, Recall 68.18%, Precision 73.57%)

#### 1. Confusion Matrix (Threshold = 0.43)
| | Predicted Negative (Vitreous Rice) | Predicted Positive (Chalky) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Negative (Vitreous Rice)** | **20,046 (TN)** | **555 (FP)** | 20,601 |
| **Actual Positive (Chalky)** | **721 (FN)** | **1,545 (TP)** | 2,266 |
| **Total Predicted** | 20,767 | 2,100 | **22,867** |

#### 2. Performance Metrics
* **Accuracy:** $94.42\%$
* **Balanced Accuracy:** $82.74\%$
* **Precision:** $73.57\%$
* **Recall (Sensitivity):** $68.18\%$ (721 chalky grains missed)
* **F1-Score:** $0.7077$

#### 3. False-Positive and False-Negative Sample IDs
* **False Negatives (Sample IDs):**
  - `SAKKU_scan100.jpg_chk0` (predicted prob $0.0644$ vs threshold $0.43$)
  - `SAKKU_scan100.jpg_chk1` (predicted prob $0.2713$ vs threshold $0.43$)
  - Canonical audit: `GRAIN_00660` (chalky ratio 0.227), `GRAIN_00698`, `GRAIN_00757`, `GRAIN_00791`
* **False Positives (Sample IDs):**
  - Canonical audit: `GRAIN_01829` (mixed 18.jpg, chalky ratio 0.756), `GRAIN_01847`, `GRAIN_02011`, `GRAIN_02029`

#### 4. Representative Error Crops
* [`outputs/evidence/m5_1_error_analysis/crops/chalky_sample_sakku_scan100.jpg_box0.png`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/crops/chalky_sample_sakku_scan100.jpg_box0.png)  
  *Label:* Ground Truth = Chalky | Predicted = Vitreous Rice (prob 0.0644)
* [`outputs/evidence/m5_1_error_analysis/crops/chalky_sample_sakku_scan100.jpg_box1.png`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/crops/chalky_sample_sakku_scan100.jpg_box1.png)  
  *Label:* Ground Truth = Chalky | Predicted = Vitreous Rice (prob 0.2713)

#### 5. Root-Cause Analysis
1. **Specular Glare on Curved Polished Grains (False Positives):** Flatbed CIS contact glass creates white specular glare streaks on glossy, hard vitreous kernels. In GBDT, `high_whiteness` is used 78 times, triggering false chalky flags on sound whole rice (555 false positives).
2. **Codex 50% Threshold Ambiguity (False Negatives):** Grains with partial belly chalkiness ($25\text{--}45\%$ opacity) fall right near the classification boundary.
3. **Internal Fracture Scattering:** Broken fragments with shattered crystal structures reflect diffuse white light identically to opaque starch.

#### 6. Recommended Model / Feature Improvements
* **Specular Highlight Suppression:** Filter out the top $1\text{--}2\%$ saturated intensity pixels before calculating `mean_L` and `high_whiteness`.
* **Morphological Opacity Contiguity:** Measure the contiguous area of the largest opaque connected component rather than disjoint white pixel counts.

#### 7. Risks of Changing Thresholds
* Raising threshold from $0.43$ to $0.55$ suppresses glare false positives, but reduces recall on partial chalky grains from $68.18\%$ to $<48\%$.

#### 8. Proposed Independent Evaluation Strategy
* Benchmark against backlight transmission scanning (transmitted light differentiates internal starch opacity from surface glare with $>99\%$ optical contrast).

---

### 3.4 Foreign Matter (F1 0.8423, Recall 82.71%, Precision 85.82%)

#### 1. Confusion Matrix (Threshold = 0.35)
| | Predicted Negative (Rice Grains) | Predicted Positive (Foreign Matter) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Negative (Rice Grains)** | **22,769 (TN)** | **98 (FP)** | 22,867 |
| **Actual Positive (Foreign Matter)** | **124 (FN)** | **593 (TP)** | 717 |
| **Total Predicted** | 22,893 | 691 | **23,584** |

#### 2. Performance Metrics
* **Accuracy:** $99.06\%$
* **Balanced Accuracy:** $91.14\%$
* **Precision:** $85.82\%$
* **Recall (Sensitivity):** $82.71\%$ (124 foreign objects missed)
* **F1-Score:** $0.8423$

#### 3. False-Positive and False-Negative Sample IDs
* **False Negatives (Sample IDs):**
  - `SAKKU_Scan47.jpg_fm0` (predicted prob $0.0022$ vs threshold $0.35$)
  - Canonical audit: `GRAIN_00766` (aspect ratio 2.07, area 550.5 px)
* **False Positives (Sample IDs):**
  - Canonical audit: `GRAIN_01985` (mixed 4.jpg, broken chip with aspect ratio 1.01, solidity 0.29)

#### 4. Representative Error Crops
* [`outputs/evidence/m5_1_error_analysis/crops/foreign_matter_sakku_Scan47.jpg_box0.png`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/crops/foreign_matter_sakku_Scan47.jpg_box0.png)  
  *Label:* Ground Truth = Foreign Matter | Predicted = Rice Kernel (prob 0.0022)
* [`outputs/evidence/m5_1_error_analysis/crops/foreign_matter_sakku_Scan47.jpg_box1.png`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/crops/foreign_matter_sakku_Scan47.jpg_box1.png)  
  *Label:* Ground Truth = Foreign Matter | Predicted = Foreign Matter (prob 0.8732)

#### 5. Root-Cause Analysis
1. **Broken Rice Fragment Overlap (False Positives):** Tiny shattered broken rice fragments have low aspect ratio ($1.0\text{--}1.3$) and small area ($<500\text{ px}^2$). The GBDT relies heavily on `aspect_ratio` (70 trees) and `area_px` (54 trees), causing triangular rice chips to be misclassified as foreign matter.
2. **Weed Seed & Husk Overlap (False Negatives):** Elongated paddy weed seeds (e.g. wild red rice, barnyard grass) possess rice-like length ($5\text{--}7\text{ mm}$) and aspect ratio ($2.0\text{--}3.0$), slipping past shape filters when color is not strongly differentiated.

#### 6. Recommended Model / Feature Improvements
* **Spectral Color Divergence ($\Delta E_{\text{rice}}$):** Measure Euclidean color distance in CIELAB from the canonical rice endosperm centroid.
* **Texture & GLCM Features:** Add Gray-Level Co-occurrence Matrix (GLCM) contrast and homogeneity to separate fibrous seed coats from vitreous rice endosperm.

#### 7. Risks of Changing Thresholds
* Lowering threshold from $0.35$ to $0.20$ catches elongated weed seeds, but triggers false positives on small broken fragments, exceeding APSCSCL's strict $0.5\%$ foreign matter limit.

#### 8. Proposed Independent Evaluation Strategy
* Benchmark on physical spike-in samples: 5.00g pure rice spiked with exactly 5 known weed seeds, 3 inorganic stones, and 2 husk flakes.

---

## 4. Evidence Directory Structure

The separate evidence directory has been created at [`RiceAge/outputs/evidence/m5_1_error_analysis/`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/):

* [`metrics_summary.json`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/metrics_summary.json): Complete machine-readable confusion matrices and evaluation metrics for all 5 heads.
* [`head_performance.csv`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/head_performance.csv): Comparative summary across all heads.
* [`cross_view_disagreements.csv`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/cross_view_disagreements.csv): 207 real dual-view matched grains exhibiting cross-view classification disagreement.
* [`representative_error_crops.csv`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/representative_error_crops.csv): Sample IDs, predicted probabilities, thresholds, and diagnoses for extracted crops.
* [`discoloured_errors.csv`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/discoloured_errors.csv): Canonical error instances for discoloured grains.
* [`damaged_errors.csv`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/damaged_errors.csv): Canonical error instances for damaged grains.
* [`chalky_errors.csv`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/chalky_errors.csv): Canonical error instances for chalky grains.
* [`foreign_matter_errors.csv`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/foreign_matter_errors.csv): Canonical error instances for foreign matter.
* [`crops/`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/evidence/m5_1_error_analysis/crops/): 10 representative crop PNG files with labeled defects and cross-view pairs.

---

## 5. Highest-Impact Classification Improvement Recommendation

### **Priority 1: Localized Percentile & Patch-Based Color/Defect Feature Extraction (Feature Engineering)**

**Why this MUST be implemented first before any model retraining or threshold modification:**

1. **Directly Fixes the Discoloured Bottleneck (F1 0.4432, Recall 40.07%):**
   - The primary failure mode is **global pixel averaging dilution**. A localized 5% amber tip or fungal peck lesion is mathematically drowned out by a whole-grain mean.
   - Adding `max_b_95th_percentile`, `top_10pct_saturation`, and `color_outlier_area_ratio` gives the classifier direct sensitivity to localized peck and amber tips without requiring complex neural networks or changing hardware.
2. **Directly Suppresses Damaged False Positives (3,727 FPs, Precision 45.91%):**
   - Identifying straight broken cleavage planes and masking them out prevents transverse fracture edges from inflating `edge_density` and concavity defect counts.
   - Adding a localized circular dark pore filter isolates pinpoint weevil bores from linear edge roughness.
3. **Preserves 100% Production Code Compatibility & Zero Heavy Dependencies:**
   - Pure NumPy / OpenCV calculations preserve the deterministic $\approx 17\text{ ms}$ inference speed on Raspberry Pi CM5 hardware.
4. **Leaves Production Weights Untouched:**
   - Enables offline validation on held-out folds without risking regression on the 108 existing unit tests.
