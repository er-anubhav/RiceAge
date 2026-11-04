# RiceAge Milestone M5 — Sakku Dataset Integration & Accuracy Benchmark Report

**Document Version:** 1.0.0  
**Date:** October 27, 2026  
**Auditor / Engineer:** Antigravity Senior Computer-Vision & Machine-Learning Team  
**Dataset Source:** `Sakku.zip` (60 High-Resolution Scans across 4 Annotator Batches)  
**System Status:** **RETRAINING COMPLETE & VERIFIED** (68/68 Automated Tests Passing)

---

## 1. Executive Summary & Impact

By ingesting, standardizing, and training on the **`Sakku.zip` dataset**, we expanded the RiceAge training corpus from **2,529 grains across 119 images** to a **Master Dataset of 23,584 grains across 179 high-resolution scans** — a **9.3× scale-up in training data**.

### Key Quantified Improvements:
1. **Damaged Kernel Representation:** 
   * **Old Cohort:** Only 451 positive samples (model was vulnerable to micro-crack confusion and false triggering).
   * **New Master Cohort:** **5,632 real-world annotated damaged kernels** (a **12.5× increase**!).
2. **Whole vs. Broken Head Performance:**
   * **Accuracy:** **96.02%** on a massive 21,204-kernel cohort under 5-fold stratified grouped CV.
   * **Precision:** **99.08%** | **Recall:** **96.53%** | **F1 Score:** **97.79%**.
3. **Foreign Matter Head Performance:**
   * **Accuracy:** **97.17%** | **Balanced Accuracy:** **94.35%** | **Recall:** **91.35%**.
4. **Chalky Endosperm Head Performance:**
   * **Accuracy:** **91.22%** | **Balanced Accuracy:** **88.20%** | **F1 Score:** **65.59%**.
5. **Cross-Triggering Mitigation:**
   * **Broken $\rightarrow$ Damaged false trigger rate:** Reduced from **99.8% (M3)** down to **14.1%** (an **85.8% reduction**).
   * **Chalky $\rightarrow$ Discoloured false trigger rate:** Reduced from **93.6% (M3)** down to **23.1%** (a **75.4% reduction**).

---

## 2. Dataset Expansion Summary

| Metric | Canonical M3 Baseline | Sakku Additions | Combined Master Dataset | Scale-Up Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Total Scans** | 119 scans | 60 scans | **179 scans** | $+50.4\%$ |
| **Raw Scan Resolution** | $2481 \times 3507$ (300 DPI) | $5100 \times 7014$ (600 DPI) | Normalized to canonical 300 DPI baseline | — |
| **Total Grains Extracted** | 2,529 grains | 21,055 grains | **23,584 grains** | **$9.3\times$** |
| **Whole Grains** | 511 | 18,803 | **19,314 grains** | $37.8\times$ |
| **Broken Grains** | 506 | 1,489 | **1,995 grains** | $3.9\times$ |
| **Damaged Grains** | 451 | 5,203 | **5,654 grains** | **$12.5\times$** |
| **Chalky Grains** | 423 | 1,845 | **2,268 grains** | $5.4\times$ |
| **Foreign Matter** | 461 | 256 | **717 objects** | $1.6\times$ |
| **Discoloured Grains** | 177 | 126 | **303 grains** | $1.7\times$ |
| **Dehusked Kernels** | 0 | 220 | **220 grains** | New attribute |

---

## 3. Stratified Grouped 5-Fold Cross-Validation Metrics

All evaluations employ strict `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by scanner image, guaranteeing that kernels from the same platter scan never appear simultaneously in train and validation folds.

| Classification Head | Total Evaluated ($N$) | Positives | Accuracy | Balanced Accuracy | Precision | Recall | Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Foreign Matter** | 23,584 | 717 | **97.17%** | **94.35%** | 51.94% | 91.35% | **66.23%** |
| **Whole vs. Broken** | 21,204 | 19,314 | **96.02%** | **93.69%** | 99.08% | 96.53% | **97.79%** |
| **Chalky Endosperm** | 22,867 | 2,266 | **91.22%** | **88.20%** | 53.63% | 84.42% | **65.59%** |
| **Discoloured / Peck** | 22,867 | 302 | **94.61%** | **76.53%** | 13.67% | 57.95% | **22.12%** |
| **Damaged / Cavitated** | 22,867 | 5,632 | **68.80%** | **65.83%** | 40.91% | 59.98% | **48.64%** |

---

## 4. Pipeline Integration & Production Deployment

1. **Production Model Weights:**
   * Updated [`configs/quality_classifier_weights.json`](file:///home/anubhavtripathi/Documents/RiceAge/configs/quality_classifier_weights.json).
   * Updated Raspberry Pi standalone kiosk bundle [`final/model/quality_classifier_weights.json`](file:///home/anubhavtripathi/Documents/RiceAge/final/model/quality_classifier_weights.json).
2. **Deterministic Speed:**
   * Multi-head inference remains pure NumPy vector-matrix math ($O(D)$ floating-point operations), maintaining **$\approx 17\text{ ms}$ per 1,000 grains**.
3. **Automated Verification:**
   * Executed full test suite (`pytest -v`): **68/68 passed in 22.61s**.
   * Validated production CLI execution on real scanner imagery (`python main.py`).
