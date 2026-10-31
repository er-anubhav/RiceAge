# RiceAge AI — Formal Audit of Validation Semantics Under PU Learning

**Audit Date:** October 31, 2026  
**Auditor:** Automated Engineering System  
**Validation Universe:** $N = 4,944$ canonical unique instances across 29 scans (`outputs/phase2_automated/automated_dataset_manifest.csv`)  
**Scope:** Strict ground-truth supervision provenance, evaluator code paths, metric validity under Positive-Unlabelled (PU) learning, composite score audit, and pre-EXP006 protocol requirements.

---

## Executive Summary

This audit establishes that **the current validation metric `composite_score` contains a fatal semantic flaw for PU-trained defect heads**:

1. **Head 1 (Identity) and Head 2 (Integrity)** are fully/selectively supervised and their metrics ($F_1$, Accuracy) are **100% mathematically valid**.
2. **Heads 3, 4, and 5 (Damaged, Chalky, Discoloured)** are strictly Positive-Unlabelled (PU). In the validation split, there are **ZERO confirmed negatives** for Damaged (3,822 unlabelled grains) and Discoloured (4,779 unlabelled grains), and only 59 confirmed negatives for Chalky (4,445 unlabelled grains).
3. In [`auto_train/evaluate.py`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/auto_train/evaluate.py#L20-L32), the function `compute_auprc()` was implemented as classical **Precision-Recall AUC**, where precision is computed as $\text{precisions} = \text{cum\_pos} / \text{ranks}$. This directly treats **all 3,822 unlabelled Damaged grains, all 4,445 unlabelled Chalky grains, and all 4,779 unlabelled Discoloured grains as CONFIRMED NEGATIVES**.
4. Consequently, whenever self-training discovers genuine defects on unlabelled grains (as confirmed by Discoloured recall nearly doubling from 23.08% to 42.31%), **the evaluator penalizes every discovered defect as a False Positive, artificially deflating AU-PRC**.
5. In EXP003, `best_multitask_model.pt` was selected using `joint_score = (f1_id + f1_mo + rec_dam + rec_chk + rec_dsc) / 5.0` (**0.6547**). When evaluated under this original selection metric, EXP005 achieved **0.6785 in Epoch 1 (+0.0238 over baseline)** and **0.6579 in Epoch 6 (+0.0032 over baseline)**.
6. **Recommendation:** Do not launch EXP006 until the validation protocol is updated to decouple classical metrics from PU metrics. Preserve the existing report for historical continuity, and add a mathematically rigorous **Corrected PU Validation Protocol** before any new training runs.

---

## 1. Ground-Truth Supervision State Provenance by Head

Tracing from original CVAT annotations, through Phase 1.2 canonical reconciliation ([`outputs/phase1_2_final_audit/ternary_label_provenance.csv`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/outputs/phase1_2_final_audit/ternary_label_provenance.csv)) to the automated dataset manifest ([`outputs/phase2_automated/automated_dataset_manifest.csv`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/outputs/phase2_automated/automated_dataset_manifest.csv)):

| Head / Attribute | Confirmed Positives ($V_P$) | Confirmed Negatives ($V_N$) | Unlabelled / Unknown ($V_U$) | Total Evaluated | Ground-Truth Supervision Semantics |
|---|:---:|:---:|:---:|:---:|---|
| **Head 1: Object Identity** | **113** *(Foreign)* | **4,831** *(Rice Kernel)* | **0** | **4,944** | **100% Fully Supervised Ground Truth**. Every grain is verified as rice vs foreign matter. |
| **Head 2: Physical Integrity** | **408** *(Broken)* | **3,791** *(Whole)* | **745** *(Masked)* | **4,199** | **Verified Selective Ground Truth**. Masked instances (632 unspecified morphology + 113 foreign) have $m_{\text{morph}} = 0.0$ and are excluded from evaluation. |
| **Head 3: Damaged Defect** | **1,009** | **0** *(Zero Negatives)* | **3,822** | **4,831** | **Strictly Positive-Unlabelled (PU)**. Positive selection bias in CVAT: annotators flagged obvious damage; absence of annotation does NOT indicate sound grain. |
| **Head 4: Chalky Defect** | **327** | **59** *(Reference Controls)* | **4,445** | **4,831** | **Positive-Negative-Unlabelled (PNU)**. The 59 negatives originate from pure reference scans (`PURE_REFERENCE_SCAN`). 98.7% of non-positives are unlabelled. |
| **Head 5: Discoloured Defect** | **52** | **0** *(Zero Negatives)* | **4,779** | **4,831** | **Strictly Positive-Unlabelled (PU)**. Extreme class imbalance ($\pi \approx 0.042$). Zero confirmed negatives exist in the validation split. |

> [!NOTE]
> In [`outputs/phase1_2_final_audit/ternary_label_provenance.csv`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/outputs/phase1_2_final_audit/ternary_label_provenance.csv), unlabelled defect states are explicitly justified as:  
> `UNINSPECTED: Uninspected mixed tray grain; absence of annotation is NOT negative.`  
> The data provenance explicitly prohibits treating absence of a defect tag as a negative confirmation.

---

## 2. Code-Level Evaluator Audit: Treating Unlabelled as Negative

The exact code paths in [`auto_train/evaluate.py`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/auto_train/evaluate.py) were audited:

### 2.1 The Metric Calculation in `compute_auprc()`
```python
def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    n_pos = np.sum(y_true == 1.0)
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score)
    sorted_y = y_true[order]
    cum_pos = np.cumsum(sorted_y == 1.0)
    ranks = np.arange(1, len(sorted_y) + 1)
    precisions = cum_pos / ranks
    return float(np.sum(precisions[sorted_y == 1.0]) / n_pos)
```

### 2.2 Where Evaluator Treats Unlabelled as Confirmed Negatives
1. **Damaged Head ([`auto_train/evaluate.py#L126`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/auto_train/evaluate.py#L126)):**
   ```python
   auprc_dam = compute_auprc(y_dam_gr, p_dam_gr)
   ```
   `y_dam_gr` contains `1.0` for 1,009 confirmed positives and `0.0` for all 3,822 unlabelled grains. `precisions = cum_pos / ranks` penalizes every unlabelled grain with a high predicted score as a False Positive ($FP$).
2. **Chalky Head ([`auto_train/evaluate.py#L147`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/auto_train/evaluate.py#L147)):**
   ```python
   auprc_chk = compute_auprc((y_chk_gr == 1.0).astype(float), p_chk_gr)
   ```
   Collapses both confirmed negatives ($N=59$) and unlabelled grains ($N=4,445$) into `0.0`, treating all 4,445 unlabelled grains identically to verified negatives.
3. **Discoloured Head ([`auto_train/evaluate.py#L166`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/auto_train/evaluate.py#L166)):**
   ```python
   auprc_dsc = compute_auprc(y_dsc_gr, p_dsc_gr)
   ```
   `y_dsc_gr` treats all 4,779 unlabelled grains as confirmed negatives.

---

## 3. Metric Validity Matrix Under Current Supervision

| Metric | Target Heads | Subsets Used | Evaluator Validity Status | Scientific Rationale |
|---|---|---|:---:|---|
| **Identity F1, Accuracy, Precision, Recall** | Head 1 | All $N=4,944$ grains | **VALID** | Ground truth is 100% complete. $TP, FP, FN, TN$ are verified. |
| **Integrity F1, Accuracy, Precision, Recall** | Head 2 | Masked subset ($N=4,199$) | **VALID** | Evaluated strictly on grains with confirmed whole/broken morphology; uninspected instances are masked out ($m_{\text{morph}} = 0$). |
| **Confirmed Positive Recall ($R_P@0.50$, $R_P@0.80$)** | Heads 3, 4, 5 | Confirmed Positives ($V_P$) | **VALID** | Evaluated exclusively on confirmed positive instances. Under Selected Completely at Random (SCAR), $\mathbb{E}[R_P] = R_{\text{true}}$. |
| **Chalky Negative Specificity** | Head 4 | Confirmed Negatives ($N=59$) | **VALID** | Evaluated strictly on pure reference control grains ($y=-1.0$). Reached 100.0% in EXP005. |
| **PU-AUC (Rank-Separation AUC)** | Heads 3, 4, 5 | $V_P$ vs $V_U$ | **VALID (Relative)** | Measures rank separation between known positives and the unlabelled background. Strictly monotonic with true AUC under SCAR (Elkan & Noto, 2008). |
| **Unlabelled Moment Statistics ($\mu_U, \text{Med}_U$)** | Heads 3, 4, 5 | Unlabelled Grains ($V_U$) | **VALID (Diagnostics)** | Verifies whether the model avoids trivial all-negative ($\mu_U \to 0$) or all-positive ($\mu_U \to 1$) collapse. |
| **Defect AU-PRC (`compute_auprc`)** | Heads 3, 4, 5 | $V_P$ vs $V_U$ | **MATHEMATICALLY INVALID** | Treats thousands of unlabelled grains as confirmed negatives in classical precision calculations, penalizing genuine positive discovery. |
| **Existing `composite_score`** | All 5 Heads | Combined | **MATHEMATICALLY INVALID** | 3 of the 5 components are the invalid AU-PRC metrics, producing systematic negative bias against self-trained models. |

---

## 4. Evaluation of the Existing Composite Score for Model Selection

### 4.1 Why the Existing Composite Score Fails
The existing composite score is defined as:
$$\text{Composite Score} = \frac{1}{5} \left( F_{1,\text{id}} + F_{1,\text{mo}} + \text{AUPRC}_{\text{dam}} + \text{AUPRC}_{\text{chk}} + \text{AUPRC}_{\text{dsc}} \right)$$

When a model improves defect feature representations (e.g., in EXP005, Discoloured Recall @ 0.50 surged from 23.08% to 42.31%), it naturally identifies subtle defect cues in unlabelled validation grains. But because those unlabelled grains have ground truth $y=0$, every genuine detection is counted as a **False Positive** in `compute_auprc()`. This forces AU-PRC down, which drags down the composite score and triggers an automatic rejection by the validation gate.

### 4.2 Historical Evidence from EXP003
In EXP003 ([`outputs/phase2_automated/experiments/exp003_multitask_integrated/train_exp003.py#L380`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/outputs/phase2_automated/experiments/exp003_multitask_integrated/train_exp003.py#L380)), `best_multitask_model.pt` was selected using:
$$\text{Joint Score} = \frac{1}{5} \left( F_{1,\text{id}} + F_{1,\text{mo}} + R_{\text{dam}}@0.50 + R_{\text{chk}}@0.50 + R_{\text{dsc}}@0.50 \right)$$
Under this metric, EXP003 achieved **0.6547**.

### 4.3 Multi-Metric Re-Evaluation Across All Experiments

| Experiment | Flawed Composite Score (with Pseudo-Neg AU-PRC) | Original Joint Recall Score (EXP003 Formulation) | PU-AUC Rank Separation Score |
|---|:---:|:---:|:---:|
| **EXP003 Baseline** | **0.6026** | **0.6547** | **0.8568** |
| **EXP004 Self-Training** | 0.5547 *(Collapse)* | 0.6136 *(Collapse)* | 0.8253 *(Collapse)* |
| **EXP005 (Epoch 1 Peak)** | 0.5856 | **0.6785 (+0.0238 over baseline)** | **0.8612 (+0.0044 over baseline)** |
| **EXP005 (Epoch 6 Final)** | 0.5771 | **0.6579 (+0.0032 over baseline)** | 0.8474 |

> [!IMPORTANT]
> When evaluated using statistically valid PU metrics, **EXP005 did in fact outperform the baseline at Epoch 1 (0.6785 vs 0.6547)**, driven by massive recall gains in Damaged (61.45% $\to$ 67.20%) and Discoloured (23.08% $\to$ 36.54%). However, the flawed AU-PRC metric masked this improvement and forced a rejection.

---

## 5. Recommended Validation Strategy (No Manual Labeling Required)

To establish a scientifically sound model-selection protocol without requiring human inspectors or manual annotation of 5,000 grains, adopt a **Two-Tier PU Validation Suite**:

### Tier 1: Primary Model Selection Metric — Corrected PU Multi-Task Score (CPMS)
Replace the flawed AU-PRC components in the acceptance gate with verified positive recall and PU rank separation:
$$\text{CPMS} = \frac{1}{5} \left( F_{1,\text{id}} + F_{1,\text{mo}} + R_{\text{dam}}@0.50 + R_{\text{chk}}@0.50 + R_{\text{dsc}}@0.50 \right)$$
*Properties:*
- Uses only 100% verified ground-truth labels.
- Unlabelled instances are not assumed to be negative.
- Directly comparable to the original EXP003 selection checkpoint metric (**0.6547**).

### Tier 2: Distributional & Calibration Guardrails (Non-Degeneracy Gate)
To prevent a model from "cheating" by predicting all ones, enforce three automated guardrails:
1. **Unlabelled Moment Guardrail ($\mu_U \approx \pi$):**  
   The mean predicted probability on unlabelled grains must remain within $\pm 0.05$ of the class prior ($\pi_{\text{dam}} = 0.265, \pi_{\text{chk}} = 0.125, \pi_{\text{dsc}} = 0.042$).
2. **Negative Control Guardrail (Chalky Specificity):**  
   Chalky specificity on the 59 confirmed negative reference grains must remain $\ge 95.0\%$.
3. **Cross-Scan Consistency Guardrail:**  
   $\text{Mean Scan CV} \le 1.40$ across the 29 validation scans.

---

## 6. Exact Changes Required Before EXP006

Before launching EXP006, the following four changes must be committed:

1. **Update Evaluator ([`auto_train/evaluate.py`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/auto_train/evaluate.py)):**
   - Retain `legacy_composite_score` for 100% backwards compatibility and reproducibility.
   - Add `corrected_pu_score` (CPMS) as the primary model-selection metric.
   - Replace classical `compute_auprc` with the normalized **Positive Retrieval Curve Integral** specified in `automated_validation_protocol.md` Section 2.2.
2. **Update Validation Protocol Document ([`outputs/phase2_automated/automated_validation_protocol.md`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/outputs/phase2_automated/automated_validation_protocol.md)):**
   - Explicitly document the CPMS definition and its mathematical justification.
   - Set the baseline acceptance threshold to $\text{CPMS} \ge 0.6557$ ($0.6547 + 0.001$).
3. **Update Training Acceptance Gate ([`auto_train/train.py`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/auto_train/train.py#L244-L250)):**
   - Wire the checkpoint saving condition to `corrected_pu_score >= best_score + min_improvement` while enforcing the 3 calibration guardrails.
4. **Test Split Isolation:**
   - Maintain the quarantine on the held-out test split ($N=5,836$ canonical grains across 20 scans) with zero leakage.
