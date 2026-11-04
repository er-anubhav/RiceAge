# RiceAge Milestone M4 — Remaining Limitations & Pilot Readiness Report

**Milestone:** M4 — End-to-End Quality Analysis Integration  
**Date:** October 12, 2026  
**Auditor / Engineer:** Antigravity Senior Computer-Vision & Quality Systems Team  
**System Status:** **INTEGRATION COMPLETE (INTERNAL VALIDATION COMPLETE)**  
**Readiness Verdict:** **PILOT PENDING UNSEEN MULTI-SCANNER DATASET**

---

## 1. Executive Summary

Milestone M4 has successfully integrated the complete RiceAge quality classification, grading, and reporting pipeline into the real application entrypoint. A raw flatbed scanner image flows seamlessly through bezel clearing, Otsu binarization, touching grain separation, per-crop feature extraction, multi-head defect inference, denominator-reconciled metrics aggregation, and standards-based quality grading screening.

However, in accordance with the project's strict engineering ethos—**"do not claim production readiness without evidence"**—this report discloses the remaining technical, optical, and regulatory limitations of the system. These disclosures define the scope of required work prior to commercial pilot deployment.

---

## 2. Technical and Data Limitations

### 2.1 Single-Label Annotation Bias in Ground-Truth Data
- **Finding:** The 119 CVAT annotated images in `data/dataset/` assign exactly **one** categorical label to each grain (e.g., a grain is marked strictly as `Whole`, `Broken`, `Chalky`, `Damaged`, `Discoloured`, or `Foreign Matter`).
- **Limitation:** In commercial agriculture, rice defects are inherently multi-label. A grain may be both broken and insect-damaged, or whole, chalky, and yellowed.
- **Impact on ML Models:** Because true co-occurring defects were forced into single labels during annotation, training single-head binary classifiers inevitably introduces label noise. Although M4 reduced cross-triggering by $>77-85\%$, a residual cross-triggering rate remains ($\approx 14.0\%$ Broken $\to$ Damaged, $\approx 21.5\%$ Chalky $\to$ Discoloured).
- **Remediation:** Full multi-label re-annotation by certified grain inspectors is required before headline defect classification accuracy can exceed $90\%$ on Damaged and Discoloured kernels.

### 2.2 2D Planar Optical Scanning Constraints
- **Finding:** The system evaluates grains positioned on a flat glass platen under top-down directional scanner illumination.
- **Limitation:** 
  1. **Hidden Faces:** Approximately $50\%$ of the grain surface area (the side in direct contact with the glass) is obscured from view. Insect damage, pinholes, or fungal spotting on the bottom face cannot be detected.
  2. **Internal Fissuring:** Fissured (checked) grains appear intact in standard reflected light. Accurately detecting internal stress cracks requires a specialized transillumination lightbox (backlighting).
  3. **Multi-Layer Stacking:** The pipeline assumes a dispersed monolayer of grains. Stacked, heaped, or overlapping multi-grain piles cannot be resolved by 2D watershed segmentation.

### 2.3 Optical Count vs. Gravimetric Mass Fraction Disparity
- **Finding:** Codex CXS 198-1995, USDA Milled Rice, and Indian Standard IS 4333 define defect tolerances as **mass percentages** ($w/w$, grams per 100 grams). RiceAge measures **optical kernel counts** ($n/n$).
- **Limitation:** A broken kernel fragment weighing $10\text{ mg}$ contributes the same $+1$ count as a whole head-rice kernel weighing $25\text{ mg}$. Consequently, optical count percentages systematically overestimate broken rice fractions relative to official laboratory gravimetric balances.
- **Remediation:** RiceAge results must always be presented as **Optical Screening Estimates**, and must never be converted into speculative mass fractions using unverified volumetric formulas.

### 2.4 Single-Scanner Hardware & Illumination Bias
- **Finding:** The 119 dataset scans originate from specific flatbed scanner units (primarily HP Scanjet G3110 / Canon flatbed platens) at nominal resolutions of 600–1200 DPI.
- **Limitation:** 
  1. **Color Profile Variance:** Colorimetry features ($L^*a^*b^*$, HSV) are sensitive to scanner lamp aging (CCFL vs. LED), glass reflectance, and scanner driver color profiles.
  2. **Smartphone / Camera Incompatibility:** The pipeline's Otsu thresholding and bezel margin clearing algorithms are calibrated for dark scanner backgrounds with telecentric platen optics. They cannot be used directly on uncalibrated smartphone camera photos taken with ambient shadows, perspective distortion, or uneven illumination.

### 2.5 Unmeasured Non-Optical Quality Criteria
- **Finding:** Commercial grain contracts require verification of non-optical quality criteria.
- **Limitation:** The following statutory requirements cannot be determined from optical scan imagery:
  - **Moisture Content:** Maximum $14.0\%$ (requires capacitive moisture meter or 130°C oven test).
  - **Odor:** Must be free from commercially objectionable odors (mustiness, sourness, chemical contamination).
  - **Rodent Contamination / Uric Acid:** Maximum $100\text{ mg/kg}$ under IS 4333 (requires enzymatic spectrophotometry).
  - **Live Insect Infestation:** Requires physical sample sifting.

---

## 3. Recommended Roadmap for Future Milestones

| Priority | Initiative | Objective | Proposed Milestone |
| :---: | :--- | :--- | :---: |
| **1** | **Multi-Label Dataset Expansion** | Annotate 5,000 grains with independent binary labels for Whole/Broken, Chalkiness %, Insect Damage, and Discoloration. | M5.0 |
| **2** | **Multi-Scanner Transfer Benchmark** | Collect scans across 5 distinct scanner models (Epson, Canon, HP, Brother) to calibrate color normalization (Macbeth color checker). | M5.1 |
| **3** | **Hardware Calibration Target** | Implement an in-scan color and dimensional calibration strip (e.g. ColorChecker target placed on the platen). | M5.2 |
| **4** | **Dual-Illumination Hardware** | Prototype reflected + transillumination imaging to capture internal fissuring and 360° kernel translucency. | M6.0 |

---

## 4. Final Verdict

**FINAL VERDICT: INTEGRATION COMPLETE**  
*(Internal Software Validation Complete; Commercial Pilot Pending Unseen Multi-Scanner Dataset)*

### Summary of Acceptance Criteria Verification:
1. **Application Invocation:** Real pipeline (`RiceQualityPipeline.run()` and `main.py`) invokes M3/M4 modules directly on raw images. $\rightarrow$ **MET**
2. **Reconciled Reporting:** Raw scans produce reconciled HTML, JSON, and CSV reports. $\rightarrow$ **MET**
3. **Zero Ground-Truth Dependence:** Inference never accesses annotation files or CVAT metadata. $\rightarrow$ **MET**
4. **Physical Object Conservation:** Multi-defect grains increment defect tallies but count as exactly one physical grain in denominators. $\rightarrow$ **MET**
5. **Grading Safety:** Results labeled strictly as "Optical Screening Estimate (Provisional)", regulatory certification claimed as "NONE", and non-optical requirements flagged as "INCOMPLETE". $\rightarrow$ **MET**
6. **Test Suite:** All 68 regression and integration tests pass without failures. $\rightarrow$ **MET**
7. **Metric Decoupling:** Detection, per-crop classification, and end-to-end results reported separately. $\rightarrow$ **MET**
