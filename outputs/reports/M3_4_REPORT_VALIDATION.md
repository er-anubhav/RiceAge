# RiceAge M3.4: Quality Report Generation & Schema Validation

**Document Version:** 1.0.0  
**Audit Date:** 2026-10-12  
**Lead Auditor:** Senior Computer Vision & Machine Learning QA Engineer, Google DeepMind Antigravity Team  
**Module Target:** [`src/quality_reporting.py`](file:///home/anubhavtripathi/Documents/RiceAge/src/quality_reporting.py)  
**Deliverable Files:**
- Report Validation Document: [`outputs/reports/M3_4_REPORT_VALIDATION.md`](file:///home/anubhavtripathi/Documents/RiceAge/outputs/reports/M3_4_REPORT_VALIDATION.md)
- Representative HTML Report (Mixed): [`outputs/reports/SAMPLE_MIXED_01_report.html`](file:///home/anubhavtripathi/Documents/RiceAge/outputs/reports/SAMPLE_MIXED_01_report.html)
- Representative JSON Report (Mixed): [`outputs/reports/SAMPLE_MIXED_01_report.json`](file:///home/anubhavtripathi/Documents/RiceAge/outputs/reports/SAMPLE_MIXED_01_report.json)
- Representative CSV Report (Mixed): [`outputs/reports/SAMPLE_MIXED_01_grains.csv`](file:///home/anubhavtripathi/Documents/RiceAge/outputs/reports/SAMPLE_MIXED_01_grains.csv)
- Representative HTML Report (Whole): [`outputs/reports/SAMPLE_WHOLE_02_report.html`](file:///home/anubhavtripathi/Documents/RiceAge/outputs/reports/SAMPLE_WHOLE_02_report.html)
- Representative JSON Report (Whole): [`outputs/reports/SAMPLE_WHOLE_02_report.json`](file:///home/anubhavtripathi/Documents/RiceAge/outputs/reports/SAMPLE_WHOLE_02_report.json)
- Representative CSV Report (Whole): [`outputs/reports/SAMPLE_WHOLE_02_grains.csv`](file:///home/anubhavtripathi/Documents/RiceAge/outputs/reports/SAMPLE_WHOLE_02_grains.csv)

---

## 1. Overview of Reporting Engine Architecture

The RiceAge reporting engine transforms multi-head classification predictions and quality grading evaluations into tripartite export formats:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ MULTI-HEAD QUALITY CLASSIFIER (Foreign Matter, Whole/Broken, Chalky, Damaged, Disc.)   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ QUALITY GRADING ENGINE (Counts, Non-Redundant Denominators, Regulatory Standards)       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
         ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
         │ Machine JSON Export │ │ Granular CSV Export │ │ Human HTML Report   │
         │ (Audit Traceability)│ │ (Kernel By Kernel)  │ │ (Quality Dashboard) │
         └─────────────────────┘ └─────────────────────┘ └─────────────────────┘
```

---

## 2. Report Schema Validation & Data Integrity

Each report guarantees compliance with the following structural contracts:

### A. Machine-Readable JSON Contract
The JSON output contains four top-level sections:
1. `report_metadata`: Sample identification, ISO 8601 generation timestamp, pipeline version, and source image path.
2. `quality_metrics`: Total detected objects, total rice grains, foreign matter count, Whole/Broken counts, Chalky, Damaged, and Discoloured counts, computed percentages, and mathematical denominator formulas.
3. `grading_decision`: Standard profile ID, legal citation, assigned grade, compliance flag, passed criteria, failed criteria, unsupported laboratory criteria, and audit notes.
4. `grain_classifications`: Instance-level array containing `object_id`, boolean flags (`is_rice`, `is_foreign_matter`, `is_chalky`, `is_damaged`, `is_discoloured`), morphology (`Whole`, `Broken`, `N/A`), and confidence scores.

### B. Machine-Readable CSV Contract
A flat table recording every detected physical object with 8 standardized columns:
`sample_id, object_id, is_rice, is_foreign_matter, morphology, is_chalky, is_damaged, is_discoloured`.

### C. Human-Readable Standalone HTML Contract
A responsive dashboard rendered in inline CSS (requiring zero external network requests) featuring:
- Grade status badge with dynamic color coding (Green for Compliant, Red for Substandard, Amber for Incomplete, Gray for Indeterminate).
- Summary KPI cards for Total Rice Kernels, Foreign Matter, Head Rice Ratio, and Total Defect Flags.
- Tabular breakdown detailing measured counts, percentages, mathematical denominators, and regulatory limits.
- Regulatory Certification & Laboratory Limitations notice highlighting unmeasured non-optical criteria.

---

## 3. Representative Sample Validation

### Sample 1: Complex Mixed Sample (`mixed 1.jpg`, `SAMPLE_MIXED_01`)
- **Total Objects Detected:** 32 objects (24 Rice Kernels, 8 Foreign Matter items).
- **Whole / Broken Distribution:** 4 Whole (16.67%), 20 Broken (83.33%).
- **Defects Detected:** 6 Chalky (25.00%), 1 Damaged (4.17%), 2 Discoloured (8.33%).
- **Foreign Matter:** 8 objects (25.00% of total sample objects).
- **Evaluation under Codex CXS 198-1995:**
  - Broken rice (83.33%) exceeds the Grade 3 ceiling (25.0%).
  - Chalkiness (25.0%) exceeds the Grade 3 ceiling (8.0%).
  - Damaged rice (4.17%) exceeds the Grade 3 ceiling (4.0%).
  - Discoloured rice (8.33%) exceeds the Grade 3 ceiling (3.0%).
  - Foreign matter (25.0%) exceeds the Grade 3 ceiling (1.0%).
- **Assigned Grade:** `Substandard / Below FAQ` (`is_compliant = False`).

### Sample 2: Clean Whole Rice Scan (`whole 2.jpg`, `SAMPLE_WHOLE_02`)
- **Total Objects Detected:** 20 objects (20 Rice Kernels, 0 Foreign Matter).
- **Whole / Broken Distribution:** 20 Whole (100.00%), 0 Broken (0.00%).
- **Defects Detected:** 0 Chalky (0.00%), 0 Discoloured (0.00%), 2 Damaged (10.00%).
- **Small Sample Quantum Sensitivity Discovery:**
  In a 20-grain scan sheet, each grain represents **5.0%** of the sample ($1/20$).
  When 2 grains were flagged with slight surface erosion, the damaged percentage was $10.0\%$, exceeding the Codex Grade 3 limit ($\le 4.0\%$).
  This proves why commercial grading cannot rely on 20-grain test sheets and mandates statistical sample sizes ($N \ge 1,000$ to $40,000$ grains).

---

## 4. Indeterminate State Triggers & Regulatory Compliance Notice

The reporting system strictly distinguishes between optical quality grading and full legal regulatory certification:

1. **When Final Grades Cannot Be Justified:**
   - If $N_{\text{rice}} = 0$, the sample is marked `INDETERMINATE`.
   - If non-optical parameters are required (`require_full_criteria=True`), the report status is marked `INCOMPLETE`.
2. **Mandatory Disclaimer on Every Report:**
   Flatbed scanner imaging cannot measure moisture content ($14-15\%$), sensory odors, or internal kernel molds. Therefore, reports must never be presented as legal trade certificates without certified moisture and gravimetric testing.
