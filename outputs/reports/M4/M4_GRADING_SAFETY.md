# RiceAge Milestone M4 — Grading Safety & Regulatory Limitations Report

**Milestone:** M4 — End-to-End Quality Analysis Integration  
**Date:** October 12, 2026  
**Auditor / Engineer:** Antigravity Senior Computer-Vision & Quality Systems Team  
**Subject:** Standards-Based Rice Grading Compliance, Safety Guardrails, and Denominator Auditing  
**Safety Status:** **STRICT REGULATORY GUARDRAILS ENFORCED**

---

## 1. Regulatory Context & The Certification Fallacy

Grain grading standards published by statutory bodies—including the **Codex Alimentarius Commission (CXS 198-1995)**, the **United States Department of Agriculture (USDA Federal Grain Inspection Service)**, and the **Bureau of Indian Standards (IS 4333 / AGMARK)**—define binding legal grades for commercial grain contracts.

A common failure mode in automated computer-vision grading software is claiming that a flatbed optical scanner produces a "Codex Certified" or "USDA Grade 1 Certified" report. 

**This claim is legally invalid and technically false for three fundamental reasons:**
1. **Gravimetric vs. Optical Count Discrepancy:** Official standards define defect maximums as **mass percentages** ($w/w$, grams of defect per 100 grams of sample), whereas 2D optical scanners count individual **pixel objects** ($n/n$, count percentage).
2. **Mandatory Non-Optical Quality Factors:** Standards mandate non-optical laboratory testing (moisture content $\le 14\%$, olfactory evaluation for mold/rancidity, uric acid testing).
3. **Internal & Hidden Defects:** Flatbed scanners observe a single 2D projection. Fissures or insect pinholes hidden on the underside of a kernel resting on the glass are invisible without specialized multi-view or transillumination hardware.

---

## 2. Implemented Safety Guardrails (Milestone M4)

To prevent misrepresentation and protect laboratory integrity, the RiceAge M4 pipeline enforces five mandatory safety policies:

### Policy 1: Mandatory "Optical Screening Estimate (Provisional)" Labeling
All generated artifacts—including the web HTML report, the JSON data payload, and the CSV grain register—explicitly designate results as:
$$\text{"Optical Quality Screening Estimate (Provisional)"}$$
The term **"Certificate"** has been permanently removed from report titles. The JSON report metadata includes the explicit tag:
```json
"statutory_certification_claim": "NONE (Optical Count-Based Screening Only)"
```

### Policy 2: Automatic `INCOMPLETE` Status on Regulatory Certification Requests
When an inspection is evaluated with regulatory certification requested (`--require-full-criteria` or `require_full_criteria=True`), the grading engine flags the status as **`INCOMPLETE`** rather than claiming compliance:
```json
{
  "profile_id": "CODEX_CXS_198_1995",
  "assigned_grade": null,
  "status": "INCOMPLETE",
  "is_compliant": false,
  "unsupported_criteria": [
    "moisture_content (Max 14.0% - Requires calibrated moisture meter or oven-drying)",
    "red_rice_and_damaged_kernels_by_mass (Requires calibrated gravimetric laboratory scale)",
    "odor (Must be free from commercially objectionable odors and rancidity)"
  ]
}
```

### Policy 3: Rejection of Speculative Mass-Conversion Formulas
Under-engineered AI systems frequently attempt to calculate mass by estimating volume as an ellipsoid:
$$V \approx \frac{4}{3} \pi \left(\frac{L}{2}\right) \left(\frac{W}{2}\right)^2$$
and multiplying by an assumed rice bulk density ($\approx 1.4\text{ g/cm}^3$). 

**RiceAge M4 explicitly rejects this practice.** Individual grain density varies significantly with moisture content, variety (indica vs. japonica), chalky endosperm porosity (chalky rice has lower starch density), and internal cavitation. Fabricating a mass percentage from an uncalibrated 2D optical bounding box gives a false impression of laboratory precision. **All RiceAge metrics are strictly reported as verified optical count percentages.**

### Policy 4: Strict Physical Object Conservation & Denominator Guarantees
The M3.5 audit revealed risks of denominator inflation when grains have multiple coexisting defects. M4 enforces strict physical object conservation:
$$\text{Total Detected Objects} = \text{Total Rice Kernels} + \text{Foreign Matter Objects}$$
$$\text{Total Rice Kernels} = \text{Whole Kernels} + \text{Broken Kernels}$$

- Every detected object belongs to exactly one category: Foreign Matter or Rice.
- Every detected rice kernel belongs to exactly one primary morphology: Whole or Broken.
- Multi-defect grains (e.g. Broken + Chalky + Damaged) increment each defect counter and increment `multi_defect_rice_count`, but contribute **exactly 1.0** to the denominator ($\text{Total Rice Kernels}$).

### Policy 5: Explicit Denominator Attribution
Every reported percentage specifies its exact mathematical denominator:

| Metric | Numerator ($N$) | Denominator ($D$) | Mathematical Formula |
| :--- | :--- | :--- | :--- |
| **Whole Rice %** | Count of Whole Rice | Total Rice Kernels | $N_{\text{whole}} / N_{\text{rice}} \times 100$ |
| **Broken Rice %** | Count of Broken Rice | Total Rice Kernels | $N_{\text{broken}} / N_{\text{rice}} \times 100$ |
| **Chalky Kernels %** | Count of Chalky Kernels | Total Rice Kernels | $N_{\text{chalky}} / N_{\text{rice}} \times 100$ |
| **Damaged Kernels %** | Count of Damaged Kernels | Total Rice Kernels | $N_{\text{damaged}} / N_{\text{rice}} \times 100$ |
| **Discoloured Kernels %** | Count of Discoloured Kernels | Total Rice Kernels | $N_{\text{discoloured}} / N_{\text{rice}} \times 100$ |
| **Foreign Matter %** | Count of Foreign Objects | Total Detected Objects | $N_{\text{foreign}} / N_{\text{objects}} \times 100$ |

---

## 3. Regulatory Standard Criteria Audit

### 3.1 Codex Alimentarius Standard for Rice (CXS 198-1995, Revised 2019)

| Criterion | Codex Requirement | RiceAge Optical Capability | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Moisture Content** | Max 14.0% $m/m$ | **Unmeasured** (requires dielectric/capacitive moisture meter or 130°C air oven) | **UNSUPPORTED (Marked INCOMPLETE)** |
| **Broken Kernels** | Max 5.0% (Extra) to 25.0% (Grade 3) | **Measured (Optical Count %)** | **Optical Estimate Only** |
| **Chalky Kernels** | Max 2.0% (Extra) to 8.0% (Grade 3) | **Measured (Optical Count %)** | **Optical Estimate Only** |
| **Damaged Kernels** | Max 1.5% (Extra) to 4.0% (Grade 3) | **Measured (Optical Count %)** | **Optical Estimate Only** |
| **Discoloured Kernels** | Max 0.5% (Extra) to 3.0% (Grade 3) | **Measured (Optical Count %)** | **Optical Estimate Only** |
| **Foreign Matter** | Max 0.1% (Extra) to 1.0% (Grade 3) | **Measured (Optical Count %)** | **Optical Estimate Only** |
| **Odor & Foreign Taste** | Free from objectionable odors | **Unmeasured** (requires sensory panel) | **UNSUPPORTED (Marked INCOMPLETE)** |

### 3.2 USDA Standards for Milled Rice (7 CFR Part 868)

| Criterion | USDA Requirement | RiceAge Optical Capability | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Moisture** | Max 14.0% | **Unmeasured** | **UNSUPPORTED** |
| **Heat-Damaged Kernels** | Max 0.2% to 3.0% by weight | Optical browning proxy available; weight unmeasured | **Screening Proxy Only** |
| **Chalky Kernels** | Max 1.0% (US No. 1) to 15.0% (US No. 6) | **Measured (Optical Count %)** | **Optical Estimate Only** |
| **Broken Kernels** | Max 4.0% (US No. 1) to 50.0% (US No. 6) | **Measured (Optical Count %)** | **Optical Estimate Only** |
| **Foreign Material** | Max 0.1% to 2.0% by weight | **Measured (Optical Count %)** | **Optical Estimate Only** |

### 3.3 Indian Standard IS 4333 & AGMARK Rules

| Criterion | IS 4333 Requirement | RiceAge Optical Capability | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Moisture Content** | Max 14.0% (IS 4333 Part 2) | **Unmeasured** | **UNSUPPORTED** |
| **Inorganic Foreign Matter** | Max 0.2% to 0.5% (stone/sand) | Optical detection cannot distinguish stone density from seed pod | **Screening Proxy Only** |
| **Uric Acid Content** | Max 100 mg/kg (rodent feces) | **Unmeasured** (requires spectrophotometry / enzymatic assay) | **UNSUPPORTED** |
| **Dehulled Kernels** | Minimum 98.0% | High optical precision via color/texture | **Screening Proxy Only** |

---

## 4. Verification and Safety Recommendations

1. **Intended Use:** RiceAge is designed as a **rapid pre-screening and sorting tool** for grain elevators, mills, and receiving bays to detect batches that exceed defect thresholds before unloading.
2. **Statutory Audits:** Where financial settlements or export certificates require certified statutory compliance, laboratory technicians must supplement RiceAge optical screening with an approved moisture meter and an analytical balance.
3. **Operator Training:** Ensure laboratory operators understand that count percentages will differ from mass percentages when kernels are disproportionately small (broken pieces have lower mass than whole kernels).
