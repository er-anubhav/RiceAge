# RiceAge M3.5 — Quality Grading & Regulatory Standards Audit Report

**Audit Date:** 2026-10-12  
**Auditor Role:** Senior Computer Vision Engineer & Quality Assurance Auditor  
**Scope:** Verification of grading engine ([src/quality_grading.py](file:///home/anubhavtripathi/Documents/RiceAge/src/quality_grading.py)), standards traceability (Codex, USDA, IS 4333), mass vs count units, and legal certification boundaries.

---

## 1. Executive Summary & Regulatory Classification

> [!IMPORTANT]
> **LEGAL AND REGULATORY BOUNDARY STATEMENT:**  
> RiceAge implements **optical screening proxies** based on grain kernel counts. It does **NOT** perform certified legal grain inspection.  
> Official Codex, USDA, and Indian Standards (IS 4333 / AGMARK) mandate **gravimetric mass percentages (m/m)** and non-optical tests (moisture meters, olfactory analysis, enzymatic uric acid testing).  
> The grading engine is **INTERNALLY VALIDATED AS AN OPTICAL SCREENING TOOL**, but must be designated **UNVERIFIED / PROVISIONAL FOR STATUTORY REGULATORY CERTIFICATION**.

| Profile Identifier | Regulatory Standard | Governing Authority | Grade Tiers | Gravimetric Unit Required | Non-Optical Factors Required | Audit Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `CODEX_CXS_198_1995` | Codex Standard for Rice (CXS 198-1995, Rev. 2019) | FAO / WHO Codex Alimentarius | Extra, Grade 1, 2, 3 | Yes (% by mass) | Moisture, Odor, Live Insects | **OPTICAL PROXY ONLY** |
| `USDA_MILLED_RICE` | US Standards for Milled Rice (7 CFR Part 868) | USDA FGIS | US No. 1 to No. 6 | Yes (% by mass) | Moisture, Odor, Gravimetric Red Rice | **OPTICAL PROXY ONLY** |
| `IS_4333_AGMARK` | Indian Standard IS 4333 / AGMARK Rules | BIS / DMI | Grade I, II, III | Yes (% by mass) | Moisture (IS 4333 Pt 2), Uric Acid | **OPTICAL PROXY ONLY** |

---

## 2. Unit Mismatch Audit: Kernel Count vs Gravimetric Mass

### 2.1 The Fundamental Physical Mismatch
Official international rice standards define defect maximum allowances strictly by weight percentage:
$$\text{Defect \% (Standard)} = \frac{\text{Mass of Defective Kernels (grams)}}{\text{Total Sample Mass (grams)}} \times 100 \quad [\% \text{ m/m}]$$

The RiceAge system computes defect percentages by visual count:
$$\text{Defect \% (RiceAge)} = \frac{\text{Count of Defective Kernels}}{\text{Total Rice Kernels}} \times 100 \quad [\% \text{ by count}]$$

### 2.2 Operational Consequences
1. **Broken Rice Underestimation:**
   Broken rice grains are physically smaller and lighter than whole grains (typically $25\% - 65\%$ the mass of an intact kernel).
   - If a 100-grain sample has 10 broken grains and 90 whole grains, the **count percentage is 10.0%**.
   - If each whole grain weighs 22 mg and each broken fragment averages 11 mg:
     $$\text{Mass broken} = 10 \times 11\text{ mg} = 110\text{ mg}$$
     $$\text{Mass whole} = 90 \times 22\text{ mg} = 1,980\text{ mg}$$
     $$\text{Mass \%} = \frac{110}{2090} \times 100 = 5.26\%$$
   The count percentage (10.0%) **doubles** the apparent broken defect level compared to gravimetric standards!
2. **Foreign Matter Density Disparity:**
   Husk fragments and straw debris have far lower bulk density than mineral stones. A count-based percentage treats a lightweight 2 mm husk piece identically to a 2 mm dense gravel stone, violating standard gravimetric fraction tables.

---

## 3. Sample Quantum & Statistical Power Audit

### 3.1 The 20-Grain Discretization Problem
The current RiceAge dataset consists of 119 scans containing approximately **20 physical grains per image sheet**.

In a 20-grain sample:
- **1 grain = 5.0%**
- **2 grains = 10.0%**
- **3 grains = 15.0%**

### 3.2 Impossibility of High-Tier Grading on 20 Grains
Consider the official threshold tables implemented in `src/quality_grading.py`:
- **Codex Extra:** Maximum allowable chalky rice = **2.0%**, maximum damaged = **1.5%**, maximum foreign matter = **0.1%**.
- **USDA No. 1:** Maximum allowable damaged = **0.5%**, maximum discoloured = **0.1%**.

Under single-sheet 20-grain scanning:
- If a sample contains **0 chalky grains**, chalky % = 0.0% $\rightarrow$ **Passes**.
- If a sample contains **1 chalky grain**, chalky % = 5.0% $\rightarrow$ **Fails Codex Extra and Grade 1 immediately!**
- If a sample contains **1 piece of foreign matter**, foreign matter % = 4.8% $\rightarrow$ **Fails all defined Codex and USDA grades!**

Official commercial inspection standards specify working samples of **$25\text{g} - 100\text{g}$** ($1,200 - 5,000$ grains). Single-sheet scans of 20 grains have severe discrete quantum errors ($\pm 5.0\%$) and cannot reliably distinguish a 1.5% threshold from a 5.0% threshold.

---

## 4. Non-Optical Measurement Handling

`src/quality_grading.py` correctly handles missing measurements through its dual-status design:

```python
# From src/quality_grading.py
if require_full_criteria and len(profile.unsupported_criteria) > 0:
    status = "INCOMPLETE"
    notes.append("Status is marked INCOMPLETE because required non-optical regulatory criteria (e.g. moisture content) cannot be verified from scanner images.")
elif assigned_grade is not None:
    status = "DECIDED"
    notes.append(f"Sample meets optical requirements for {assigned_grade}.")
```

### Verification Against Minimum Test Requirements
1. **Default Mode (`require_full_criteria=False`):** Produces status `DECIDED` with grade `Extra`, `Grade_1`, etc., and appends mandatory audit notes clarifying that the evaluation is optical-only.
2. **Certification Mode (`require_full_criteria=True`):** Produces status `INCOMPLETE` whenever mandatory non-optical factors (moisture, odor, insect infestation) are unverified.
3. **Empty Input Handling:** When `total_rice_grains == 0`, produces status `INDETERMINATE` with explicit failure explanation.

---

## 5. Formal Recommendations for Milestone M4

1. **Volume / Mass Proxy Derivation:**
   In Milestone M4, estimate individual grain volume from extracted dimensions:
   $$V \approx \frac{4}{3} \pi \left(\frac{\text{length}}{2}\right) \left(\frac{\text{width}}{2}\right)^2$$
   Multiply by standard milled rice grain density ($\rho \approx 1.40\text{ g/cm}^3$) to approximate gravimetric mass fractions.
2. **Sample Aggregation:**
   Require scanning of at least 5 to 10 sheets ($100 - 200$ grains minimum) before generating a sample-level grading decision to reduce the quantum step size to $\le 0.5\%$.
3. **Report Disclaimers:**
   Every generated HTML, PDF, and JSON report must prominently display:
   > *"This certificate is an optical quality screening estimate. Official trade certification requires gravimetric laboratory inspection according to ISO 7301 / Codex CXS 198-1995."*
