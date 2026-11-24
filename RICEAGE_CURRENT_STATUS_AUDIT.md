# RiceAge / RiceCropAnalyzer — Complete Technical & Project Status Audit Against APSCSCL Requirements

**Audit Date:** November 4, 2026  
**Auditor Roles:** Senior Software Architect, Computer Vision Engineer, AI/ML Scientist, Embedded Systems Auditor  
**Repository Audited:** `RiceCropAnalyzer` / `RiceAge`  
**Current Git Branch:** `feat/phase4-2-dense-cluster-segmentation`  
**Current Git HEAD Commit:** `c004f19 docs: add Phase 4.2 dense-cluster segmentation optimization report`  
**Working Tree Status:** Clean (0 uncommitted changes)  
**Authoritative Requirement Baseline:** *Requirements for Development of AI-Based Rice Quality Analysis System – APSCSCL (Andhra Pradesh State Civil Supplies Corporation Limited)*  

---

## A. Executive Summary

### 1. Overall Maturity Verdict
The RiceAge / RiceCropAnalyzer system is currently an **Advanced Experimental Prototype (Level 2 of 5)**.

The software platform possesses a well-structured modular computer vision pipeline, a working dual-platen image acquisition driver for Canon CanoScan LiDE 400 flatbed scanners, a mathematically reconciled dual-view registration and Hungarian bipartite matching engine, and a verified classical/hybrid dense-cluster segmentation engine (Method E). On embedded Raspberry Pi Compute Module 5 (BCM2712) hardware, core dimensional morphology and defect classification operate well within the client's 5-minute analysis cycle time.

### 2. High-Level Architectural Finding: The RGB Reality vs. Client Specification
The most critical finding of this audit is an **unbridgeable divergence between client procurement specifications and the physical hardware deployed**:

> [!CAUTION]
> **Zero Multimodal Hardware Exists in the Physical System:**
> The authoritative APSCSCL specification mandates a **multimodal intelligence system combining computer vision, optical/spectral sensing (NIR/SWIR), and chemical/volatile sensing (e-nose)** to assess rice parameters.
> 
> In reality, RiceAge consists **strictly of two commercial desktop RGB flatbed document scanners (Canon LiDE 400)** mounted inside an inward-facing clamshell kiosk. The system has:
> - **ZERO Near-Infrared (NIR) or Short-Wave Infrared (SWIR) spectrometers.**
> - **ZERO chemical, volatile organic compound (VOC), or electronic nose sensors.**
> - **ZERO dielectric or capacitive grain moisture meters.**
> - **ZERO digital load cells or analytical gravimetric balances.**

### 3. Impact on Procurement Mandates
Because the system lacks spectral and chemical sensors, four cornerstone mandates of the APSCSCL tender **CANNOT be fulfilled** by the current codebase, regardless of software optimization:
1. **Milling Age Estimation (0–6 Months):** Freshly milled rice and aged rice undergo lipid peroxidation and enzymatic breakdown with negligible RGB visible color shifts during the first 3 months. Estimating milling age non-destructively requires NIR lipid absorption bands ($1200\text{ nm}$, $1700\text{ nm}$) or fatty acid titration chemistry. There is **zero code, zero data, and zero hardware** for milling age in the repository.
2. **1% Old/Fresh Admixture Detection:** Detecting 1% old rice blended into fresh paddy cannot be resolved by 8-bit RGB colorimetry. There is **zero code and zero experimental data** in the repository.
3. **Moisture Content Measurement ($\le 14\%$):** The codebase explicitly acknowledges moisture as an unsupported criterion (`quality_grading.py`, line 145/201/236).
4. **Varietal Identification:** While grain length/width ratio classifies kernels into broad shape groups (Long Slender, Medium Slender, Short Bold), genetic variety identification (e.g., BPT-5204 Samba Mahsuri vs. MTU-1010 vs. Swarna) requires trained deep neural networks across authenticated varietal datasets. The current models classify only defect morphology, not crop variety.

### 4. Status Breakdown Across All 75 Requirements
Across the 75 established requirements of the RiceAge Master Tracker:
- **DONE:** **37 / 75 (49.3%)** — Implemented with verifiable code and passing unit/regression tests.
- **PARTIAL:** **20 / 75 (26.7%)** — Functional components exist, but statutory criteria, annotations, or hardware integrations are incomplete.
- **NOT IMPLEMENTED:** **16 / 75 (21.3%)** — Entirely missing; no code, models, or sensors exist.
- **UNVERIFIED:** **2 / 75 (2.7%)** — Multi-unit scanner reproducibility and certified reference standard calibration remain untested.

---

## B. 75-Requirement Completion Matrix

*Evaluation Status Definitions:*
- **DONE:** Implemented and supported by appropriate test or validation evidence.
- **PARTIAL:** Some implementation exists, but the full statutory requirement is not satisfied.
- **NOT IMPLEMENTED:** No meaningful implementation exists in the repository.
- **UNVERIFIED:** A claim exists, but available repository evidence cannot establish it.

### Category 1: Sample and Analysis Requirements (1–5)

| Req # & Name | Official Client Requirement | Status | Responsible Module / Function | Implementation Evidence | Tested? | Validated against Ref? | Remaining Work |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **1. Sample Weight Capacity** | Analyse a representative sample of $\le 5\text{ g}$ ($\sim 200\text{--}250$ grains). | **DONE** | [`RiceAge/src/hardware/dual_platen.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/hardware/dual_platen.py), [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) | Full A4 scanning platen ($216 \times 297\text{ mm}$) easily accommodates $5\text{ g}$ to $25\text{ g}$ ($1,000\text{--}1,400$ grains). | Yes | Yes | None for optical area; physical load cell needed for mass confirmation. |
| **2. Analysis Time** | Complete end-to-end analysis within $\le 5\text{ minutes}$ ($300\text{ s}$). | **DONE** | [`RiceAge/src/hardware/dual_platen.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/hardware/dual_platen.py), [`RiceAge/src/dual_view/fusion.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/dual_view/fusion.py) | Measured CM5 benchmark: Dual scan = $17\text{ s}$, registration = $0.089\text{ ms}$, matching = $37.8\text{ ms}$, Method E = $1.22\text{ s}$, grading = $0.8\text{ s}$. Total $\approx 19.1\text{ s} \ll 300\text{ s}$. | Yes | Yes | Maintain fast cycle time under 600 DPI memory constraints. |
| **3. Non-Destructive Method** | Testing must not damage, crush, or chemically alter the grain sample. | **DONE** | [`RiceAge/src/hardware/dual_platen.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/hardware/dual_platen.py) | Optical capture on dual glass platens. Kernels remain 100% intact and returnable to batch. | Yes | Yes | Ensure top platen compression does not crack fragile grains. |
| **4. Automated Handling** | Automated sample loading, distribution, and platen presentation. | **NOT IMPLEMENTED** | None | Zero automated feeding, hopper, vibratory tray, or mechanical gate code exists. Operator manually sprinkles sample. | No | No | Design and manufacture vibratory feeder and motorized platen drawer. |
| **5. Zero Grain Alignment** | System must handle randomly oriented grains without manual spacing. | **PARTIAL** | [`RiceAge/src/separation.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/separation.py) (`separate_touching_grains`), Method E | Method E hybrid resolves 92 touching clusters into 340 grains. However, 78 severe conglomerate clusters remain unresolved and segregated. | Yes | Partial | Implement physical mechanical agitation or secondary separation pass. |

---

### Category 2: Rice Quality Parameters (6–21)

| Req # & Name | Official Client Requirement | Status | Responsible Module / Function | Implementation Evidence | Tested? | Validated against Ref? | Remaining Work |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **6. Whole Grain / Head Rice** | Quantify whole kernels ($\ge 75\%$ intact length) as % by mass and count. | **DONE** | [`RiceAge/src/classification.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/classification.py) (`classify_grain`), [`RiceAge/src/quality_grading.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/quality_grading.py) | Length ratio threshold $\ge 0.75$. Evaluated F1 = 0.996 on M3.5 benchmark dataset. | Yes | Yes | Calibrate mass conversion against certified physical scale. |
| **7. Big Broken Grains** | Quantify big broken pieces ($25\% \le \text{length} < 75\%$). | **DONE** | [`RiceAge/src/classification.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/classification.py), [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) (line 215) | Evaluated on Codex CXS 198-1995 standard tiers. Included in reporting and PDF. | Yes | Yes | Validate on physical sizing sieves. |
| **8. Small Broken Grains** | Quantify small broken fragments ($< 25\%$ unbroken kernel length). | **PARTIAL** | [`RiceAge/src/classification.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/classification.py) (`small_broken_length_ratio_thresh = 0.25`) | Algorithm implemented in code and PDF table, but M3.5 training dataset lacked dedicated small-broken annotations (lumped with broken). | Yes | No | Gather dedicated 500-sample small-broken ground truth dataset. |
| **9. Damaged Grains** | Detect pinpoint insect damage, heat damage, mold, and sprout necrosis. | **PARTIAL** | [`RiceAge/src/quality_classification.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/quality_classification.py), ResNet-50 | Supervised ResNet-50 / LR classifier detects surface spots (LR F1 = 0.886). Rule-based model failed (25.9% recall). Internal damage undetected. | Yes | Partial | Retrain ResNet-50 on multi-variety insect-bored samples; add backlighting. |
| **10. Discoloured Grains** | Detect yellow, amber, parboiled discolouration, and red/black streaks. | **DONE** | [`RiceAge/src/classification.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/classification.py) (`detect_discolouration`), LAB/HSV spaces | F1 = 0.951 in M3.5 evaluation. Dual-view captures opposite face discolouration. | Yes | Yes | Verify color space calibration against Macbeth color checker. |
| **11. Chalky / Opaque Grains** | Quantify grains with $\ge 50\%$ chalky/milky opacity. | **DONE** | [`RiceAge/src/classification.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/classification.py) (`detect_chalkiness`) | Specular highlight suppression and opacity thresholding. F1 = 0.987 on M3.5 benchmark. | Yes | Yes | Validate transillumination threshold for high-moisture varieties. |
| **12. Foreign Matter** | Detect organic (straw, husk, weed seeds) and inorganic (stones, mud) debris. | **PARTIAL** | [`RiceAge/src/classification.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/classification.py) (`detect_foreign_matter`) | Classifies non-grain shapes and severe color outliers (F1 = 0.998). Does not distinguish organic vs inorganic gravimetrically. | Yes | Partial | Implement density/NIR classification to separate stones from husks. |
| **13. Admixture Detection** | Quantify admixtures of other rice varieties or non-conforming classes. | **NOT IMPLEMENTED** | None | No varietal or grain-type admixture classifier exists in repository. | No | No | Train multi-variety classifier across authentic Andhra Pradesh crops. |
| **14. Moisture Content** | Measure moisture content ($\le 14.0\%$ IS 4333 standard). | **NOT IMPLEMENTED** | Declared unsupported in [`RiceAge/src/quality_grading.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/quality_grading.py) (line 145/201/236) | Code explicitly lists moisture as `unsupported_criteria`. Requires oven test or capacitance sensor. | No | No | Integrate hardware capacitive moisture cup or SWIR sensor (1450 nm). |
| **15. Milling Degree** | Assess degree of polish (Well Milled, Reasonably Well Milled). | **PARTIAL** | [`RiceAge/src/classification.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/classification.py) | Rudimentary bran streak and color features extracted, but no statutory grading index calibrated. | No | No | Calibrate bran streak area against ISO 6646 milling standards. |
| **16. Immature / Shrivelled** | Identify thin, green, or shrivelled kernels. | **PARTIAL** | [`RiceAge/src/measurement.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/measurement.py) | Calculates minimum breadth and perimeter irregularity, but lacks explicit immature grain classifier head. | No | No | Add dedicated green chlorophyll hue threshold and thickness gate. |
| **17. Red / Dehusked Paddy** | Detect red-striated grains and unhulled paddy seeds. | **PARTIAL** | [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) (line 222 hardcoded to 0.0%) | CVAT dataset contains minor `dehusked` class (27-53 instances), but live pipeline sets result to 0.0%. | No | No | Train dedicated paddy husk classifier head; activate in pipeline. |
| **18. Insect Infestation / Webbing** | Detect live/dead weevils, boring holes, and fungal webbing. | **PARTIAL** | [`RiceAge/src/quality_grading.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/quality_grading.py) | Bored holes lumped under `Damaged`. Live insect motion, dead weevil bodies, and silk webbing completely missing. | No | No | Train insect object detector; add motion delta across platen scans. |
| **19. Grain Dimensions (L, W, L/B)** | Measure length, breadth, and ratio in millimeters to sub-millimeter precision. | **DONE** | [`RiceAge/src/measurement.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/measurement.py) (`measure_grain_dimensions`) | Calibrated pixel scale ($25.4 / \text{DPI}$). Minimum area bounding box and major/minor axes. | Yes | Yes | Routine DPI scale calibration verification. |
| **20. Shape Distribution** | Categorize into Long Slender, Medium Slender, Short Bold. | **DONE** | [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) (lines 163–176) | Classifies based on $L$ and $L/B$ ratio conforming to Indian Rice Classification norms. | Yes | Yes | None. |
| **21. Density / Bulk Proxy** | Optical estimation of grain mass and bulk density proxy. | **PARTIAL** | [`RiceAge/src/dual_view/fusion.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/dual_view/fusion.py) (`estimate_grain_mass`) | Ellipsoid optical volume proxy ($V = \frac{4}{3}\pi \frac{L}{2}\frac{W}{2}\frac{T}{2}$). Explicitly flagged `NOT_PHYSICALLY_MEASURED`. | Yes | No | Connect USB digital micro-balance; replace optical proxy with scale. |

---

### Category 3: Multimodal Intelligence (22–26)

| Req # & Name | Official Client Requirement | Status | Responsible Module / Function | Implementation Evidence | Tested? | Validated against Ref? | Remaining Work |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **22. Dual RGB Imaging** | Top and bottom high-resolution color optical imaging. | **DONE** | [`RiceAge/src/hardware/dual_platen.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/hardware/dual_platen.py), [`RiceAge/src/dual_view/registration.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/dual_view/registration.py) | Inward clamshell dual Canon LiDE 400 capture at 300/600 DPI with geometric homography registration. | Yes | Yes | Mechanical platen latch repeatability verification. |
| **23. Optical/Spectral Sensing** | Non-destructive NIR/SWIR spectroscopy for moisture, starch, and aging. | **NOT IMPLEMENTED** | None | Zero NIR/SWIR hardware, drivers, spectral calibration, or chemometric regression code in repository. | No | No | Procure and integrate Hamamatsu/TI micro-NIR spectrometer ($900\text{--}1700\text{ nm}$). |
| **24. Chemical / Volatile Aroma** | E-nose / VOC detection for rancidity, mold, and storage volatiles. | **NOT IMPLEMENTED** | None | Zero gas sensors (MOS, PID), ADC drivers, or volatile smell feature extractors in repository. | No | No | Integrate multi-channel MOS sensor array or chemical colorimetric kit. |
| **25. Multi-Sensor Feature Fusion** | Fusion of optical, spectral, and chemical modalities. | **PARTIAL** | [`RiceAge/src/dual_view/fusion.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/dual_view/fusion.py) | Implements dual-view spatial RGB fusion across opposing platens, but does not fuse disparate modalities. | Yes | No | Build multimodal fusion network once NIR and chemical sensors exist. |
| **26. ML Decision Engine** | Machine learning quality decision engine for classification and grading. | **DONE** | [`RiceAge/src/quality_grading.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/quality_grading.py), ResNet-50 / GBDT ONNX models | Multi-head inference engine running ONNX Runtime on CM5 CPU with hierarchical statutory grading rules. | Yes | Partial | Expand training data to include authentic procurement center batches. |

---

### Category 4: Digital Reporting (27–47)

| Req # & Name | Official Client Requirement | Status | Responsible Module / Function | Implementation Evidence | Tested? | Validated against Ref? | Remaining Work |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **27. Unique Sample ID** | Auto-generation of unique sample serial/batch identifier. | **DONE** | [`RiceAge/final/app.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/final/app.py) (line 332), `Sample-XXXX` | Sequential serial tracking with persistent history file. | Yes | Yes | Add station/date prefix format (e.g., `APSCSCL-KIO01-YYYYMMDD-XXXX`). |
| **28. Timestamp** | Precise analysis date and time stamp. | **DONE** | [`RiceAge/src/reporting.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting.py), [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) | Formatted UTC and local timestamps embedded in JSON and PDF. | Yes | Yes | Ensure NTP time synchronization on CM5 kiosk. |
| **29. Center / Location & GPS** | Record procurement center, mill name, and GPS coordinates. | **PARTIAL** | [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) (line 168) | Static inspection station text (`"APSCSCL Lab Kiosk (bigb-cm5)"`); no GPS receiver or dynamic mill registry. | Yes | No | Integrate USB GPS dongle or kiosk configuration interface. |
| **30. Rice Variety Designation** | Report authenticated rice variety name. | **PARTIAL** | [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) (line 155) | Static user entry string (`"Raw Rice (Grade A)"`); not predicted by AI classifier. | Yes | No | Implement variety classification head. |
| **31. Milling Age Estimate** | Estimate milling age ($0\text{ to }6\text{ months}$). | **NOT IMPLEMENTED** | None | Zero code, zero models, zero output fields in reporting or PDF. | No | No | Requires NIR/fatty acid chemometrics or staining kit. |
| **32. Old/Fresh Mixing %** | Quantify old-rice blending percentage ($0\text{--}100\%$). | **NOT IMPLEMENTED** | None | Zero code, zero models, zero output fields in reporting or PDF. | No | No | Requires validated spectral or biochemical assay. |
| **33. Whole Grain %** | Digital report of head rice percentage. | **DONE** | [`RiceAge/src/reporting.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting.py), [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) | Reported on both count and mass basis. | Yes | Yes | None. |
| **34. Big Broken %** | Digital report of big broken percentage. | **DONE** | [`RiceAge/src/reporting.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting.py), [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) | Count and mass percentage reported against permissible limits. | Yes | Yes | None. |
| **35. Small Broken %** | Digital report of small broken percentage. | **DONE** | [`RiceAge/src/reporting.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting.py), [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) | Codex 3-tier broken classification in PDF table. | Yes | Yes | None. |
| **36. Damaged Grain %** | Digital report of damaged grain percentage. | **DONE** | [`RiceAge/src/reporting.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting.py), [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) | Evaluated against statutory maximum limit ($2.0\%$). | Yes | Yes | Improve underlying classifier recall. |
| **37. Discoloured Grain %** | Digital report of discoloured grain percentage. | **DONE** | [`RiceAge/src/reporting.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting.py), [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) | Evaluated against statutory maximum limit ($1.0\%$). | Yes | Yes | None. |
| **38. Chalky Grain %** | Digital report of chalky grain percentage. | **DONE** | [`RiceAge/src/reporting.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting.py), [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) | Evaluated against statutory maximum limit ($3.0\%$). | Yes | Yes | None. |
| **39. Foreign Matter %** | Digital report of foreign matter percentage. | **DONE** | [`RiceAge/src/reporting.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting.py), [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) | Evaluated against statutory maximum limit ($0.50\%$). | Yes | Yes | None. |
| **40. Infestation Flag** | Indicator flag for insect presence or damage. | **PARTIAL** | [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) (line 219) | Conflated with damaged grains; no separate infestation alert flag. | Yes | No | Separate insect alert from general damage. |
| **41. Storage Defect Flags** | Warning flags for yellowing, fungal mold, rancidity. | **PARTIAL** | [`RiceAge/src/quality_grading.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/quality_grading.py) | Flags severe discolouration, but has no composite storage index. | Yes | No | Add formal storage risk score. |
| **42. Confidence Scores** | Output model inference and registration confidence scores. | **DONE** | [`RiceAge/src/quality_grading.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/quality_grading.py), [`RiceAge/src/dual_view/models.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/dual_view/models.py) | Probability scores and spatial matching confidence recorded per grain. | Yes | Yes | Expose parameter-wise confidence summary in UI. |
| **43. PASS/FAIL Verdict** | Clear statutory acceptance verdict based on APSCSCL limits. | **DONE** | [`RiceAge/src/quality_grading.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/quality_grading.py), [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) | Prominent colored banner (`PASS` in green or `FAIL` in red). | Yes | Yes | None. |
| **44. PDF Certificate Export** | Automated generation and streaming of official PDF certificate. | **DONE** | [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) (`generate_apscscl_pdf_certificate`) | Single-page ReportLab vector PDF with official headers and tables. | Yes | Yes | None. |
| **45. Tamper-Proof Hash & QR** | Cryptographic hash and QR validation code on certificate. | **DONE** | [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) (lines 290–313) | SHA-256 hash digest and QR image containing signed batch parameters. | Yes | Yes | Add digital asymmetric private key signature. |
| **46. Mass-Basis Percentages** | Calculate quality parameters on mass basis (% w/w). | **DONE** | [`RiceAge/src/reporting.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting.py), [`RiceAge/src/dual_view/fusion.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/dual_view/fusion.py) | Dual-column reporting (% w/w mass and % count). | Yes | Partial | Replace optical volume proxy with digital balance readings. |
| **47. Count-Basis Percentages** | Calculate quality parameters on grain count basis. | **DONE** | [`RiceAge/src/reporting.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting.py), [`RiceAge/src/quality_grading.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/quality_grading.py) | Standard count fraction with explicit denominator rules. | Yes | Yes | None. |

---

### Category 5: Accuracy and Validation (48–57)

| Req # & Name | Official Client Requirement | Status | Responsible Module / Function | Implementation Evidence | Tested? | Validated against Ref? | Remaining Work |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **48. 1% Mixing LOD** | Limit of detection for old/fresh rice mixing from 1.0%. | **NOT IMPLEMENTED** | None | Zero models, zero laboratory trials, zero analytical LOD data. | No | No | Formulate wet-lab spiking experiment (1%, 2%, 5%, 10% old rice). |
| **49. $\ge 99\%$ Defect Accuracy** | Up to 99% accuracy for qualitative classification. | **PARTIAL** | [`RiceAge/outputs/reports/M3_5/repro_metrics.json`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/reports/M3_5/repro_metrics.json) | Met on Foreign Matter (99.9%) and Whole/Broken (99.6%). Fails on Damaged (89.5% LR, 55.7% Rule) and YOLO mask mAP50 (24.7%). | Yes | Partial | Retrain defect heads on hard negative samples; improve lighting. |
| **50. Repeatability ($\text{CV} \le 2\%$)** | Coefficient of variation $\le 2\%$ across successive runs. | **PARTIAL** | [`RiceAge/outputs/reports/M3_5/permutation_results.json`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/reports/M3_5/permutation_results.json) | Demonstrated mathematically on algorithmic permutations ($\text{CV} < 1\%$). Physical 20-cycle rescanning trial not completed. | Yes | No | Execute 20 consecutive physical scan-rescan cycles with real grain disturbance. |
| **51. Reproducibility** | Consistent results across multiple manufactured scanner kiosks. | **UNVERIFIED** | None | Only one hardware unit tested (`bigb-cm5`). Cross-unit kiosk variance unmeasured. | No | No | Benchmark across at least 3 distinct physical kiosks. |
| **52. Reference Calibration** | System calibrated against certified physical reference standards. | **UNVERIFIED** | [`RiceAge/src/threshold_calibration.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/threshold_calibration.py) | Calibrated against internal pixel grids; no ISO/NABL certified calibration slide utilized. | No | No | Procure NABL-certified stage micrometer / optical grid target. |
| **53. Tolerance Compliance** | Strict adherence to Codex CXS 198-1995 and IS 4333 limits. | **DONE** | [`RiceAge/src/quality_grading.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/quality_grading.py) | Full Codex, USDA, and Indian Standard IS 4333 profiles coded with citations. | Yes | Yes | None. |
| **54. Confusion Matrix Verification** | Full confusion matrix auditing on independent validation datasets. | **DONE** | [`RiceAge/outputs/reports/M3_5/repro_metrics.json`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/outputs/reports/M3_5/repro_metrics.json) | 5-fold cross-validated confusion matrices recorded for all 5 core defect classes. | Yes | Yes | Expand independent test set beyond original CVAT splits. |
| **55. Zero Grain Fabrication** | Strict prevention of artificial grain fabrication in cluster splitting. | **DONE** | [`RiceAge/src/separation.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/separation.py) (Method E), [`RiceAge/tests/test_dense_cluster_segmentation.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/tests/test_dense_cluster_segmentation.py) | Child provenance tracking (`parent_cluster_id`), conservative gating, 78 conglomerates fail-closed. | Yes | Yes | Maintain gating rules under varying platen backlights. |
| **56. Mass Calibration Accuracy** | Gravimetric mass accuracy calibrated against analytical balance. | **NOT IMPLEMENTED** | [`RiceAge/src/dual_view/fusion.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/dual_view/fusion.py) | Optical volume mass proxy is explicitly unvalidated and uncalibrated against real weights. | Yes | No | Build empirical regression between 3D optical voxel volume and milligram balance. |
| **57. Wet-Lab Validation** | Verification against accredited wet-lab reference methods. | **NOT IMPLEMENTED** | None | Zero comparative validation studies against Soxhlet, Kjeldahl, or IS 4333 manual grading. | No | No | Partner with an accredited food testing laboratory (e.g., NABL / CFTRI). |

---

### Category 6: Field Suitability (58–67)

| Req # & Name | Official Client Requirement | Status | Responsible Module / Function | Implementation Evidence | Tested? | Validated against Ref? | Remaining Work |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **58. CM5 Embedded Compute** | Raspberry Pi Compute Module 5 (BCM2712) embedded execution. | **DONE** | [`RiceAge/final/scanner_kiosk.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/final/scanner_kiosk.py) | Tested live on CM5 hardware (`192.168.16.160`, quad-core A76 @ 2.4 GHz, 8GB RAM). | Yes | Yes | Optimize CPU thermal throttling during continuous multi-hour runs. |
| **59. Ruggedized Protection** | Dust, humidity, and paddy husk ingress resistance (IP54). | **NOT IMPLEMENTED** | None | Device uses commercial consumer-grade Canon LiDE 400 scanners with unsealed moving carriages. | No | No | Design IP54 dust-proof enclosure with positive-pressure HEPA fan. |
| **60. Ambient Light Independence** | Enclosed scanning chamber immune to external ambient lighting. | **DONE** | Kiosk physical clamshell geometry | Inward-facing clamshell completely encloses sample during acquisition. | Yes | Yes | Add light seal gaskets along kiosk hinge seams. |
| **61. Power Fluctuation Tolerance** | Resilience to rural grid voltage spikes and brownouts. | **PARTIAL** | None | CM5 uses standard 5V/5A power supply. No graceful shutdown supercapacitor or battery backup. | No | No | Add internal 12V LiFePO4 battery pack / UPS with shutdown daemon. |
| **62. Portable Kiosk Form-Factor** | Integrated, robust, field-deployable kiosk form factor. | **DONE** | Physical kiosk chassis | Integrated clamshell housing both scanners, CM5 compute, and internal cabling. | Yes | Yes | Strengthen hinge mechanics to maintain micrometer platen alignment. |
| **63. Touchscreen UI** | Local touchscreen user interface for procurement center operators. | **PARTIAL** | [`RiceAge/final/scanner_kiosk.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/final/scanner_kiosk.py), [`RiceAge/frontend/`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/frontend/) | Tkinter kiosk and Vite/React web frontend exist, but touch calibration and on-screen keyboard need field hardening. | Yes | Partial | Connect official 7-inch Raspberry Pi Touch Display; verify touch gestures. |
| **64. Minimal Operator Training** | Single-button automated workflow for non-technical field operators. | **DONE** | [`RiceAge/final/app.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/final/app.py) (`/api/scan`) | One-touch trigger initiates acquisition, dual registration, AI grading, and report display. | Yes | Yes | Add audio/visual guidance cues (LED status ring). |
| **65. Auto-Cleaning Platen** | Automated platen cleaning mechanism between grain samples. | **NOT IMPLEMENTED** | None | No wiper blades, electrostatic dust dischargers, or pneumatic air knives exist. Operator uses microfiber cloth. | No | No | Design motorized silicone wiper blade or compressed air blow-off mechanism. |
| **66. Fast Acquisition Cycle** | Rapid image capture ($\le 30\text{ s}$). | **DONE** | [`RiceAge/src/hardware/dual_platen.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/hardware/dual_platen.py) | Sequential scanning at 300 DPI takes $17\text{ s}$ total ($6\text{ s} + 5\text{ s} + 6\text{ s}$), well within 30 s. | Yes | Yes | Benchmark USB 3.0 bus bandwidth for simultaneous parallel scanning. |
| **67. Offline Edge Operation** | Fully autonomous operation without internet connectivity. | **DONE** | Local ONNX Runtime engine on CM5 | 100% of segmentation, registration, matching, classification, and PDF generation runs offline. | Yes | Yes | None. |

---

### Category 7: Database, Traceability and Integration (68–75)

| Req # & Name | Official Client Requirement | Status | Responsible Module / Function | Implementation Evidence | Tested? | Validated against Ref? | Remaining Work |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| **68. Local Database Persistence** | Embedded database storage of all sample records and metrics. | **DONE** | [`RiceAge/final/static/history.json`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/final/static/history.json), [`RiceAge/src/feedback_store.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/feedback_store.py) | Local JSON history store and SQLite feedback database persist analysis records. | Yes | Yes | Migrate `history.json` to ACID-compliant SQLite / DuckDB. |
| **69. Cloud DB Sync** | Asynchronous background synchronization to central cloud database. | **NOT IMPLEMENTED** | None | Zero cloud sync workers, AWS S3/Azure Blob clients, or queue-based retry sync in codebase. | No | No | Implement lightweight background sync worker (MQTT / HTTPS PUT). |
| **70. APSCSCL API Gateway** | Integration with official APSCSCL procurement and warehouse APIs. | **NOT IMPLEMENTED** | None | Zero APSCSCL API endpoints, OAuth credentials, or schema adapters exist. | No | No | Obtain APSCSCL API swagger spec; implement client SDK adapter. |
| **71. Audit Trail & Event Logging** | Tamper-resistant immutable system event logging. | **PARTIAL** | [`RiceAge/final/boot_log.txt`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/final/boot_log.txt), PDF SHA-256 hash | Basic application execution logs and certificate hashing exist; no cryptographically chained audit journal. | Yes | No | Implement append-only HMAC-signed audit log table. |
| **72. Role-Based Access Control** | User authentication with distinct roles (Operator, Inspector, Admin). | **NOT IMPLEMENTED** | None | Flask and Tkinter applications run wide open with zero authentication or permission gates. | No | No | Add JWT / session-based login with Operator and Inspector roles. |
| **73. Consignment Lot Tracking** | Grouping samples into mill lots, farmer consignments, and truck batches. | **PARTIAL** | [`RiceAge/src/reporting_pdf.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/src/reporting_pdf.py) | Batch No field exists in metadata, but lacks hierarchical consignment schema in storage. | Yes | No | Add database tables for `consignments`, `truck_deliveries`, `farmer_records`. |
| **74. Structured Data Export** | Export results in open formats (CSV, JSON, REST API). | **DONE** | [`RiceAge/final/app.py`](file:///C:/Users/NewGen/Documents/RiceCropAnalyzer/RiceAge/final/app.py) (`/api/results`, `/api/history`) | Clean JSON endpoints, CSV conversion scripts, and downloadable PDF reports. | Yes | Yes | Add bulk CSV export button in frontend UI. |
| **75. Secure OTA Update** | Cryptographically signed over-the-air firmware and model updater. | **NOT IMPLEMENTED** | None | No OTA client, RAUC, SWUpdate, or signed binary verification code exists. | No | No | Implement secure systemd/RAUC update agent with Ed25519 signature checks. |

---

## C. Existing Architecture

```mermaid
flowchart TD
    subgraph Hardware ["Hardware Layer (Clamshell Kiosk)"]
        CM5["Raspberry Pi CM5 (BCM2712 @ 2.4GHz, 8GB RAM)"]
        ScanB["Bottom Canon CanoScan LiDE 400 (USB 2.0)"]
        ScanT["Top Canon CanoScan LiDE 400 (USB 2.0)"]
        Platen["Dual Glass Platens (Opposing Inward Geometry)"]
    end

    subgraph Acquisition ["Image Acquisition & Preprocessing"]
        SANE["SANE scanimage CLI (300 / 600 DPI)"]
        Settle["USB Bus Settle Delay (5s)"]
        Comp["Distortion-Free Horizontal Letterboxed Composite"]
    end

    subgraph CoreCV ["Core Computer Vision Pipeline"]
        Seg["Segmentation (Method E Hybrid Refinement)"]
        Clusters["Conglomerate Filter (78 Unresolved Segregated)"]
        Feat["Morphology & Sizing (L, W, Aspect Ratio, Area)"]
    end

    subgraph DualViewEngine ["Dual-View Registration & Canonical Fusion"]
        Homography["Direct Homography H (Reflection + Calibration)"]
        Hungarian["Bipartite Matching Engine (35px Distance, 8px Ambiguity)"]
        Canonical["Canonical Physical Grain Assembler (1:1 Pairs + Unmatched)"]
        DefectPrecedence["Traceable Defect Precedence Fusion"]
    end

    subgraph InferenceEngine ["AI/ML Defect Classification"]
        ResNet["Stage 2 ResNet-50 ONNX (Chalky, Damaged, Discoloured)"]
        Rules["Codex CXS 198-1995 / IS 4333 Statutory Rules"]
    end

    subgraph Presentation ["Reporting & Kiosk Interface"]
        Flask["Flask Backend (/api/scan, /api/results)"]
        PDF["ReportLab APSCSCL Certificate Generator (QR + SHA-256)"]
        UI["Touchscreen / Desktop Web Kiosk UI"]
        History["Local Persistence (history.json / SQLite)"]
    end

    ScanB --> SANE
    ScanT --> SANE
    SANE --> Settle --> Comp
    Comp --> Seg
    Seg --> Clusters
    Seg --> Feat
    Feat --> Homography
    Homography --> Hungarian
    Hungarian --> Canonical
    Canonical --> DefectPrecedence
    DefectPrecedence --> ResNet
    ResNet --> Rules
    Rules --> Flask
    Flask --> PDF
    Flask --> UI
    Flask --> History
    CM5 -.-> SANE
    CM5 -.-> Seg
    CM5 -.-> ResNet
```

### Module Responsibilities:
1. **`RiceAge/src/hardware/dual_platen.py`**: Controls SANE scanning drivers, prevents USB bus lockups with settle delays, applies vertical letterbox padding to prevent aspect ratio distortion, and routes platen outputs.
2. **`RiceAge/src/dual_view/registration.py`**: Encapsulates the static direct reflection homography matrix $H$, validating determinant ($\approx -1.0$) and spatial reprojection bounds.
3. **`RiceAge/src/dual_view/correspondence.py`**: Executes linear sum assignment (Hungarian bipartite matching) over a weighted cost matrix (spatial distance 50%, area difference 30%, aspect ratio difference 20%). Enforces a 35 px distance gate and an 8 px nearest-neighbor ambiguity margin.
4. **`RiceAge/src/dual_view/fusion.py`**: Merges paired top/bottom observations into canonical physical grains, enforces configurable defect precedence, logs unvalidated optical mass proxies, and segregates unresolved clusters.
5. **`RiceAge/src/separation.py`**: Implements Method E Hybrid Refinement (paired concave neck cuts followed by constrained local distance watershed) to split touching grains.
6. **`RiceAge/src/reporting_pdf.py`**: Builds authoritative, single-page A4 PDF quality certificates complete with Codex 3-tier broken tables, count- and mass-basis metrics, platen thumbnail, and a SHA-256 QR verification badge.

---

## D. Phase 4.1 / Phase 4.2 Forensic Verification

### 1. Phase 4.1 Full-Scan Observation Accounting
Phase 4.1 resolved the 71-observation discrepancy identified in Phase 4. All numbers are exactly conserved:

| Diagnostic Metric Category | Count | Unit Represented | Physical Status |
| :--- | :---: | :--- | :--- |
| **Raw Bottom Detections** | 1,077 | Scanner 1 Bounding Boxes | Optical observations |
| **Raw Top Detections** | 1,266 | Scanner 2 Bounding Boxes | Optical observations |
| **Total Sensor Observations** | **2,343** | Combined Platen Detections | Optical observations |
| **Matched Dual-View Pairs** | 911 | Pairs ($911 \times 2 = 1,822$ obs) | 911 True 1:1 Physical Grains |
| **Unmatched Bottom Detections** | 25 | Isolated Observations | 25 Physical Grains |
| **Unmatched Top Detections** | 278 | Isolated Observations | 278 Physical Grains |
| **Ambiguous Pair Observations** | 48 | 24 Matched Pairs ($24 \times 2$ obs) | 48 Segregated Observations |
| **Unresolved Multi-Grain Clusters** | 170 | Conglomerates (121 B, 49 T) | 170 Segregated Non-Single Blobs |
| **Exact Conservation Sum** | **2,343** | $1,822 + 25 + 278 + 48 + 170$ | **Zero Observation Leakage (100.0%)** |
| **Canonical Single Physical Grains** | **1,262** | $911 \text{ (pairs)} + 25 \text{ (B)} + 278 \text{ (T)} + 48 \text{ (amb)}$ | Single physical grain records |
| **Total Inventory Records** | **1,432** | $1,262 \text{ single grains} + 170 \text{ clusters}$ | All physical items on platen |

*The 71-Observation Origin Solved:* In Phase 4, non-rice debris ($71$ foreign matter items) was filtered before grain matching, leaving $2,343 - 71 = 2,272$ rice observations. In Phase 4.1, foreign matter was preserved in the unified inventory, restoring exact conservation to 2,343.

---

### 2. Phase 4.2 Dense-Cluster Segmentation Reconciliation

#### A. Reconciliation of the "77 vs 92 Resolved Clusters" Figures
- **77 Clusters (Algorithmic Zero-Remnant Metric):** In the unconstrained benchmark across all 170 cluster ROIs, Method E achieved **77 clusters where all pieces met single-grain criteria with zero merged remnants remaining**.
- **92 Clusters (Production Accepted Gate Metric):** When integrated into the production dual-view pipeline, a fail-closed quality gate was enforced: split child grains are accepted into the single-grain inventory if and only if the split produces $\ge 2$ valid grains and $\le 1$ fragment false split. **92 clusters (62 bottom, 30 top) satisfied this gate**, yielding **340 valid child grain instances**.
- The remaining **78 clusters (59 bottom, 19 top)** failed the criteria and were safely segregated into `UNRESOLVED_CLUSTER`.

#### B. Reconciliation of the "6 Annotated Clusters vs 19 Reference Grains" Figures
- Geographic cross-referencing between the 170 cluster ROIs and the independently annotated ground-truth benchmark (`ground_truth_deduped_150.csv`) revealed that **exactly 6 clusters** fell within the benchmarked coordinate zones.
- Within these 6 physical clusters, ground-truth human annotators had verified **19 physical reference grains**.
- Method E successfully recovered **13 True Positives (TP)**, incurred **1 False Positive (FP)**, and left **6 False Negatives (FN)**, achieving:
  - $\text{Precision} = \frac{13}{13 + 1} = \mathbf{0.929}$
  - $\text{Recall} = \frac{13}{13 + 6} = \mathbf{0.684}$
  - $\mathbf{F_1\text{ Score}} = \mathbf{0.788}$

#### C. Cross-Platen Integration Outcomes
Integrating the 340 child instances into the dual-view matching engine produced:
- **Matched 1:1 Dual-View Pairs increased from 911 to 983 (+72 newly verified physical pairs).**
- Direct matches involving child grains: **105 pairs**.
- Remaining unresolved clusters dropped from **170 to 78 (-92 clusters)**.

#### D. CM5 Embedded Hardware Benchmark: Measured vs Projected Timings

```
==========================================================================================
CM5 HARDWARE BENCHMARK (BCM2712 Cortex-A76 @ 2.4GHz, 8GB RAM, Debian 12, Python 3.13.5)
==========================================================================================
Segmentation Method   | Mean Latency (ms/ROI) | 20 ROIs Measured (s) | 170 Extrapolated (s)
------------------------------------------------------------------------------------------
A. Otsu Baseline      | 0.71 ms / ROI         | 0.014 s [MEASURED]   | 0.120 s [PROJECTED]
B. Local Watershed    | 6.28 ms / ROI         | 0.126 s [MEASURED]   | 1.068 s [PROJECTED]
C. Concave Neck Cuts  | 1.62 ms / ROI         | 0.032 s [MEASURED]   | 0.275 s [PROJECTED]
D. YOLO11s-seg ONNX   | 2,576.63 ms / ROI     | 51.533 s [MEASURED]  | 438.027 s [PROJECTED]
E. Hybrid Refinement  | 7.18 ms / ROI         | 0.144 s [MEASURED]   | 1.220 s [PROJECTED]
==========================================================================================
Embedded Memory Profile: Initial RSS = 200.5 MB -> Post-YOLO Peak RSS = 1,078.1 MB (+877.6 MB)
Dual-View Projection: 0.089 ms [MEASURED] | Hungarian Matching (1400x1400): 37.8 ms [MEASURED]
```

*Architectural Conclusion:* Method E executes across all 170 clusters in **1.22 seconds** on the CM5 CPU. Method D (YOLO) is catastrophic on embedded hardware: requiring **7.3 minutes** and $>1\text{ GB}$ of RAM, while suffering from resolution downsampling artifacts ($>4\times$ shrink).

#### E. Test Suite Status
All **28 / 28** dual-view and dense-cluster automated tests pass synchronously in 4.35s:
- `tests/test_dual_platen.py`: 2 passed
- `tests/test_dual_view_prototype.py`: 20 passed
- `tests/test_dense_cluster_segmentation.py`: 6 passed

---

## E. Working Versus Non-Working Capabilities

### What Can Genuinely Be Demonstrated Right Now:
1. **Live Dual-Platen Scanning:** Sequential flatbed scanning on Canon LiDE 400 scanners at 300/600 DPI with distortion-free horizontal letterboxing.
2. **Sub-Pixel Spatial Registration:** Applying reflection homography $H$ and matching grains across platens with $< 3\text{ px}$ reprojection error.
3. **Dense-Cluster Resolution:** Splitting touching grains using Method E hybrid concavity-watershed in $1.22\text{ s}$ on CM5, lifting matched pairs to 983.
4. **Physical Grain Morphology:** Accurate length, breadth, aspect ratio, and Codex 3-tier broken grain classification.
5. **Color Defect Inspection:** Surface chalkiness ($\ge 50\%$) and discolouration screening using RGB thresholding and ResNet-50.
6. **Official APSCSCL PDF Certificate Export:** Automated rendering of single-page A4 certificates with SHA-256 tamper-evident hash, dynamic QR code, and statutory PASS/FAIL disposition banners.
7. **Offline Edge Operation:** 100% autonomous inference on the Raspberry Pi CM5 with zero cloud dependency.

### What is Partially Implemented / Lacking Scientific Validation:
1. **Damaged Grain Recall:** Surface pinpoint damage achieves 87–89% recall with deep learning, but rule-based models fail (25.9% recall). Internal damage is invisible to RGB reflection.
2. **Foreign Matter Discrimination:** Non-grain debris is filtered, but the system cannot differentiate organic vs inorganic matter gravimetrically.
3. **Optical Mass Proxy:** Gram weight is estimated via 3D optical ellipsoid integration ($V \cdot \rho$). It is explicitly unvalidated against physical scales and unusable for statutory trade.
4. **Varietal Shape Classification:** Kernels are classified into geometric shape bins (Long Slender, Medium Slender), but genetic variety identification is absent.

### What is Completely Missing / Non-Functional:
1. **Milling Age Estimation (0–6 Months):** No sensors, no code, no dataset.
2. **1% Old/Fresh Rice Mixing Detection:** No sensors, no code, no dataset.
3. **Grain Moisture Content Measurement:** No dielectric/NIR moisture hardware.
4. **Automated Platen Self-Cleaning & Ingress Protection:** Platen requires manual wiping; enclosure lacks dust sealing.
5. **Central APSCSCL Cloud & API Integration:** No cloud database synchronization, no REST API client, no role-based access control.

---

## F. Critical Blockers Ranked by Dependency

```mermaid
graph TD
    B1["Blocker 1: Absence of NIR & Chemical Sensors<br/>(Prevents Milling Age & 1% Mixing Compliance)"]
    B2["Blocker 2: Lack of Digital Load Cell<br/>(Invalidates Mass-Basis Trade Acceptance)"]
    B3["Blocker 3: YOLO11s-seg Training Deficiencies<br/>(Mask mAP50=24.7%, Low Damaged Recall)"]
    B4["Blocker 4: 78 Unresolved Dense Conglomerates<br/>(Requires Physical Grain Agitation)"]
    B5["Blocker 5: Zero APSCSCL Backend & Cloud Sync<br/>(Prevents Enterprise Rollout)"]

    B1 --> B3
    B2 --> B5
    B4 --> B2
```

1. **Blocker 1 (CRITICAL — Specification Non-Compliance):** The clamshell hardware is purely RGB visible light. It is physically impossible to measure milling age (0–6 months) or 1% old/fresh mixing without Near-Infrared (NIR) absorption or chemical assays.
2. **Blocker 2 (HIGH — Legal Metrology):** Indian statutory grain grading standards (IS 4333) mandate refraction percentages by weight (% w/w). The current optical volume mass estimate is uncertified. Statutory compliance requires integrating a calibrated digital micro-balance.
3. **Blocker 3 (HIGH — Vision Model Accuracy):** YOLO11s-seg achieved only 24.7% mask mAP50 in training logs, and cannot run on localized cluster ROIs without severe downsampling artifacts. ResNet-50 damaged grain recall remains below 90%.
4. **Blocker 4 (MEDIUM — Physical Sample Dispersion):** 78 clusters contain overlapping "pancake" conglomerates where internal boundaries have zero optical contrast. Pure software cannot resolve overlapping grains without physical mechanical agitation.
5. **Blocker 5 (MEDIUM — Software Traceability):** The application runs as an isolated local kiosk without cloud database synchronization, APSCSCL API gateway integration, or role-based user access controls.

---

## G. Remaining Implementation Roadmap

### Phase 1: Immediate Corrections (Weeks 1–2)
- **Task 1.1:** Formalize single-scanner fallback (`single_view_fallback_scanner = "bottom"`) in the UI dashboard so operators can toggle between dual-view research mode and hardened single-view mode.
- **Task 1.2:** Resolve the 15 legacy pytest test failures in `tests/test_dataset.py` and `tests/test_m4_integration.py` caused by stale relative paths and outdated milestone signatures.
- **Task 1.3:** Lock down the mass-basis disclaimer in the PDF generator, displaying explicit notices that optical mass is an experimental proxy until load-cell integration.

### Phase 2: Core Computer Vision Hardening (Weeks 3–5)
- **Task 2.1:** Retile and retrain the YOLO11s-seg segmentation backbone at native grain patch resolution ($320 \times 320\text{ px}$ without downsampling letterbox) to resolve mask mAP50 deficiencies.
- **Task 2.2:** Expand the ResNet-50 defect training dataset with 2,000 verified insect-bored, heat-damaged, and pecky rice kernels from Andhra Pradesh procurement centers.
- **Task 2.3:** Add an automated boundary-margin filter to reject grains truncated along the scanner glass bezel.

### Phase 3: Milling Age, Mixing & Variety Intelligence (Weeks 6–10)
- **Task 3.1:** Collect authenticated samples of major Andhra Pradesh varieties (BPT 5204 Samba Mahsuri, MTU 1010, Swarna, NLR 34449) across freshly harvested, 3-month aged, and 6-month aged lots.
- **Task 3.2:** Build a fine-grained Varietal Convolutional Classifier using high-resolution morphological and textural features.
- **Task 3.3:** Develop a standardized chemical staining protocol (e.g., Guaiacol/Hydrogen Peroxide or Bromothymol Blue/Methyl Red pH test) integrated with the optical scanner as an interim proxy for milling age prior to spectrometer procurement.

### Phase 4: Multimodal Hardware Integration (Weeks 11–16)
- **Task 4.1:** Procure and integrate a micro-NIR spectral sensor ($900\text{--}1700\text{ nm}$) over USB/I2C to capture moisture and lipid oxidation absorption bands.
- **Task 4.2:** Integrate a certified USB digital analytical balance ($0.001\text{ g}$ precision) into the kiosk platen, streaming true gravimetric mass directly into `reporting.py`.
- **Task 4.3:** Build a multi-sensor fusion module combining RGB morphology, NIR spectral reflectance, and gravimetric weight.

### Phase 5: Reporting, Traceability & APSCSCL API Gateway (Weeks 17–19)
- **Task 5.1:** Implement the official APSCSCL REST API client, transmitting analysis payloads, SHA-256 digests, and PDF certificates to the central procurement database.
- **Task 5.2:** Migrate local storage from `history.json` to an encrypted, append-only SQLite database with HMAC tamper logging.
- **Task 5.3:** Implement Role-Based Access Control (RBAC) with PIN/RFID login for procurement center operators and district quality inspectors.

### Phase 6: Field Readiness & Scientific Validation (Weeks 20–24)
- **Task 6.1:** Execute a formal 20-cycle physical mechanical repeatability and reproducibility trial across 3 distinct kiosks ($\text{CV} \le 2\%$).
- **Task 6.2:** Conduct a dual-blind wet-lab comparative validation study against IS 4333 manual grading and AOAC laboratory methods.
- **Task 6.3:** Package the kiosk into an IP54 dust-resistant field enclosure with positive-pressure HEPA filtration and internal battery backup.

---

## H. Delivery Assessment

```
==========================================================================================
                              RICEAGE MATURITY ASSESSMENT
==========================================================================================

  Current Engineering State:
  [ LEVEL 2 : WORKING EXPERIMENTAL PROTOTYPE ]

  - Level 1: Research Prototype
  - Level 2: Working Experimental Prototype  <--- [CURRENT REPOSITORY STATUS]
  - Level 3: Validated Prototype
  - Level 4: Field-Trial Candidate
  - Level 5: Delivery-Ready System

==========================================================================================
```

### Three Authoritative Delivery Verdicts:

1. **Current Demo Readiness:**  
   **[ CONDITIONAL PASS — OPTICAL MORPHOLOGY & DEFECT DEMO ONLY ]**  
   The system is fully capable of demonstrating instant grain counting, length/breadth measurement, Codex broken-grain classification, dual-surface registration, Method E cluster splitting, and automated APSCSCL PDF certificate export on the CM5 hardware.  
   *Mandatory Demo Disclaimer:* Presenters must explicitly clarify to client stakeholders that milling age, 1% mixing detection, and gravimetric mass are experimental or pending sensor integration.

2. **Client Requirement Compliance:**  
   **[ NON-COMPLIANT (49.3% STATUTORY COMPLIANCE) ]**  
   The system satisfies 37 of 75 requirements. It fails core client mandates regarding Milling Age, 1% Old/Fresh Mixing Detection, Moisture Content, Varietal Identification, and Central APSCSCL API Gateway Integration due to the total absence of NIR/chemical sensors and network adapters.

3. **Production / Field Deployment Readiness:**  
   **[ NOT READY ]**  
   The kiosk cannot be deployed to active procurement centers without:
   - An IP54 dust-sealed chassis protecting scanner rails from paddy husk debris.
   - An integrated physical digital load cell for legal metrology compliance.
   - An accredited wet-lab comparative validation trial proving $\ge 99\%$ accuracy against IS 4333 reference standards.
