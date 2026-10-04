# Rice Grain Quality Analyzer

A modular, production-oriented Computer Vision and Machine Learning system for automated rice grain detection, touching-grain separation, geometric and photometric quality measurement, and international-standard quality grading from controlled scanner/imaging setups.

---

## 1. Project Overview & First Principles

In industrial and agricultural grain inspection, jumping prematurely to heavy deep learning models often leads to brittle, uncalibrated systems. This repository follows a **disciplined, classical-first computer vision engineering methodology**:

1. **Phase M0 (Current):** Establish a robust, deterministic Classical CV baseline for segmentation, touching-grain separation, counting, and metric extraction using real controlled-imaging sample data.
2. **Phase M1 (Next):** Refine complex touching-grain separation via concave neck contour analysis and validate against a physical ground-truth count tray.
3. **Phase M2 (Future):** Ingest and curate public grain defect datasets (GrainDet, GrainBench, etc.) to train lightweight, targeted classification models for internal defects (chalkiness, peck/pest damage, discoloration).
4. **Phase M3 (Future):** Fine-tune and validate against actual custom defective and foreign-material samples once acquired.

---

## 2. Milestone M0 Technical Results

Evaluation conducted on real flatbed scanner imagery (`sample_1489x2048.jpg` derived from raw 1200 DPI scan `laserjet_1200dpi_sample.jpg`).

| Metric | Measured Value | Notes |
|---|---|---|
| **Total Detected Grains** | **726** | Algorithmic estimate (no manual ground truth exists) |
| **Initial Connected Components** | **590** | Baseline CC without separation |
| **Naturally Isolated Grains** | **494** | 68.0% of total kernels (preserved intact) |
| **Touching Clusters Detected** | **80** | Multi-grain clusters identified via solidity & area |
| **Clusters Successfully Separated** | **72 (90.0%)** | Split into **208** individual grain kernels |
| **Unresolved Clusters (Under-segmented)** | **8 (10.0%)** | Complex parallel touching triplets flagged explicitly |
| **Boundary Truncated Grains** | **19** | Grains touching platter borders (excluded from stats) |
| **Broken Grains (<75% whole median)** | **128 (17.6%)** | Codex CXS 198-1995 standard rule-based baseline |
| **Whole Grains** | **571 (78.7%)** | Aspect ratio $\ge 1.6$, length $\ge 75\%$ median |
| **Mean Grain Length** | **4.73 mm** (median 4.84 mm) | Calibrated at 0.14499 mm/pixel |
| **Mean Grain Width** | **2.17 mm** (median 2.15 mm) | Standard medium/long grain proportions |
| **Mean Aspect Ratio** | **2.23** (median 2.20) | Typical for indica/hybrid rice |

---

## 3. Physical Calibration Analysis

- **Raw Image:** HP LaserJet flatbed scanner, resolution $10,200 \times 14,028$ pixels at **1200 DPI**.
  $$\text{Scale}_{\text{raw}} = \frac{25.4\text{ mm}}{1200\text{ pixels}} \approx 0.0211667\text{ mm/pixel} \quad (47.244\text{ pixels/mm})$$
- **Standardized Sample Image:** $1489 \times 2048$ pixels.
  $$\text{Downsampling Factor} = \frac{10200}{1489} \approx 6.8502\times$$
  $$\text{Effective DPI} = \frac{1200}{6.8502} \approx 175.18\text{ DPI}$$
  $$\text{Scale}_{\text{sample}} = 0.0211667 \times 6.8502 \approx 0.14499\text{ mm/pixel} \quad (6.897\text{ pixels/mm})$$
- All measurements are reported in both pixels (`_px`) and calibrated millimeters (`_mm`, `_mm2`).

---

## 4. Pipeline Architecture

```
RAW IMAGE / STANDARDIZED SAMPLE
    │
    ▼
[ Preprocessing & Border Clearing ] ──> Detects saturated scanner bezel; zeroes margins; flags border grains
    │
    ▼
[ Denoising & Filtering ] ───────────> 5x5 Gaussian kernel (sigma=0.8) preserves grain boundaries
    │
    ▼
[ Segmentation / Binarization ] ─────> Otsu thresholding (selected over Triangle, Yen, Li, Adaptive)
    │
    ▼
[ Morphological Cleanup ] ───────────> Ellipse opening (3x3), hole filling (specular glints), noise removal
    │
    ▼
[ Cluster Geometry Analysis ] ───────> Evaluates area and contour solidity (A / A_hull)
    │
    ├── (Solidity >= 0.92, Area in range) ──> [ Isolated Single Grains ] (Preserved intact)
    └── (Solidity < 0.92 or Area > 650px) ──> [ Marker-Controlled Watershed ]
                                                    │
                                                    ├── Local Euclidean Distance Transform (DIST_L2)
                                                    ├── Peak Local Maxima Detection (min_distance=8px)
                                                    └── Inverted Distance Basins Flooding
    │
    ▼
[ Grain Measurement Extraction ] ───> Minimum-area bounding box, fitted ellipse, centroid, area, perimeter,
                                      aspect ratio, circularity, solidity, RGB color, highlight ratio
    │
    ▼
[ Standard Quality Categorization ] ─> International Standard (Codex CXS 198-1995 / USDA / IRRI)
    │
    ▼
[ Output Artifacts ] ────────────────> Visualizations, individual crops, binary masks, CSV, JSON, Markdown Report
```

---

## 5. Repository Structure

```
RiceAge/
├── configs/
│   ├── __init__.py
│   └── default_config.py          # Dataclass-based configurable pipeline parameters
│
├── data/
│   ├── raw/                       # Original high-res 1200 DPI scan (10200x14028)
│   ├── samples/                   # Standardized test samples (1489x2048)
│   ├── processed/
│   └── annotations/
│
├── src/
│   ├── preprocessing/             # Calibration, scanner bezel clearing, boundary tagging, denoise
│   ├── segmentation/              # Binarization methods (Otsu, Triangle, Yen, Li), morphology
│   ├── separation/                # Connected components, distance transform, selective watershed
│   ├── measurement/               # Per-grain morphology, geometry, orientation, color, highlight ratio
│   ├── classification/            # Standard taxonomy and rule-based quality categorization
│   ├── reporting/                 # Visualizations, crops/masks exporter, markdown report generator
│   └── pipeline.py                # Master RiceQualityPipeline orchestrator
│
├── scripts/
│   └── run_baseline.py            # CLI entrypoint to run the end-to-end pipeline
│
├── experiments/
│   ├── exp001_m0_baseline.py      # Reproducible experiment runner
│   └── exp001_log.json            # Structured experiment log
│
├── tests/
│   ├── test_calibration.py
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   ├── test_separation.py
│   ├── test_measurement.py
│   └── test_pipeline_integration.py
│
├── outputs/
│   ├── visualizations/
│   │   ├── original.jpg           # Clean original image
│   │   ├── segmentation.jpg       # Binary mask + boundary edge overlay
│   │   ├── instances.jpg          # Color-coded instance map with unique grain IDs
│   │   └── measurements.jpg       # Oriented bounding boxes, major axes, diagnostic color coding
│   ├── crops/                     # outputs/crops/grain_XXXX.png (individual RGB crops)
│   ├── masks/                     # outputs/masks/grain_XXXX.png (individual binary masks)
│   ├── grain_measurements.csv     # Complete per-grain metrics table
│   ├── summary.json               # Aggregated population statistics and separation metrics
│   └── reports/
│       └── M0_BASELINE_REPORT.md  # Detailed technical experiment report
│
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Recommended: [uv](https://github.com/astral-sh/uv) or standard `venv`

### Setup
```bash
# Clone or navigate to the repository
cd /home/anubhavtripathi/Documents/RiceAge

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### Running the Pipeline
```bash
# Run baseline on standardized sample
python scripts/run_baseline.py

# Or run with custom arguments
python scripts/run_baseline.py \
  --image data/samples/sample_1489x2048.jpg \
  --raw-reference data/raw/laserjet_1200dpi_sample.jpg \
  --output-dir outputs \
  --max-crops 1000
```

### Running Tests
```bash
pytest -v
```

---

## 7. Public Datasets Strategy & Roadmap

In compliance with first principles, defect classification will NOT use invented labels or uncalibrated synthetic data. Public datasets will be evaluated and ingested in Phase M2 according to the following registry:

| Dataset Name | Source / Focus | Annotation Type | Resolution | Classes | Relevance / Notes |
|---|---|---|---|---|---|
| **GrainDet** | Rice & grain defect detection benchmark | Bounding Boxes | High ($2000\times 2000+$) | Whole, Broken, Chalky, Damaged, Foreign | Directly relevant for multi-defect anchor pretraining |
| **GrainBench** | Comprehensive multi-species grain benchmark | Instance Segmentation & Classification | Variable | Broken, Discolored, Unripe, Foreign | Multi-grain impurity benchmarking |
| **Peruvian Rice Dataset** | Milled rice quality assessment | Single grain crops & labels | Standard ($256\times 256$) | Whole, Half-broken, Severe-broken, Chalky | Useful for grain-crop quality classifier head |
| **Kaggle Rice Image Dataset** | Rice cultivar classification (Arborio, Basmati, Jasmine, etc.) | Cropped individual grains | $250\times 250$ | 5 Cultivars | Cultivar identification, not defect classification |

---

## 8. Taxonomy Definition

### Current Real Data Taxonomy
- `WHOLE_RICE`: Grain with length $\ge 75\%$ of median unbroken length and aspect ratio $\ge 1.6$.
- `BROKEN_RICE`: Grain fragment with length $< 75\%$ of median unbroken length.

### Future Target Production Taxonomy
- `WHOLE_RICE`
- `BROKEN_RICE`
- `CHALKY_RICE`
- `DISCOLORED_RICE`
- `DAMAGED_RICE`
- `PEST_DAMAGED`
- `UNRIPE_RICE`
- `FOREIGN_MATERIAL`
- `OTHER_GRAIN`
