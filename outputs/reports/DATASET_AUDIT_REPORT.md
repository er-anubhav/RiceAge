# RiceAge: Verified Dataset Audit Report

**Date:** 2026-10-10  
**Audit Author:** Senior Computer Vision Engineer, RiceAge  
**Target Dataset:** `data/dataset/`  
**Dataset Name:** RiceAge Defect & Quality Ground-Truth Dataset (Manifest v1.0.0)  

---

## 1. Executive Summary

A comprehensive automated and manual verification of `data/dataset/` was performed on 2026-10-10. All raw image files, overlay images, CVAT XML 1.1 annotation files, and auxiliary CSV metadata tables were parsed and cross-referenced.

| Metric | Verified Value | Discrepancy / Audit Note |
|---|---|---|
| **Total Scanner Images** | **119** | Matches manifest and metadata.csv exactly |
| **Total Annotated Visual Overlays** | **119** | Matches images 1:1 in `annotated_images/` |
| **Total Polygon Annotations** | **2,529** | Exactly matches CVAT XML and `grains_annotations.csv` |
| **Image Resolution** | **2481 x 3507 px** | Uniform across all 119 images (Standard A4 flatbed scan) |
| **Color Space / Channels** | **24-bit RGB (3 channels)** | Standard JPEG RGB images |
| **Physical Resolution** | **300 DPI** | $0.084667\text{ mm/pixel}$ ($11.81\text{ pixels/mm}$) |
| **Image Duplicates** | **1 Exact Duplicate Pair** | `whole 5.jpg` has identical file hash to `whole 4.jpg` |
| **Annotation Consistency** | **100% Agreement** | `all_annotations.xml` sum matches per-category XMLs and CSVs |
| **Malformed Coordinates** | **0** | All coordinates valid and bounded within $[0, 2481] \times [0, 3507]$ |
| **Polygon Vertices** | **91.0% Quadrilaterals** | 2,301 are 4-point boxes; 228 are 3 to 14 points |
| **License & Provenance** | **Undocumented** | No explicit LICENSE file, DOI, citation, or author metadata found |

---

## 2. Directory Layout & Storage

```
data/dataset/
├── images/                        # 119 original raw JPEG scanner images
│   ├── broken/                    # 20 images (broken 1.jpg .. broken 20.jpg)
│   ├── chalky/                    # 20 images (chalky 1.jpg .. chalky 20.jpg)
│   ├── damaged/                   # 19 images (Damaged 1.jpg .. Damaged 19.jpg)
│   ├── foreign_matter/            # 20 images (foreign 1.jpg .. foreign 20.jpg)
│   ├── mixed/                     # 20 images (mixed 1.jpg .. mixed 20.jpg)
│   └── whole/                     # 20 images (whole 1.jpg .. whole 20.jpg)
├── annotated_images/              # 119 visual overlay ground truth PNGs
│   ├── broken/                    # 20 images
│   ├── chalky/                    # 20 images
│   ├── damaged/                    # 19 images
│   ├── foreign_matter/            # 20 images
│   ├── mixed/                     # 20 images
│   └── whole/                     # 20 images
├── annotations/                   # CVAT XML 1.1 annotation files
│   ├── broken.xml                 # 398 instances
│   ├── chalky.xml                 # 462 instances
│   ├── damaged.xml                # 371 instances
│   ├── foreign_matter.xml         # 369 instances
│   ├── mixed.xml                  # 534 instances
│   ├── whole.xml                  # 395 instances
│   └── all_annotations.xml       # 2,529 instances (merged)
├── metadata.csv                   # Image-level index table (119 rows)
├── grains_annotations.csv         # Granular per-grain annotation records (2,529 rows)
├── manifest.json                  # Machine-readable JSON manifest
└── README.md                      # Dataset documentation
```

---

## 3. Condition Breakdown & Class Distribution

The dataset provides 6 condition labels recorded as polygon attributes in CVAT XML:

| Condition | Instance Count | % of Dataset | Mean Instances / Image | Min / Max per Image |
|---|---|---|---|---|
| **Whole** | 511 | 20.21% | 4.29 | 0 / 20 |
| **Broken** | 506 | 20.01% | 4.25 | 0 / 20 |
| **Foreign Matter** | 461 | 18.23% | 3.87 | 0 / 29 |
| **Damaged** | 451 | 17.83% | 3.79 | 0 / 20 |
| **Chalky** | 423 | 16.73% | 3.55 | 0 / 25 |
| **Discoloured** | 177 | 7.00% | 1.49 | 0 / 18 |
| **Total** | **2,529** | **100.00%** | **21.25** | **11 / 37** |

### Category-Level Image Distribution

| Category | Images | Total Grains | Predominant Conditions Present |
|---|---|---|---|
| `broken` | 20 | 398 | Broken (398) |
| `chalky` | 20 | 462 | Chalky (423), Whole (39) |
| `damaged` | 19 | 371 | Damaged (371) |
| `foreign_matter` | 20 | 369 | Foreign Matter (369) |
| `mixed` | 20 | 534 | Discoloured (177), Damaged (80), Broken (108), Foreign Matter (92), Whole (77) |
| `whole` | 20 | 395 | Whole (395) |

---

## 4. Polygon Geometry & Annotation Characteristics

- **Bounding Coordinates:**
  - $X \in [11.9, 2369.14]\text{ px}$ (within width 2481 px)
  - $Y \in [557.17, 3380.37]\text{ px}$ (within height 3507 px)
- **Polygon Polygon Areas:**
  - Minimum area: $325.0\text{ px}$
  - Median area: $3,646.4\text{ px}$
  - Mean area: $5,281.5\text{ px}$
  - Maximum area: $93,277.5\text{ px}$ (large foreign debris / paper fragment)
- **Polygon Vertex Distribution:**
  - 4 vertices (oriented bounding polygon): **2,301** (90.98%)
  - 5 vertices: **142** (5.61%)
  - 6 vertices: **33** (1.30%)
  - 7 vertices: **28** (1.11%)
  - 3 vertices: **16** (0.63%)
  - 8+ vertices: **9** (0.36%)
- **Key Observation:** The annotations are primarily tightly oriented 4-point bounding polygons rather than pixel-perfect multi-point boundary contours. Therefore, IoU matching between classical CV fitted contours and GT polygons should use an appropriate threshold (e.g. $\text{IoU} \ge 0.50$ or $\ge 0.40$).

---

## 5. Identified Data Integrity Warnings & Anomalies

1. **Exact Duplicate Image:**
   - `data/dataset/images/whole/whole 5.jpg` and `data/dataset/images/whole/whole 4.jpg` share the identical MD5 hash (`b7b43a9b9a674d8bb8568c4a1eb6f8cb`).
   - The polygon annotations for the two images differ slightly (independent annotation passes on the same physical image).
   - *Actionable Recommendation:* When partitioning into train/val/test splits, `whole 4.jpg` and `whole 5.jpg` must be kept in the same split partition to prevent data leakage.
2. **Missing Category Image in `damaged`:**
   - The `damaged` folder contains 19 images (`Damaged 1.jpg` to `Damaged 19.jpg`), whereas all other folders contain 20 images. This is expected and documented in the manifest.
3. **Scanner Background & Margin Artifacts:**
   - The scanner background is dark gray (intensity ~50–60), not absolute black.
   - Saturated white scanner glass bezel edges occur at the platter edges on several images (e.g. 29 px margin on `broken 12.jpg`).
   - *Pipeline Sensitivity:* Zeroing the bezel margin creates an artificial spike at intensity 0 that corrupts global Otsu thresholding on sparse images. The pipeline must compute the threshold on the active scan ROI.
4. **License & Provenance Notice:**
   - No explicit Open Source license (e.g., CC-BY-4.0, MIT) or academic publication DOI is included in `data/dataset/README.md`.
   - The dataset must be treated as proprietary research data until formal external licensing is confirmed.
