#!/usr/bin/env python3
"""
Feature Extraction Script for Sakku Dataset (Sakku.zip).
Parses all 60 high-resolution scans and 28,000+ annotations across 4 annotator batches.
Clusters multi-label annotations to physical grains, normalizes spatial scale to 300 DPI,
extracts 35 morphological, colorimetric, and texture features, and builds:
  1. data/cache_sakku_features.csv (Sakku dataset grains)
  2. data/cache_master_all_features.csv (Combined Sakku + M3 Canonical dataset)
"""

import os
import sys
import io
import time
import zipfile
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from dataclasses import asdict
from typing import Dict, List, Tuple, Set, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.quality_classification import extract_quality_features_from_crop


def box_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def extract_sakku_features(
    zip_path: str = "data/Sakku.zip",
    output_sakku_csv: str = "data/cache_sakku_features.csv",
    output_master_csv: str = "data/cache_master_all_features.csv",
    m3_csv_path: str = "data/cache_m3_all_features.csv"
):
    if not os.path.isabs(zip_path):
        candidates = [
            os.path.join(PROJECT_ROOT, zip_path),
            os.path.join(PROJECT_ROOT, "data", "Sakku.zip"),
            os.path.join(PROJECT_ROOT, "Sakku.zip")
        ]
        for c in candidates:
            if os.path.exists(c):
                zip_path = c
                break
        else:
            zip_path = os.path.join(PROJECT_ROOT, zip_path)
    if not os.path.isabs(output_sakku_csv):
        output_sakku_csv = os.path.join(PROJECT_ROOT, output_sakku_csv)
    if not os.path.isabs(output_master_csv):
        output_master_csv = os.path.join(PROJECT_ROOT, output_master_csv)
    if not os.path.isabs(m3_csv_path):
        m3_csv_path = os.path.join(PROJECT_ROOT, m3_csv_path)

    print("=" * 80)
    print("SAKKU DATASET FEATURE EXTRACTION & PIPELINE SCALING")
    print(f"Source Archive : {zip_path}")
    print(f"Output Sakku   : {output_sakku_csv}")
    print(f"Output Master  : {output_master_csv}")
    print("=" * 80)

    t_start = time.time()
    grain_records = []
    total_scans_processed = 0
    total_annotations_found = 0

    with zipfile.ZipFile(zip_path, 'r') as root_zip:
        inner_zips = sorted([n for n in root_zip.namelist() if n.endswith('.zip')])
        print(f"Found {len(inner_zips)} batch archives in {os.path.basename(zip_path)}: {inner_zips}")

        for zname in inner_zips:
            batch_id = os.path.splitext(os.path.basename(zname))[0]
            print(f"\nProcessing batch: {batch_id}...")
            inner_bytes = root_zip.read(zname)
            
            with zipfile.ZipFile(io.BytesIO(inner_bytes), 'r') as inner_zip:
                # Find annotations XML
                xml_candidates = [n for n in inner_zip.namelist() if n.endswith('annotations.xml')]
                if not xml_candidates:
                    print(f"  [!] No annotations.xml found in {zname}, skipping.")
                    continue
                
                xml_data = inner_zip.read(xml_candidates[0])
                root = ET.fromstring(xml_data)
                img_elements = {el.attrib.get('name').lower(): el for el in root.findall('.//image')}

                # Locate image files
                img_names = [n for n in inner_zip.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"  Found {len(img_names)} scan images in batch.")

                for img_path in sorted(img_names):
                    base_name = os.path.basename(img_path)
                    elem = img_elements.get(base_name.lower())
                    if elem is None:
                        print(f"  [!] No XML record for image {base_name}, skipping.")
                        continue

                    # Read and decode image
                    img_data = inner_zip.read(img_path)
                    nparr = np.frombuffer(img_data, np.uint8)
                    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        print(f"  [!] Failed to decode image {base_name}, skipping.")
                        continue

                    h_img, w_img = img_bgr.shape[:2]
                    # Canonical 300 DPI baseline width is 2481 px
                    scale_to_300dpi = 2481.0 / max(1.0, float(w_img))

                    # Parse all boxes & polygons
                    raw_items = []
                    for child in list(elem.findall('box')):
                        xtl = float(child.attrib['xtl'])
                        ytl = float(child.attrib['ytl'])
                        xbr = float(child.attrib['xbr'])
                        ybr = float(child.attrib['ybr'])
                        lbl = child.attrib.get('label', 'Unknown')
                        raw_items.append({'box': (xtl, ytl, xbr, ybr), 'label': lbl})
                        total_annotations_found += 1

                    for child in list(elem.findall('polygon')):
                        pts_str = child.attrib.get('points', '')
                        lbl = child.attrib.get('label', 'Unknown')
                        if pts_str:
                            coords = [tuple(map(float, pt.split(','))) for pt in pts_str.split(';')]
                            xs = [c[0] for c in coords]
                            ys = [c[1] for c in coords]
                            raw_items.append({'box': (min(xs), min(ys), max(xs), max(ys)), 'label': lbl})
                            total_annotations_found += 1

                    # Spatial clustering of co-located boxes (multi-label grains)
                    grain_clusters: List[Dict[str, Any]] = []
                    for item in raw_items:
                        matched = False
                        for c in grain_clusters:
                            if box_iou(c['box'], item['box']) > 0.40:
                                c['labels'].add(item['label'])
                                c['box'] = (
                                    min(c['box'][0], item['box'][0]),
                                    min(c['box'][1], item['box'][1]),
                                    max(c['box'][2], item['box'][2]),
                                    max(c['box'][3], item['box'][3])
                                )
                                matched = True
                                break
                        if not matched:
                            grain_clusters.append({'box': item['box'], 'labels': {item['label']}})

                    # Extract features for each unique grain
                    extracted_in_scan = 0
                    for g_idx, c in enumerate(grain_clusters):
                        xtl, ytl, xbr, ybr = c['box']
                        labels: Set[str] = c['labels']

                        # Multi-label boolean flags
                        is_whole = 1 if 'Whole' in labels else 0
                        is_broken = 1 if 'Broken' in labels else 0
                        is_fm = 1 if 'Foreign Matter' in labels else 0
                        is_chalky = 1 if 'Chalky' in labels else 0
                        is_damaged = 1 if 'Damaged' in labels else 0
                        is_disc = 1 if 'Discoloured' in labels else 0
                        is_dehusked = 1 if 'Dehusked' in labels else 0

                        # Crop and binarize
                        pad = 6
                        x0 = max(0, int(xtl) - pad)
                        y0 = max(0, int(ytl) - pad)
                        x1 = min(w_img, int(xbr) + pad)
                        y1 = min(h_img, int(ybr) + pad)
                        crop_bgr = img_bgr[y0:y1, x0:x1]
                        if crop_bgr.size == 0 or crop_bgr.shape[0] < 4 or crop_bgr.shape[1] < 4:
                            continue

                        gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                        _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        box_poly = np.array([
                            [int(xtl) - x0, int(ytl) - y0],
                            [int(xbr) - x0, int(ytl) - y0],
                            [int(xbr) - x0, int(ybr) - y0],
                            [int(xtl) - x0, int(ybr) - y0]
                        ], dtype=np.int32)
                        poly_mask = np.zeros(crop_bgr.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(poly_mask, [box_poly], 255)
                        grain_mask = cv2.bitwise_and(binary, binary, mask=poly_mask)

                        gid = f"SAKKU_{batch_id}_{base_name}_{g_idx:04d}"
                        # Primary condition for backward compatibility
                        primary_cond = (
                            "Foreign Matter" if is_fm else
                            ("Damaged" if is_damaged else
                            ("Chalky" if is_chalky else
                            ("Discoloured" if is_disc else
                            ("Broken" if is_broken else
                            ("Dehusked" if is_dehusked else "Whole")))))
                        )

                        feats = extract_quality_features_from_crop(
                            crop_bgr=crop_bgr,
                            grain_mask=grain_mask,
                            grain_id=gid,
                            image_name=base_name,
                            category=batch_id,
                            gt_condition=primary_cond
                        )

                        if feats is not None:
                            d = asdict(feats)
                            # Normalize spatial dimensions to canonical 300 DPI
                            d['length_px'] = d['length_px'] * scale_to_300dpi
                            d['width_px'] = d['width_px'] * scale_to_300dpi
                            d['perimeter_px'] = d['perimeter_px'] * scale_to_300dpi
                            d['area_px'] = d['area_px'] * (scale_to_300dpi ** 2)
                            d['max_defect_depth'] = d['max_defect_depth'] * scale_to_300dpi
                            d['eq_diameter'] = np.sqrt(4.0 * d['area_px'] / np.pi)
                            d['extent'] = d['area_px'] / np.maximum(1e-5, d['length_px'] * d['width_px'])

                            # Attach explicit multi-label flags
                            d['is_whole'] = is_whole
                            d['is_broken'] = is_broken
                            d['is_foreign_matter'] = is_fm
                            d['is_chalky'] = is_chalky
                            d['is_damaged'] = is_damaged
                            d['is_discoloured'] = is_disc
                            d['is_dehusked'] = is_dehusked
                            d['batch_source'] = batch_id
                            d['raw_labels'] = "|".join(sorted(labels))

                            grain_records.append(d)
                            extracted_in_scan += 1

                    total_scans_processed += 1
                    print(f"    {base_name:15s} | Annotations: {len(raw_items):4d} | Grains Extracted: {extracted_in_scan:4d}")

    # Build Sakku DataFrame
    df_sakku = pd.DataFrame(grain_records)
    os.makedirs(os.path.dirname(output_sakku_csv), exist_ok=True)
    df_sakku.to_csv(output_sakku_csv, index=False)
    print("\n" + "=" * 80)
    print(f"Saved {len(df_sakku)} Sakku grain feature vectors to: {output_sakku_csv}")

    # Print summary breakdown
    print("\nSakku Multi-Label Distribution:")
    for col in ['is_whole', 'is_broken', 'is_foreign_matter', 'is_chalky', 'is_damaged', 'is_discoloured', 'is_dehusked']:
        cnt = df_sakku[col].sum()
        pct = (cnt / len(df_sakku)) * 100.0
        print(f"  {col:20s}: {cnt:5d} ({pct:5.2f}%)")

    # Combine with original M3 dataset
    if os.path.exists(m3_csv_path):
        df_m3 = pd.read_csv(m3_csv_path)
        print(f"\nMerging with original M3 dataset ({len(df_m3)} grains from {m3_csv_path})...")
        
        # Add multi-label columns to M3 if not present
        if 'is_foreign_matter' not in df_m3.columns:
            df_m3['is_foreign_matter'] = (df_m3['gt_condition'] == 'Foreign Matter').astype(int)
            df_m3['is_whole'] = (df_m3['gt_condition'] == 'Whole').astype(int)
            df_m3['is_broken'] = (df_m3['gt_condition'] == 'Broken').astype(int)
            df_m3['is_chalky'] = (df_m3['gt_condition'] == 'Chalky').astype(int)
            df_m3['is_damaged'] = (df_m3['gt_condition'] == 'Damaged').astype(int)
            df_m3['is_discoloured'] = (df_m3['gt_condition'] == 'Discoloured').astype(int)
            df_m3['is_dehusked'] = 0
            df_m3['batch_source'] = 'canonical_m3'
            df_m3['raw_labels'] = df_m3['gt_condition']

        if 'extent' not in df_m3.columns:
            df_m3['extent'] = df_m3['area_px'] / np.maximum(1e-5, df_m3['length_px'] * df_m3['width_px'])
        if 'eq_diameter' not in df_m3.columns:
            df_m3['eq_diameter'] = np.sqrt(4.0 * df_m3['area_px'] / np.pi)

        df_master = pd.concat([df_m3, df_sakku], ignore_index=True)
        df_master.to_csv(output_master_csv, index=False)
        print(f"[+] Successfully saved MASTER dataset with {len(df_master)} grains to: {output_master_csv}")
    else:
        df_sakku.to_csv(output_master_csv, index=False)

    elapsed = time.time() - t_start
    print(f"\nCompleted extraction of {len(grain_records)} grains from {total_scans_processed} scans in {elapsed:.1f}s!")
    print("=" * 80)
    return df_sakku


if __name__ == "__main__":
    extract_sakku_features()
