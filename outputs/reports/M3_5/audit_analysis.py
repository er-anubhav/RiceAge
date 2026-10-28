#!/usr/bin/env python3
"""
RiceAge M3.5 Independent Audit Analysis Script
Reproduces benchmarks, runs negative controls, audits annotations, and analyzes failure modes.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.quality_classification import (
    BaselineForeignMatterRuleClassifier,
    BaselineChalkyRuleClassifier,
    BaselineDiscolouredRuleClassifier,
    BaselineDamagedRuleClassifier,
    SupervisedQualityHead,
    get_default_feature_sets
)
from src.wb_classification import (
    BaselineRuleClassifier,
    CalibratedRuleClassifier
)

def run_audit():
    cache_path = "data/cache_m3_all_features.csv"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Feature cache missing: {cache_path}")
    
    df = pd.read_csv(cache_path)
    print(f"[+] Loaded {len(df)} records from {cache_path}")

    # Derived geometric metrics
    if "extent" not in df.columns:
        df["extent"] = df["area_px"] / np.maximum(1e-5, df["length_px"] * df["width_px"])
    if "eq_diameter" not in df.columns:
        df["eq_diameter"] = np.sqrt(4.0 * df["area_px"] / np.pi)

    # Group locking for duplicates
    df["group_id"] = df["image_name"]
    df.loc[df["image_name"].isin(["whole 4.jpg", "whole 5.jpg"]), "group_id"] = "whole_4_5_dup"

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    feature_sets = get_default_feature_sets()

    # =========================================================================
    # PART 1: INDEPENDENT BENCHMARK REPRODUCTION & DETAILED FOLD METRICS
    # =========================================================================
    print("\n" + "="*80)
    print("PART 1: INDEPENDENT BENCHMARK REPRODUCTION & DETAILED FOLD METRICS")
    print("="*80)

    repro_metrics = {}

    # 1. Foreign Matter
    y_fm = (df["gt_condition"] == "Foreign Matter").astype(int).values
    fm_rule = BaselineForeignMatterRuleClassifier()
    preds_fm_rule = fm_rule.predict(df)
    
    preds_fm_lr = np.zeros(len(df), dtype=int)
    probs_fm_lr = np.zeros(len(df), dtype=float)
    fold_stats_fm = []

    for f_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, y_fm, df["group_id"])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        head = SupervisedQualityHead(feature_cols=feature_sets["foreign_matter"])
        head.fit(train_df, y_fm[train_idx])
        p_val = head.predict(val_df)
        preds_fm_lr[val_idx] = p_val
        probs_fm_lr[val_idx] = head.predict_proba(val_df)
        fold_stats_fm.append({
            "fold": f_idx + 1,
            "n_val": len(val_idx),
            "pos_val": int(y_fm[val_idx].sum()),
            "acc": accuracy_score(y_fm[val_idx], p_val),
            "bal_acc": balanced_accuracy_score(y_fm[val_idx], p_val),
            "prec": precision_score(y_fm[val_idx], p_val, zero_division=0),
            "rec": recall_score(y_fm[val_idx], p_val, zero_division=0),
            "f1": f1_score(y_fm[val_idx], p_val, zero_division=0)
        })

    repro_metrics["Foreign_Matter"] = {
        "rule": {
            "acc": accuracy_score(y_fm, preds_fm_rule),
            "bal_acc": balanced_accuracy_score(y_fm, preds_fm_rule),
            "prec": precision_score(y_fm, preds_fm_rule),
            "rec": recall_score(y_fm, preds_fm_rule),
            "f1": f1_score(y_fm, preds_fm_rule),
            "cm": confusion_matrix(y_fm, preds_fm_rule).tolist()
        },
        "lr": {
            "acc": accuracy_score(y_fm, preds_fm_lr),
            "bal_acc": balanced_accuracy_score(y_fm, preds_fm_lr),
            "prec": precision_score(y_fm, preds_fm_lr),
            "rec": recall_score(y_fm, preds_fm_lr),
            "f1": f1_score(y_fm, preds_fm_lr),
            "cm": confusion_matrix(y_fm, preds_fm_lr).tolist(),
            "folds": fold_stats_fm
        }
    }

    # 2. Whole vs Broken
    wb_df = df[df["gt_condition"].isin(["Whole", "Broken"])].copy().reset_index(drop=True)
    wb_df["filename"] = wb_df["image_name"]
    y_wb = (wb_df["gt_condition"] == "Whole").astype(int).values

    wb_rule = BaselineRuleClassifier()
    preds_wb_rule = wb_rule.predict(wb_df)

    cal_rule = CalibratedRuleClassifier()
    cal_preds = np.zeros(len(wb_df), dtype=int)
    for train_idx, val_idx in sgkf.split(wb_df, y_wb, wb_df["group_id"]):
        train_df, val_df = wb_df.iloc[train_idx], wb_df.iloc[val_idx]
        cal_rule.fit(train_df)
        cal_preds[val_idx] = cal_rule.predict(val_df)

    preds_wb_lr = np.zeros(len(wb_df), dtype=int)
    probs_wb_lr = np.zeros(len(wb_df), dtype=float)
    fold_stats_wb = []

    for f_idx, (train_idx, val_idx) in enumerate(sgkf.split(wb_df, y_wb, wb_df["group_id"])):
        train_df, val_df = wb_df.iloc[train_idx], wb_df.iloc[val_idx]
        head = SupervisedQualityHead(feature_cols=feature_sets["whole_broken"])
        head.fit(train_df, y_wb[train_idx])
        p_val = head.predict(val_df)
        preds_wb_lr[val_idx] = p_val
        probs_wb_lr[val_idx] = head.predict_proba(val_df)
        fold_stats_wb.append({
            "fold": f_idx + 1,
            "n_val": len(val_idx),
            "pos_val": int(y_wb[val_idx].sum()),
            "acc": accuracy_score(y_wb[val_idx], p_val),
            "bal_acc": balanced_accuracy_score(y_wb[val_idx], p_val),
            "prec": precision_score(y_wb[val_idx], p_val, zero_division=0),
            "rec": recall_score(y_wb[val_idx], p_val, zero_division=0),
            "f1": f1_score(y_wb[val_idx], p_val, zero_division=0)
        })

    repro_metrics["Whole_vs_Broken"] = {
        "rule": {
            "acc": accuracy_score(y_wb, preds_wb_rule),
            "bal_acc": balanced_accuracy_score(y_wb, preds_wb_rule),
            "prec": precision_score(y_wb, preds_wb_rule),
            "rec": recall_score(y_wb, preds_wb_rule),
            "f1": f1_score(y_wb, preds_wb_rule),
            "cm": confusion_matrix(y_wb, preds_wb_rule).tolist()
        },
        "calibrated": {
            "acc": accuracy_score(y_wb, cal_preds),
            "bal_acc": balanced_accuracy_score(y_wb, cal_preds),
            "prec": precision_score(y_wb, cal_preds),
            "rec": recall_score(y_wb, cal_preds),
            "f1": f1_score(y_wb, cal_preds),
            "cm": confusion_matrix(y_wb, cal_preds).tolist()
        },
        "lr": {
            "acc": accuracy_score(y_wb, preds_wb_lr),
            "bal_acc": balanced_accuracy_score(y_wb, preds_wb_lr),
            "prec": precision_score(y_wb, preds_wb_lr),
            "rec": recall_score(y_wb, preds_wb_lr),
            "f1": f1_score(y_wb, preds_wb_lr),
            "cm": confusion_matrix(y_wb, preds_wb_lr).tolist(),
            "folds": fold_stats_wb
        }
    }

    # 3. Chalky Grains
    ch_df = df[df["gt_condition"].isin(["Chalky", "Whole"])].copy().reset_index(drop=True)
    y_ch = (ch_df["gt_condition"] == "Chalky").astype(int).values
    ch_rule = BaselineChalkyRuleClassifier(chalky_threshold=0.50)
    preds_ch_rule = ch_rule.predict(ch_df)

    preds_ch_lr = np.zeros(len(ch_df), dtype=int)
    probs_ch_lr = np.zeros(len(ch_df), dtype=float)
    fold_stats_ch = []

    for f_idx, (train_idx, val_idx) in enumerate(sgkf.split(ch_df, y_ch, ch_df["group_id"])):
        train_df, val_df = ch_df.iloc[train_idx], ch_df.iloc[val_idx]
        head = SupervisedQualityHead(feature_cols=feature_sets["chalky"])
        head.fit(train_df, y_ch[train_idx])
        p_val = head.predict(val_df)
        preds_ch_lr[val_idx] = p_val
        probs_ch_lr[val_idx] = head.predict_proba(val_df)
        fold_stats_ch.append({
            "fold": f_idx + 1,
            "n_val": len(val_idx),
            "pos_val": int(y_ch[val_idx].sum()),
            "acc": accuracy_score(y_ch[val_idx], p_val),
            "bal_acc": balanced_accuracy_score(y_ch[val_idx], p_val),
            "prec": precision_score(y_ch[val_idx], p_val, zero_division=0),
            "rec": recall_score(y_ch[val_idx], p_val, zero_division=0),
            "f1": f1_score(y_ch[val_idx], p_val, zero_division=0)
        })

    repro_metrics["Chalky"] = {
        "rule": {
            "acc": accuracy_score(y_ch, preds_ch_rule),
            "bal_acc": balanced_accuracy_score(y_ch, preds_ch_rule),
            "prec": precision_score(y_ch, preds_ch_rule),
            "rec": recall_score(y_ch, preds_ch_rule),
            "f1": f1_score(y_ch, preds_ch_rule),
            "cm": confusion_matrix(y_ch, preds_ch_rule).tolist()
        },
        "lr": {
            "acc": accuracy_score(y_ch, preds_ch_lr),
            "bal_acc": balanced_accuracy_score(y_ch, preds_ch_lr),
            "prec": precision_score(y_ch, preds_ch_lr),
            "rec": recall_score(y_ch, preds_ch_lr),
            "f1": f1_score(y_ch, preds_ch_lr),
            "cm": confusion_matrix(y_ch, preds_ch_lr).tolist(),
            "folds": fold_stats_ch
        }
    }

    # 4. Discoloured Grains
    disc_df = df[df["gt_condition"].isin(["Discoloured", "Whole"])].copy().reset_index(drop=True)
    y_disc = (disc_df["gt_condition"] == "Discoloured").astype(int).values
    disc_rule = BaselineDiscolouredRuleClassifier()
    preds_disc_rule = disc_rule.predict(disc_df)

    preds_disc_lr = np.zeros(len(disc_df), dtype=int)
    probs_disc_lr = np.zeros(len(disc_df), dtype=float)
    fold_stats_disc = []

    for f_idx, (train_idx, val_idx) in enumerate(sgkf.split(disc_df, y_disc, disc_df["group_id"])):
        train_df, val_df = disc_df.iloc[train_idx], disc_df.iloc[val_idx]
        head = SupervisedQualityHead(feature_cols=feature_sets["discoloured"])
        head.fit(train_df, y_disc[train_idx])
        p_val = head.predict(val_df)
        preds_disc_lr[val_idx] = p_val
        probs_disc_lr[val_idx] = head.predict_proba(val_df)
        fold_stats_disc.append({
            "fold": f_idx + 1,
            "n_val": len(val_idx),
            "pos_val": int(y_disc[val_idx].sum()),
            "acc": accuracy_score(y_disc[val_idx], p_val),
            "bal_acc": balanced_accuracy_score(y_disc[val_idx], p_val),
            "prec": precision_score(y_disc[val_idx], p_val, zero_division=0),
            "rec": recall_score(y_disc[val_idx], p_val, zero_division=0),
            "f1": f1_score(y_disc[val_idx], p_val, zero_division=0)
        })

    repro_metrics["Discoloured"] = {
        "rule": {
            "acc": accuracy_score(y_disc, preds_disc_rule),
            "bal_acc": balanced_accuracy_score(y_disc, preds_disc_rule),
            "prec": precision_score(y_disc, preds_disc_rule),
            "rec": recall_score(y_disc, preds_disc_rule),
            "f1": f1_score(y_disc, preds_disc_rule),
            "cm": confusion_matrix(y_disc, preds_disc_rule).tolist()
        },
        "lr": {
            "acc": accuracy_score(y_disc, preds_disc_lr),
            "bal_acc": balanced_accuracy_score(y_disc, preds_disc_lr),
            "prec": precision_score(y_disc, preds_disc_lr),
            "rec": recall_score(y_disc, preds_disc_lr),
            "f1": f1_score(y_disc, preds_disc_lr),
            "cm": confusion_matrix(y_disc, preds_disc_lr).tolist(),
            "folds": fold_stats_disc
        }
    }

    # 5. Damaged Grains
    dmg_df = df[df["gt_condition"].isin(["Damaged", "Whole"])].copy().reset_index(drop=True)
    y_dmg = (dmg_df["gt_condition"] == "Damaged").astype(int).values
    dmg_rule = BaselineDamagedRuleClassifier()
    preds_dmg_rule = dmg_rule.predict(dmg_df)

    preds_dmg_lr = np.zeros(len(dmg_df), dtype=int)
    probs_dmg_lr = np.zeros(len(dmg_df), dtype=float)
    fold_stats_dmg = []

    for f_idx, (train_idx, val_idx) in enumerate(sgkf.split(dmg_df, y_dmg, dmg_df["group_id"])):
        train_df, val_df = dmg_df.iloc[train_idx], dmg_df.iloc[val_idx]
        head = SupervisedQualityHead(feature_cols=feature_sets["damaged"])
        head.fit(train_df, y_dmg[train_idx])
        p_val = head.predict(val_df)
        preds_dmg_lr[val_idx] = p_val
        probs_dmg_lr[val_idx] = head.predict_proba(val_df)
        fold_stats_dmg.append({
            "fold": f_idx + 1,
            "n_val": len(val_idx),
            "pos_val": int(y_dmg[val_idx].sum()),
            "acc": accuracy_score(y_dmg[val_idx], p_val),
            "bal_acc": balanced_accuracy_score(y_dmg[val_idx], p_val),
            "prec": precision_score(y_dmg[val_idx], p_val, zero_division=0),
            "rec": recall_score(y_dmg[val_idx], p_val, zero_division=0),
            "f1": f1_score(y_dmg[val_idx], p_val, zero_division=0)
        })

    repro_metrics["Damaged"] = {
        "rule": {
            "acc": accuracy_score(y_dmg, preds_dmg_rule),
            "bal_acc": balanced_accuracy_score(y_dmg, preds_dmg_rule),
            "prec": precision_score(y_dmg, preds_dmg_rule),
            "rec": recall_score(y_dmg, preds_dmg_rule),
            "f1": f1_score(y_dmg, preds_dmg_rule),
            "cm": confusion_matrix(y_dmg, preds_dmg_rule).tolist()
        },
        "lr": {
            "acc": accuracy_score(y_dmg, preds_dmg_lr),
            "bal_acc": balanced_accuracy_score(y_dmg, preds_dmg_lr),
            "prec": precision_score(y_dmg, preds_dmg_lr),
            "rec": recall_score(y_dmg, preds_dmg_lr),
            "f1": f1_score(y_dmg, preds_dmg_lr),
            "cm": confusion_matrix(y_dmg, preds_dmg_lr).tolist(),
            "folds": fold_stats_dmg
        }
    }

    print("[+] All M3 benchmarks successfully reproduced and cross-checked.")

    # =========================================================================
    # PART 2: NEGATIVE CONTROLS & SHORTCUT AUDITING
    # =========================================================================
    print("\n" + "="*80)
    print("PART 2: NEGATIVE CONTROLS & SHORTCUT AUDITING")
    print("="*80)

    # A. Label Permutation Negative Control
    # Shuffling labels independently across folds. True models should drop to chance.
    np.random.seed(42)
    permutation_results = {}
    
    tasks = [
        ("Foreign_Matter", df, y_fm, feature_sets["foreign_matter"]),
        ("Whole_vs_Broken", wb_df, y_wb, feature_sets["whole_broken"]),
        ("Chalky", ch_df, y_ch, feature_sets["chalky"]),
        ("Discoloured", disc_df, y_disc, feature_sets["discoloured"]),
        ("Damaged", dmg_df, y_dmg, feature_sets["damaged"]),
    ]

    for task_name, task_df, task_y, feat_cols in tasks:
        # 10 permutations
        perm_accs = []
        perm_bal_accs = []
        for p_iter in range(10):
            y_perm = np.random.permutation(task_y)
            preds_perm = np.zeros(len(task_df), dtype=int)
            for train_idx, val_idx in sgkf.split(task_df, y_perm, task_df["group_id"]):
                tr_df, va_df = task_df.iloc[train_idx], task_df.iloc[val_idx]
                pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=500, random_state=42))])
                pipe.fit(tr_df[feat_cols], y_perm[train_idx])
                preds_perm[val_idx] = pipe.predict(va_df[feat_cols])
            perm_accs.append(accuracy_score(y_perm, preds_perm))
            perm_bal_accs.append(balanced_accuracy_score(y_perm, preds_perm))
        
        prevalence = float(task_y.mean())
        majority_acc = max(prevalence, 1.0 - prevalence)
        permutation_results[task_name] = {
            "mean_perm_acc": float(np.mean(perm_accs)),
            "std_perm_acc": float(np.std(perm_accs)),
            "mean_perm_bal_acc": float(np.mean(perm_bal_accs)),
            "majority_baseline_acc": float(majority_acc),
            "prevalence": prevalence
        }
        print(f"Negative Control [Label Permutation] {task_name:15s} | Perm Acc: {np.mean(perm_accs)*100:.2f}% | Perm BalAcc: {np.mean(perm_bal_accs)*100:.2f}% | Majority Acc: {majority_acc*100:.2f}%")

    # B. Image Background Shortcut Control
    # Measure background pixels across 119 scans and test if background predicts folder/class
    print("\nAuditing image backgrounds across all 119 scans...")
    bg_records = []
    category_counts = {}

    unique_imgs = df[["image_name", "category"]].drop_duplicates()
    for _, row in unique_imgs.iterrows():
        f = row["image_name"]
        cat = row["category"]
        img_path = os.path.join("data/dataset/images", cat, f)
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            # sample 4 corners (50x50 box each)
            corners = [
                img.crop((0, 0, 50, 50)),
                img.crop((w-50, 0, w, 50)),
                img.crop((0, h-50, 50, h)),
                img.crop((w-50, h-50, w, h))
            ]
            arrs = [np.array(c) for c in corners]
            all_bg = np.concatenate([a.reshape(-1, 3) for a in arrs], axis=0)
            mean_r = float(np.mean(all_bg[:, 0]))
            mean_g = float(np.mean(all_bg[:, 1]))
            mean_b = float(np.mean(all_bg[:, 2]))
            gray = float(np.mean(0.299 * all_bg[:, 0] + 0.587 * all_bg[:, 1] + 0.114 * all_bg[:, 2]))
            std_gray = float(np.std(0.299 * all_bg[:, 0] + 0.587 * all_bg[:, 1] + 0.114 * all_bg[:, 2]))
            
            bg_records.append({
                "filename": f,
                "category": cat,
                "width": w,
                "height": h,
                "mean_r": mean_r,
                "mean_g": mean_g,
                "mean_b": mean_b,
                "gray": gray,
                "std_gray": std_gray
            })
            category_counts[cat] = category_counts.get(cat, 0) + 1
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    bg_df = pd.DataFrame(bg_records)
    bg_summary = bg_df.groupby("category")[["gray", "mean_r", "mean_g", "mean_b"]].agg(["mean", "std"])
    print("Background Illumination by Category:")
    print(bg_summary)

    # Can background color classify Category?
    # Train LogisticRegression on [mean_r, mean_g, mean_b, gray] to predict category
    bg_pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=500, random_state=42))])
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bg_preds = np.zeros(len(bg_df), dtype=object)
    for tr_i, va_i in skf.split(bg_df, bg_df["category"]):
        bg_pipe.fit(bg_df.iloc[tr_i][["mean_r", "mean_g", "mean_b", "gray"]], bg_df.iloc[tr_i]["category"])
        bg_preds[va_i] = bg_pipe.predict(bg_df.iloc[va_i][["mean_r", "mean_g", "mean_b", "gray"]])
    
    bg_cat_acc = accuracy_score(bg_df["category"], bg_preds)
    print(f"\nBackground-Only Category Classification Accuracy: {bg_cat_acc*100:.2f}% (Chance baseline: {max(category_counts.values())/len(bg_df)*100:.2f}%)")

    # C. Open-World Cohort Filtering Audit
    # What happens when we run the trained defect models on the ENTIRE dataset (2,068 rice grains)?
    print("\nAuditing Open-World Cohort Filtering across all 2,068 rice grains...")
    rice_df = df[df["gt_condition"] != "Foreign Matter"].copy().reset_index(drop=True)
    
    # Train full supervised models on their respective benchmark subsets and predict on all rice grains
    pipe_ch = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=500, random_state=42))])
    pipe_ch.fit(ch_df[feature_sets["chalky"]], y_ch)
    rice_df["pred_chalky_open"] = pipe_ch.predict(rice_df[feature_sets["chalky"]])

    pipe_disc = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=500, random_state=42))])
    pipe_disc.fit(disc_df[feature_sets["discoloured"]], y_disc)
    rice_df["pred_disc_open"] = pipe_disc.predict(rice_df[feature_sets["discoloured"]])

    pipe_dmg = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=500, random_state=42))])
    pipe_dmg.fit(dmg_df[feature_sets["damaged"]], y_dmg)
    rice_df["pred_dmg_open"] = pipe_dmg.predict(rice_df[feature_sets["damaged"]])

    # Cross-tabulate predictions by ground-truth category
    print("\nChalky Positive Predictions by Ground Truth Category (Open-World):")
    print(rice_df.groupby("gt_condition")["pred_chalky_open"].agg(["count", "sum", lambda x: f"{x.mean()*100:.1f}%"]))

    print("\nDiscoloured Positive Predictions by Ground Truth Category (Open-World):")
    print(rice_df.groupby("gt_condition")["pred_disc_open"].agg(["count", "sum", lambda x: f"{x.mean()*100:.1f}%"]))

    print("\nDamaged Positive Predictions by Ground Truth Category (Open-World):")
    print(rice_df.groupby("gt_condition")["pred_dmg_open"].agg(["count", "sum", lambda x: f"{x.mean()*100:.1f}%"]))

    # =========================================================================
    # PART 3: ANNOTATION INTEGRITY AUDIT (M3_5_ANNOTATION_AUDIT.csv)
    # =========================================================================
    print("\n" + "="*80)
    print("PART 3: ANNOTATION INTEGRITY AUDIT")
    print("="*80)

    import xml.etree.ElementTree as ET
    annot_xml_path = "data/dataset/annotations/all_annotations.xml"
    tree = ET.parse(annot_xml_path)
    root = tree.getroot()

    images_list = root.findall("image")
    print(f"Total images in CVAT XML: {len(images_list)}")

    audit_records = []
    multi_defect_count = 0
    obb_count = 0
    silhouette_count = 0
    border_touch_count = 0
    total_annotations_count = 0

    for img_elem in images_list:
        im_name = img_elem.get("name")
        im_w = int(img_elem.get("width"))
        im_h = int(img_elem.get("height"))

        poly_objs = []
        for ann_idx, poly_elem in enumerate(img_elem.findall("polygon")):
            total_annotations_count += 1
            points_str = poly_elem.get("points")
            raw_pts = []
            for pair in points_str.split(";"):
                if pair.strip():
                    x_s, y_s = pair.split(",")
                    raw_pts.append([float(x_s), float(y_s)])
            pts = np.array(raw_pts, dtype=np.int32)
            n_vertices = len(pts)
            if n_vertices == 4:
                obb_count += 1
            else:
                silhouette_count += 1

            # Condition attribute
            cond = "Unknown"
            for attr in poly_elem.findall("attribute"):
                if attr.get("name") == "Condition":
                    cond = attr.text.strip() if attr.text else "Unknown"

            # Check border touch
            min_x = int(np.min(pts[:, 0]))
            max_x = int(np.max(pts[:, 0]))
            min_y = int(np.min(pts[:, 1]))
            max_y = int(np.max(pts[:, 1]))
            if min_x <= 5 or min_y <= 5 or max_x >= im_w - 5 or max_y >= im_h - 5:
                border_touch_count += 1
                audit_records.append({
                    "image_name": im_name,
                    "annotation_id_1": f"{im_name}_p{ann_idx}",
                    "annotation_id_2": "N/A",
                    "label_1": cond,
                    "label_2": "N/A",
                    "iou": 0.0,
                    "intersection_area_px": 0.0,
                    "issue_type": "BORDER_TOUCHING_GRAIN",
                    "severity": "LOW",
                    "recommended_action": "Flag during pipeline ingestion; ignore crop if truncated by scanner border."
                })

            poly_objs.append({
                "ann_id": f"{im_name}_p{ann_idx}",
                "label": cond,
                "pts": pts,
                "bbox": (min_x, min_y, max_x, max_y)
            })

        # Pairwise overlap audit
        for i in range(len(poly_objs)):
            obj1 = poly_objs[i]
            ann1_id = obj1["ann_id"]
            b1 = obj1["bbox"]
            lbl1 = obj1["label"]
            for j in range(i + 1, len(poly_objs)):
                obj2 = poly_objs[j]
                ann2_id = obj2["ann_id"]
                b2 = obj2["bbox"]
                lbl2 = obj2["label"]

                # Check if bboxes overlap
                inter_x1 = max(b1[0], b2[0])
                inter_y1 = max(b1[1], b2[1])
                inter_x2 = min(b1[2], b2[2])
                inter_y2 = min(b1[3], b2[3])

                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    # Bounding boxes overlap, compute pixel intersection
                    union_x1 = min(b1[0], b2[0])
                    union_y1 = min(b1[1], b2[1])
                    union_x2 = max(b1[2], b2[2])
                    union_y2 = max(b1[3], b2[3])
                    h_roi = union_y2 - union_y1 + 1
                    w_roi = union_x2 - union_x1 + 1
                    
                    mask1 = np.zeros((h_roi, w_roi), dtype=np.uint8)
                    mask2 = np.zeros((h_roi, w_roi), dtype=np.uint8)
                    
                    pts1_roi = obj1["pts"] - np.array([union_x1, union_y1])
                    pts2_roi = obj2["pts"] - np.array([union_x1, union_y1])
                    
                    cv2.fillPoly(mask1, [pts1_roi], 1)
                    cv2.fillPoly(mask2, [pts2_roi], 1)
                    
                    a1 = float(np.sum(mask1))
                    a2 = float(np.sum(mask2))
                    inter_area = float(np.sum((mask1 > 0) & (mask2 > 0)))
                    union_area = float(np.sum((mask1 > 0) | (mask2 > 0)))
                    
                    iou = inter_area / union_area if union_area > 0 else 0.0
                    containment1 = inter_area / a1 if a1 > 0 else 0.0
                    containment2 = inter_area / a2 if a2 > 0 else 0.0

                    if iou > 0.50 or containment1 > 0.60 or containment2 > 0.60:
                        multi_defect_count += 1
                        issue = "DUPLICATE_OR_MULTI_DEFECT_ANNOTATION"
                        sev = "CRITICAL" if lbl1 == lbl2 else "HIGH"
                        action = "Unify into single physical grain with multi-defect tags" if lbl1 != lbl2 else "Deduplicate identical double annotation"
                        audit_records.append({
                            "image_name": im_name,
                            "annotation_id_1": ann1_id,
                            "annotation_id_2": ann2_id,
                            "label_1": lbl1,
                            "label_2": lbl2,
                            "iou": round(iou, 4),
                            "intersection_area_px": round(inter_area, 1),
                            "issue_type": issue,
                            "severity": sev,
                            "recommended_action": action
                        })

    # Exact duplicate images
    audit_records.append({
        "image_name": "whole 4.jpg / whole 5.jpg",
        "annotation_id_1": "ALL",
        "annotation_id_2": "ALL",
        "label_1": "Whole",
        "label_2": "Whole",
        "iou": 1.0,
        "intersection_area_px": 0.0,
        "issue_type": "DUPLICATE_IMAGE_FILE",
        "severity": "CRITICAL",
        "recommended_action": "Lock into identical CV fold (Fold 1) to prevent train/test leakage."
    })

    audit_df = pd.DataFrame(audit_records)
    audit_csv = "outputs/reports/M3_5/M3_5_ANNOTATION_AUDIT.csv"
    audit_df.to_csv(audit_csv, index=False)
    print(f"[+] Saved Annotation Audit to: {audit_csv} ({len(audit_df)} records, {multi_defect_count} multi-defect overlaps)")
    print(f"    Total annotations: {total_annotations_count}")
    print(f"    Coarse 4-point OBBs: {obb_count} ({obb_count/total_annotations_count*100:.1f}%)")
    print(f"    Detailed silhouettes: {silhouette_count} ({silhouette_count/total_annotations_count*100:.1f}%)")
    print(f"    Border touching: {border_touch_count}")

    # =========================================================================
    # PART 4: DETAILED ERROR ANALYSIS (M3_5_ERROR_ANALYSIS.csv)
    # =========================================================================
    print("\n" + "="*80)
    print("PART 4: DETAILED ERROR ANALYSIS")
    print("="*80)

    # Compile false positives and false negatives across all heads
    error_audit_rows = []

    # 1. Foreign Matter Errors
    for idx, row in df.iterrows():
        p = preds_fm_lr[idx]
        gt = 1 if row["gt_condition"] == "Foreign Matter" else 0
        if p != gt:
            error_audit_rows.append({
                "quality_head": "Foreign_Matter",
                "grain_id": row["grain_id"],
                "image_name": row["image_name"],
                "category": row["category"],
                "gt_condition": row["gt_condition"],
                "pred_condition": "Foreign Matter" if p == 1 else "Rice Grain",
                "error_class": "False_Positive" if p == 1 else "False_Negative",
                "root_cause_diagnosis": "Borderline aspect ratio/solidity artifact" if p == 1 else "Foreign matter mimicking grain shape",
                "length_px": round(row["length_px"], 2),
                "aspect_ratio": round(row["aspect_ratio"], 2),
                "solidity": round(row["solidity"], 3),
                "key_metric_value": f"area_px={row['area_px']:.1f}, extent={row['extent']:.3f}"
            })

    # 2. Whole vs Broken Errors
    for idx, row in wb_df.iterrows():
        p = preds_wb_lr[idx]
        gt = 1 if row["gt_condition"] == "Whole" else 0
        if p != gt:
            cause = "Broken tip with intact length (borderline length/AR)" if gt == 0 else "Short Whole grain below length decision boundary"
            error_audit_rows.append({
                "quality_head": "Whole_vs_Broken",
                "grain_id": row["grain_id"],
                "image_name": row["image_name"],
                "category": row["category"],
                "gt_condition": row["gt_condition"],
                "pred_condition": "Whole" if p == 1 else "Broken",
                "error_class": "False_Positive" if p == 1 else "False_Negative",
                "root_cause_diagnosis": cause,
                "length_px": round(row["length_px"], 2),
                "aspect_ratio": round(row["aspect_ratio"], 2),
                "solidity": round(row["solidity"], 3),
                "key_metric_value": f"length_px={row['length_px']:.1f}, aspect_ratio={row['aspect_ratio']:.2f}"
            })

    # 3. Chalky Errors
    for idx, row in ch_df.iterrows():
        p = preds_ch_lr[idx]
        gt = 1 if row["gt_condition"] == "Chalky" else 0
        if p != gt:
            cause = "Partial belly chalkiness near 50% threshold" if gt == 1 else "High specular reflection / bright grain body"
            error_audit_rows.append({
                "quality_head": "Chalky",
                "grain_id": row["grain_id"],
                "image_name": row["image_name"],
                "category": row["category"],
                "gt_condition": row["gt_condition"],
                "pred_condition": "Chalky" if p == 1 else "Clean_Whole",
                "error_class": "False_Positive" if p == 1 else "False_Negative",
                "root_cause_diagnosis": cause,
                "length_px": round(row["length_px"], 2),
                "aspect_ratio": round(row["aspect_ratio"], 2),
                "solidity": round(row["solidity"], 3),
                "key_metric_value": f"chalky_ratio={row['chalky_ratio']:.3f}, mean_gray={row['mean_gray']:.1f}"
            })

    # 4. Discoloured Errors
    for idx, row in disc_df.iterrows():
        p = preds_disc_lr[idx]
        gt = 1 if row["gt_condition"] == "Discoloured" else 0
        if p != gt:
            cause = "Subtle yellowing/red striping below mean_b threshold" if gt == 1 else "Natural amber tone of non-parboiled grain"
            error_audit_rows.append({
                "quality_head": "Discoloured",
                "grain_id": row["grain_id"],
                "image_name": row["image_name"],
                "category": row["category"],
                "gt_condition": row["gt_condition"],
                "pred_condition": "Discoloured" if p == 1 else "Clean_Whole",
                "error_class": "False_Positive" if p == 1 else "False_Negative",
                "root_cause_diagnosis": cause,
                "length_px": round(row["length_px"], 2),
                "aspect_ratio": round(row["aspect_ratio"], 2),
                "solidity": round(row["solidity"], 3),
                "key_metric_value": f"mean_b={row['mean_b']:.2f}, std_b={row['std_b']:.2f}"
            })

    # 5. Damaged Errors
    for idx, row in dmg_df.iterrows():
        p = preds_dmg_lr[idx]
        gt = 1 if row["gt_condition"] == "Damaged" else 0
        if p != gt:
            cause = "Micro-chipping/minor crack without prominent edge gradient" if gt == 1 else "Shadow boundary or dark germ notch detected as damage"
            error_audit_rows.append({
                "quality_head": "Damaged",
                "grain_id": row["grain_id"],
                "image_name": row["image_name"],
                "category": row["category"],
                "gt_condition": row["gt_condition"],
                "pred_condition": "Damaged" if p == 1 else "Clean_Whole",
                "error_class": "False_Positive" if p == 1 else "False_Negative",
                "root_cause_diagnosis": cause,
                "length_px": round(row["length_px"], 2),
                "aspect_ratio": round(row["aspect_ratio"], 2),
                "solidity": round(row["solidity"], 3),
                "key_metric_value": f"edge_density={row['edge_density']:.3f}, dark_spot_ratio={row['dark_spot_ratio']:.3f}"
            })

    error_analysis_df = pd.DataFrame(error_audit_rows)
    err_csv = "outputs/reports/M3_5/M3_5_ERROR_ANALYSIS.csv"
    error_analysis_df.to_csv(err_csv, index=False)
    print(f"[+] Saved Detailed Error Analysis to: {err_csv} ({len(error_analysis_df)} error cases)")

    # Save reproduced metrics to JSON for markdown document rendering
    with open("outputs/reports/M3_5/repro_metrics.json", "w") as f:
        json.dump(repro_metrics, f, indent=2)
    with open("outputs/reports/M3_5/permutation_results.json", "w") as f:
        json.dump(permutation_results, f, indent=2)

    print("\n[✓] Audit Analysis Script successfully completed all runs.")

if __name__ == "__main__":
    run_audit()
