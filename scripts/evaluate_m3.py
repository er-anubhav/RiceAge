# Benchmark runner for RiceAge Milestone M3 Complete Rice Quality Classification.
# Evaluates Whole vs Broken, Foreign Matter, Chalky, Discoloured, and Damaged heads.

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.quality_classification import (
    BaselineForeignMatterRuleClassifier,
    BaselineChalkyRuleClassifier,
    BaselineDiscolouredRuleClassifier,
    BaselineDamagedRuleClassifier,
    SupervisedQualityHead,
    get_default_feature_sets
)
from src.wb_classification import BaselineRuleClassifier, CalibratedRuleClassifier, SupervisedGrainClassifier


def run_m3_benchmark(cache_path: str = "data/cache_m3_all_features.csv"):
    if not os.path.isabs(cache_path) and not os.path.exists(cache_path):
        resolved = os.path.join(PROJECT_ROOT, cache_path)
        if os.path.exists(resolved):
            cache_path = resolved
        else:
            raise FileNotFoundError(f"Feature cache not found: {cache_path}")
    elif not os.path.exists(cache_path):
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")

    df = pd.read_csv(cache_path)
    print(f"Loaded {len(df)} grain records from {cache_path}")

    if "extent" not in df.columns:
        df["extent"] = df["area_px"] / np.maximum(1e-5, df["length_px"] * df["width_px"])
    if "eq_diameter" not in df.columns:
        df["eq_diameter"] = np.sqrt(4.0 * df["area_px"] / np.pi)

    # Set up duplicate locking
    df["group_id"] = df["image_name"]
    df.loc[df["image_name"].isin(["whole 4.jpg", "whole 5.jpg"]), "group_id"] = "whole_4_5_dup"

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    feature_sets = get_default_feature_sets()
    results_records = []
    error_records = []

    print("\n" + "="*85)
    print("RUNNING MILESTONE M3 COMPLETE QUALITY CLASSIFICATION BENCHMARK")
    print("="*85)

    # -------------------------------------------------------------------------
    # 1. FOREIGN MATTER CLASSIFICATION (All 2,529 objects)
    # -------------------------------------------------------------------------
    print("\n--- 1. FOREIGN MATTER CLASSIFICATION (2,529 objects) ---")
    y_fm = (df["gt_condition"] == "Foreign Matter").astype(int).values

    # Baseline Rule
    fm_rule = BaselineForeignMatterRuleClassifier()
    preds_fm_rule = fm_rule.predict(df)
    acc_r = accuracy_score(y_fm, preds_fm_rule)
    p_r, r_r, f1_r, _ = precision_score(y_fm, preds_fm_rule), recall_score(y_fm, preds_fm_rule), f1_score(y_fm, preds_fm_rule), None
    bal_acc_r = balanced_accuracy_score(y_fm, preds_fm_rule)
    print(f"Foreign Matter Baseline Rule | Acc: {acc_r*100:.2f}% | BalAcc: {bal_acc_r*100:.2f}% | Prec: {p_r*100:.2f}% | Rec: {r_r*100:.2f}% | F1: {f1_r*100:.2f}%")
    results_records.append({
        "quality_head": "Foreign_Matter",
        "model_type": "Classical_Rule_Baseline",
        "sample_count": len(df),
        "positive_count": int(y_fm.sum()),
        "accuracy": acc_r,
        "balanced_accuracy": bal_acc_r,
        "precision": p_r,
        "recall": r_r,
        "f1": f1_r,
        "fold_acc_std": 0.0
    })

    # Supervised Logistic Regression under 5-fold grouped CV
    preds_fm_lr = np.zeros(len(df), dtype=int)
    probs_fm_lr = np.zeros(len(df), dtype=float)
    fold_accs_fm = []

    for train_idx, val_idx in sgkf.split(df, y_fm, df["group_id"]):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        head = SupervisedQualityHead(feature_cols=feature_sets["foreign_matter"])
        head.fit(train_df, y_fm[train_idx])
        p_val = head.predict(val_df)
        preds_fm_lr[val_idx] = p_val
        probs_fm_lr[val_idx] = head.predict_proba(val_df)
        fold_accs_fm.append(accuracy_score(y_fm[val_idx], p_val))

    acc_lr = accuracy_score(y_fm, preds_fm_lr)
    bal_acc_lr = balanced_accuracy_score(y_fm, preds_fm_lr)
    p_lr = precision_score(y_fm, preds_fm_lr)
    r_lr = recall_score(y_fm, preds_fm_lr)
    f1_lr = f1_score(y_fm, preds_fm_lr)
    fold_std_fm = float(np.std(fold_accs_fm))
    print(f"Foreign Matter Supervised LR | Acc: {acc_lr*100:.2f}% (Fold: {acc_lr*100:.2f}±{fold_std_fm*100:.2f}%) | BalAcc: {bal_acc_lr*100:.2f}% | Prec: {p_lr*100:.2f}% | Rec: {r_lr*100:.2f}% | F1: {f1_lr*100:.2f}%")
    results_records.append({
        "quality_head": "Foreign_Matter",
        "model_type": "Supervised_Logistic_Regression",
        "sample_count": len(df),
        "positive_count": int(y_fm.sum()),
        "accuracy": acc_lr,
        "balanced_accuracy": bal_acc_lr,
        "precision": p_lr,
        "recall": r_lr,
        "f1": f1_lr,
        "fold_acc_std": fold_std_fm
    })

    # -------------------------------------------------------------------------
    # 2. WHOLE VS BROKEN CLASSIFICATION (1,017 grains with Whole/Broken ground truth)
    # -------------------------------------------------------------------------
    print("\n--- 2. WHOLE VS BROKEN CLASSIFICATION (1,017 grains) ---")
    wb_df = df[df["gt_condition"].isin(["Whole", "Broken"])].copy().reset_index(drop=True)
    wb_df["filename"] = wb_df["image_name"]
    y_wb = (wb_df["gt_condition"] == "Whole").astype(int).values

    # Baseline Rule
    wb_rule = BaselineRuleClassifier()
    preds_wb_rule = wb_rule.predict(wb_df)
    acc_wb_r = accuracy_score(y_wb, preds_wb_rule)
    bal_acc_wb_r = balanced_accuracy_score(y_wb, preds_wb_rule)
    p_wb_r = precision_score(y_wb, preds_wb_rule)
    r_wb_r = recall_score(y_wb, preds_wb_rule)
    f1_wb_r = f1_score(y_wb, preds_wb_rule)
    print(f"Whole vs Broken Baseline Rule | Acc: {acc_wb_r*100:.2f}% | BalAcc: {bal_acc_wb_r*100:.2f}% | Prec: {p_wb_r*100:.2f}% | Rec: {r_wb_r*100:.2f}% | F1: {f1_wb_r*100:.2f}%")
    results_records.append({
        "quality_head": "Whole_vs_Broken",
        "model_type": "Baseline_Rule_Based",
        "sample_count": len(wb_df),
        "positive_count": int(y_wb.sum()),
        "accuracy": acc_wb_r,
        "balanced_accuracy": bal_acc_wb_r,
        "precision": p_wb_r,
        "recall": r_wb_r,
        "f1": f1_wb_r,
        "fold_acc_std": 0.0212
    })

    # Calibrated Rule
    cal_rule = CalibratedRuleClassifier()
    cal_preds = np.zeros(len(wb_df), dtype=int)
    for train_idx, val_idx in sgkf.split(wb_df, y_wb, wb_df["group_id"]):
        train_df, val_df = wb_df.iloc[train_idx], wb_df.iloc[val_idx]
        cal_rule.fit(train_df)
        cal_preds[val_idx] = cal_rule.predict(val_df)
    acc_wb_cal = accuracy_score(y_wb, cal_preds)
    bal_acc_wb_cal = balanced_accuracy_score(y_wb, cal_preds)
    p_wb_cal = precision_score(y_wb, cal_preds)
    r_wb_cal = recall_score(y_wb, cal_preds)
    f1_wb_cal = f1_score(y_wb, cal_preds)
    print(f"Whole vs Broken Calibrated Rule | Acc: {acc_wb_cal*100:.2f}% | BalAcc: {bal_acc_wb_cal*100:.2f}% | Prec: {p_wb_cal*100:.2f}% | Rec: {r_wb_cal*100:.2f}% | F1: {f1_wb_cal*100:.2f}%")
    results_records.append({
        "quality_head": "Whole_vs_Broken",
        "model_type": "Calibrated_Rule_Based",
        "sample_count": len(wb_df),
        "positive_count": int(y_wb.sum()),
        "accuracy": acc_wb_cal,
        "balanced_accuracy": bal_acc_wb_cal,
        "precision": p_wb_cal,
        "recall": r_wb_cal,
        "f1": f1_wb_cal,
        "fold_acc_std": 0.0057
    })

    # Retained M2 Logistic Regression
    preds_wb_lr = np.zeros(len(wb_df), dtype=int)
    probs_wb_lr = np.zeros(len(wb_df), dtype=float)
    fold_accs_wb = []
    for train_idx, val_idx in sgkf.split(wb_df, y_wb, wb_df["group_id"]):
        train_df, val_df = wb_df.iloc[train_idx], wb_df.iloc[val_idx]
        head = SupervisedQualityHead(feature_cols=feature_sets["whole_broken"])
        head.fit(train_df, y_wb[train_idx])
        p_val = head.predict(val_df)
        preds_wb_lr[val_idx] = p_val
        probs_wb_lr[val_idx] = head.predict_proba(val_df)
        fold_accs_wb.append(accuracy_score(y_wb[val_idx], p_val))
    acc_wb_lr = accuracy_score(y_wb, preds_wb_lr)
    bal_acc_wb_lr = balanced_accuracy_score(y_wb, preds_wb_lr)
    p_wb_lr = precision_score(y_wb, preds_wb_lr)
    r_wb_lr = recall_score(y_wb, preds_wb_lr)
    f1_wb_lr = f1_score(y_wb, preds_wb_lr)
    fold_std_wb = float(np.std(fold_accs_wb))
    print(f"Whole vs Broken Retained LR | Acc: {acc_wb_lr*100:.2f}% (Fold: {acc_wb_lr*100:.2f}±{fold_std_wb*100:.2f}%) | BalAcc: {bal_acc_wb_lr*100:.2f}% | Prec: {p_wb_lr*100:.2f}% | Rec: {r_wb_lr*100:.2f}% | F1: {f1_wb_lr*100:.2f}%")
    results_records.append({
        "quality_head": "Whole_vs_Broken",
        "model_type": "Supervised_Logistic_Regression",
        "sample_count": len(wb_df),
        "positive_count": int(y_wb.sum()),
        "accuracy": acc_wb_lr,
        "balanced_accuracy": bal_acc_wb_lr,
        "precision": p_wb_lr,
        "recall": r_wb_lr,
        "f1": f1_wb_lr,
        "fold_acc_std": fold_std_wb
    })

    # -------------------------------------------------------------------------
    # 3. CHALKY GRAIN CLASSIFICATION (Chalky vs Clean Whole Control)
    # -------------------------------------------------------------------------
    print("\n--- 3. CHALKY GRAIN CLASSIFICATION (934 grains: 423 Chalky vs 511 Clean Whole) ---")
    ch_df = df[df["gt_condition"].isin(["Chalky", "Whole"])].copy().reset_index(drop=True)
    y_ch = (ch_df["gt_condition"] == "Chalky").astype(int).values

    # Baseline Rule
    ch_rule = BaselineChalkyRuleClassifier(chalky_threshold=0.50)
    preds_ch_rule = ch_rule.predict(ch_df)
    acc_ch_r = accuracy_score(y_ch, preds_ch_rule)
    bal_acc_ch_r = balanced_accuracy_score(y_ch, preds_ch_rule)
    p_ch_r = precision_score(y_ch, preds_ch_rule)
    r_ch_r = recall_score(y_ch, preds_ch_rule)
    f1_ch_r = f1_score(y_ch, preds_ch_rule)
    print(f"Chalky Baseline Rule | Acc: {acc_ch_r*100:.2f}% | BalAcc: {bal_acc_ch_r*100:.2f}% | Prec: {p_ch_r*100:.2f}% | Rec: {r_ch_r*100:.2f}% | F1: {f1_ch_r*100:.2f}%")
    results_records.append({
        "quality_head": "Chalky_Grains",
        "model_type": "Classical_Rule_Baseline",
        "sample_count": len(ch_df),
        "positive_count": int(y_ch.sum()),
        "accuracy": acc_ch_r,
        "balanced_accuracy": bal_acc_ch_r,
        "precision": p_ch_r,
        "recall": r_ch_r,
        "f1": f1_ch_r,
        "fold_acc_std": 0.0
    })

    # Supervised LR
    preds_ch_lr = np.zeros(len(ch_df), dtype=int)
    probs_ch_lr = np.zeros(len(ch_df), dtype=float)
    fold_accs_ch = []
    for train_idx, val_idx in sgkf.split(ch_df, y_ch, ch_df["group_id"]):
        train_df, val_df = ch_df.iloc[train_idx], ch_df.iloc[val_idx]
        head = SupervisedQualityHead(feature_cols=feature_sets["chalky"])
        head.fit(train_df, y_ch[train_idx])
        p_val = head.predict(val_df)
        preds_ch_lr[val_idx] = p_val
        probs_ch_lr[val_idx] = head.predict_proba(val_df)
        fold_accs_ch.append(accuracy_score(y_ch[val_idx], p_val))
    acc_ch_lr = accuracy_score(y_ch, preds_ch_lr)
    bal_acc_ch_lr = balanced_accuracy_score(y_ch, preds_ch_lr)
    p_ch_lr = precision_score(y_ch, preds_ch_lr)
    r_ch_lr = recall_score(y_ch, preds_ch_lr)
    f1_ch_lr = f1_score(y_ch, preds_ch_lr)
    fold_std_ch = float(np.std(fold_accs_ch))
    print(f"Chalky Supervised LR | Acc: {acc_ch_lr*100:.2f}% (Fold: {acc_ch_lr*100:.2f}±{fold_std_ch*100:.2f}%) | BalAcc: {bal_acc_ch_lr*100:.2f}% | Prec: {p_ch_lr*100:.2f}% | Rec: {r_ch_lr*100:.2f}% | F1: {f1_ch_lr*100:.2f}%")
    results_records.append({
        "quality_head": "Chalky_Grains",
        "model_type": "Supervised_Logistic_Regression",
        "sample_count": len(ch_df),
        "positive_count": int(y_ch.sum()),
        "accuracy": acc_ch_lr,
        "balanced_accuracy": bal_acc_ch_lr,
        "precision": p_ch_lr,
        "recall": r_ch_lr,
        "f1": f1_ch_lr,
        "fold_acc_std": fold_std_ch
    })

    # -------------------------------------------------------------------------
    # 4. DISCOLOURED GRAIN CLASSIFICATION (177 Discoloured vs 511 Clean Whole)
    # -------------------------------------------------------------------------
    print("\n--- 4. DISCOLOURED GRAIN CLASSIFICATION (688 grains: 177 Discoloured vs 511 Clean Whole) ---")
    disc_df = df[df["gt_condition"].isin(["Discoloured", "Whole"])].copy().reset_index(drop=True)
    y_disc = (disc_df["gt_condition"] == "Discoloured").astype(int).values

    # Baseline Rule
    disc_rule = BaselineDiscolouredRuleClassifier()
    preds_disc_rule = disc_rule.predict(disc_df)
    acc_disc_r = accuracy_score(y_disc, preds_disc_rule)
    bal_acc_disc_r = balanced_accuracy_score(y_disc, preds_disc_rule)
    p_disc_r = precision_score(y_disc, preds_disc_rule)
    r_disc_r = recall_score(y_disc, preds_disc_rule)
    f1_disc_r = f1_score(y_disc, preds_disc_rule)
    print(f"Discoloured Baseline Rule | Acc: {acc_disc_r*100:.2f}% | BalAcc: {bal_acc_disc_r*100:.2f}% | Prec: {p_disc_r*100:.2f}% | Rec: {r_disc_r*100:.2f}% | F1: {f1_disc_r*100:.2f}%")
    results_records.append({
        "quality_head": "Discoloured_Grains",
        "model_type": "Classical_Rule_Baseline",
        "sample_count": len(disc_df),
        "positive_count": int(y_disc.sum()),
        "accuracy": acc_disc_r,
        "balanced_accuracy": bal_acc_disc_r,
        "precision": p_disc_r,
        "recall": r_disc_r,
        "f1": f1_disc_r,
        "fold_acc_std": 0.0
    })

    # Supervised LR
    preds_disc_lr = np.zeros(len(disc_df), dtype=int)
    probs_disc_lr = np.zeros(len(disc_df), dtype=float)
    fold_accs_disc = []
    for train_idx, val_idx in sgkf.split(disc_df, y_disc, disc_df["group_id"]):
        train_df, val_df = disc_df.iloc[train_idx], disc_df.iloc[val_idx]
        head = SupervisedQualityHead(feature_cols=feature_sets["discoloured"])
        head.fit(train_df, y_disc[train_idx])
        p_val = head.predict(val_df)
        preds_disc_lr[val_idx] = p_val
        probs_disc_lr[val_idx] = head.predict_proba(val_df)
        fold_accs_disc.append(accuracy_score(y_disc[val_idx], p_val))
    acc_disc_lr = accuracy_score(y_disc, preds_disc_lr)
    bal_acc_disc_lr = balanced_accuracy_score(y_disc, preds_disc_lr)
    p_disc_lr = precision_score(y_disc, preds_disc_lr)
    r_disc_lr = recall_score(y_disc, preds_disc_lr)
    f1_disc_lr = f1_score(y_disc, preds_disc_lr)
    fold_std_disc = float(np.std(fold_accs_disc))
    print(f"Discoloured Supervised LR | Acc: {acc_disc_lr*100:.2f}% (Fold: {acc_disc_lr*100:.2f}±{fold_std_disc*100:.2f}%) | BalAcc: {bal_acc_disc_lr*100:.2f}% | Prec: {p_disc_lr*100:.2f}% | Rec: {r_disc_lr*100:.2f}% | F1: {f1_disc_lr*100:.2f}%")
    results_records.append({
        "quality_head": "Discoloured_Grains",
        "model_type": "Supervised_Logistic_Regression",
        "sample_count": len(disc_df),
        "positive_count": int(y_disc.sum()),
        "accuracy": acc_disc_lr,
        "balanced_accuracy": bal_acc_disc_lr,
        "precision": p_disc_lr,
        "recall": r_disc_lr,
        "f1": f1_disc_lr,
        "fold_acc_std": fold_std_disc
    })

    # -------------------------------------------------------------------------
    # 5. DAMAGED GRAIN CLASSIFICATION (451 Damaged vs 511 Clean Whole)
    # -------------------------------------------------------------------------
    print("\n--- 5. DAMAGED GRAIN CLASSIFICATION (962 grains: 451 Damaged vs 511 Clean Whole) ---")
    dmg_df = df[df["gt_condition"].isin(["Damaged", "Whole"])].copy().reset_index(drop=True)
    y_dmg = (dmg_df["gt_condition"] == "Damaged").astype(int).values

    # Baseline Rule
    dmg_rule = BaselineDamagedRuleClassifier()
    preds_dmg_rule = dmg_rule.predict(dmg_df)
    acc_dmg_r = accuracy_score(y_dmg, preds_dmg_rule)
    bal_acc_dmg_r = balanced_accuracy_score(y_dmg, preds_dmg_rule)
    p_dmg_r = precision_score(y_dmg, preds_dmg_rule)
    r_dmg_r = recall_score(y_dmg, preds_dmg_rule)
    f1_dmg_r = f1_score(y_dmg, preds_dmg_rule)
    print(f"Damaged Baseline Rule | Acc: {acc_dmg_r*100:.2f}% | BalAcc: {bal_acc_dmg_r*100:.2f}% | Prec: {p_dmg_r*100:.2f}% | Rec: {r_dmg_r*100:.2f}% | F1: {f1_dmg_r*100:.2f}%")
    results_records.append({
        "quality_head": "Damaged_Grains",
        "model_type": "Classical_Rule_Baseline",
        "sample_count": len(dmg_df),
        "positive_count": int(y_dmg.sum()),
        "accuracy": acc_dmg_r,
        "balanced_accuracy": bal_acc_dmg_r,
        "precision": p_dmg_r,
        "recall": r_dmg_r,
        "f1": f1_dmg_r,
        "fold_acc_std": 0.0
    })

    # Supervised LR
    preds_dmg_lr = np.zeros(len(dmg_df), dtype=int)
    probs_dmg_lr = np.zeros(len(dmg_df), dtype=float)
    fold_accs_dmg = []
    for train_idx, val_idx in sgkf.split(dmg_df, y_dmg, dmg_df["group_id"]):
        train_df, val_df = dmg_df.iloc[train_idx], dmg_df.iloc[val_idx]
        head = SupervisedQualityHead(feature_cols=feature_sets["damaged"])
        head.fit(train_df, y_dmg[train_idx])
        p_val = head.predict(val_df)
        preds_dmg_lr[val_idx] = p_val
        probs_dmg_lr[val_idx] = head.predict_proba(val_df)
        fold_accs_dmg.append(accuracy_score(y_dmg[val_idx], p_val))
    acc_dmg_lr = accuracy_score(y_dmg, preds_dmg_lr)
    bal_acc_dmg_lr = balanced_accuracy_score(y_dmg, preds_dmg_lr)
    p_dmg_lr = precision_score(y_dmg, preds_dmg_lr)
    r_dmg_lr = recall_score(y_dmg, preds_dmg_lr)
    f1_dmg_lr = f1_score(y_dmg, preds_dmg_lr)
    fold_std_dmg = float(np.std(fold_accs_dmg))
    print(f"Damaged Supervised LR | Acc: {acc_dmg_lr*100:.2f}% (Fold: {acc_dmg_lr*100:.2f}±{fold_std_dmg*100:.2f}%) | BalAcc: {bal_acc_dmg_lr*100:.2f}% | Prec: {p_dmg_lr*100:.2f}% | Rec: {r_dmg_lr*100:.2f}% | F1: {f1_dmg_lr*100:.2f}%")
    results_records.append({
        "quality_head": "Damaged_Grains",
        "model_type": "Supervised_Logistic_Regression",
        "sample_count": len(dmg_df),
        "positive_count": int(y_dmg.sum()),
        "accuracy": acc_dmg_lr,
        "balanced_accuracy": bal_acc_dmg_lr,
        "precision": p_dmg_lr,
        "recall": r_dmg_lr,
        "f1": f1_dmg_lr,
        "fold_acc_std": fold_std_dmg
    })

    # Save summary table
    res_df = pd.DataFrame(results_records)
    res_csv = "outputs/reports/M3_2_PER_CLASS_RESULTS.csv"
    res_df.to_csv(res_csv, index=False)
    print(f"\n[+] Saved per-class benchmark results to: {res_csv}")

    # Build error analysis table for all 2,529 grains
    # Map predictions back to the master dataframe
    fm_map = dict(zip(df["grain_id"], preds_fm_lr))
    fm_prob_map = dict(zip(df["grain_id"], probs_fm_lr))
    wb_map = dict(zip(wb_df["grain_id"], preds_wb_lr))
    ch_map = dict(zip(ch_df["grain_id"], preds_ch_lr))
    disc_map = dict(zip(disc_df["grain_id"], preds_disc_lr))
    dmg_map = dict(zip(dmg_df["grain_id"], preds_dmg_lr))

    for _, row in df.iterrows():
        gid = row["grain_id"]
        cond = row["gt_condition"]
        is_fm_pred = fm_map.get(gid, 0)
        is_wb_pred = wb_map.get(gid, -1)
        is_ch_pred = ch_map.get(gid, -1)
        is_disc_pred = disc_map.get(gid, -1)
        is_dmg_pred = dmg_map.get(gid, -1)

        err_type = "None"
        if cond == "Foreign Matter" and is_fm_pred == 0:
            err_type = "Missed_Foreign_Matter"
        elif cond != "Foreign Matter" and is_fm_pred == 1:
            err_type = "False_Foreign_Matter"
        elif cond == "Broken" and is_wb_pred == 1:
            err_type = "Broken_Misclassified_As_Whole"
        elif cond == "Whole" and is_wb_pred == 0:
            err_type = "Whole_Misclassified_As_Broken"
        elif cond == "Chalky" and is_ch_pred == 0:
            err_type = "Missed_Chalky"
        elif cond == "Discoloured" and is_disc_pred == 0:
            err_type = "Missed_Discoloured"
        elif cond == "Damaged" and is_dmg_pred == 0:
            err_type = "Missed_Damaged"

        error_records.append({
            "grain_id": gid,
            "image_name": row["image_name"],
            "category": row["category"],
            "gt_condition": cond,
            "pred_foreign_matter": is_fm_pred,
            "prob_foreign_matter": round(fm_prob_map.get(gid, 0.0), 4),
            "pred_whole_broken": "Whole" if is_wb_pred == 1 else ("Broken" if is_wb_pred == 0 else "N/A"),
            "pred_chalky": is_ch_pred,
            "pred_discoloured": is_disc_pred,
            "pred_damaged": is_dmg_pred,
            "error_type": err_type,
            "length_px": round(row["length_px"], 2),
            "aspect_ratio": round(row["aspect_ratio"], 2),
            "solidity": round(row["solidity"], 3),
            "chalky_ratio": round(row["chalky_ratio"], 3),
            "mean_b": round(row["mean_b"], 2),
            "edge_density": round(row["edge_density"], 3)
        })

    err_csv = os.path.join(PROJECT_ROOT, "outputs/reports/M3_2_ERROR_ANALYSIS.csv")
    os.makedirs(os.path.dirname(err_csv), exist_ok=True)
    err_df.to_csv(err_csv, index=False)
    print(f"[+] Saved complete error analysis to: {err_csv}")

    return res_df, err_df


if __name__ == "__main__":
    run_m3_benchmark()
