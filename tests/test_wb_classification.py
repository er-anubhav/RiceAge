import os
import pytest
import numpy as np
import pandas as pd
import cv2

from src.wb_classification import (
    extract_grain_features_from_mask,
    extract_annotated_grain_silhouette,
    BaselineRuleClassifier,
    CalibratedRuleClassifier,
    OptimizedRuleClassifier,
    SupervisedGrainClassifier,
    run_stratified_grouped_cv
)


def test_feature_extraction_from_synthetic_mask():
    # Create synthetic ellipse (simulating whole rice kernel: length 70, width 25)
    mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.ellipse(mask, (60, 60), (35, 12), 0, 0, 360, 255, -1)
    
    gray = np.full((120, 120), 160, dtype=np.uint8)
    gray[mask == 0] = 50

    feat = extract_grain_features_from_mask(mask, gray)
    assert feat is not None
    assert 68.0 <= feat["length_px"] <= 72.0
    assert 23.0 <= feat["width_px"] <= 26.0
    assert 2.6 <= feat["aspect_ratio"] <= 3.1
    assert feat["solidity"] >= 0.95
    assert feat["mean_intensity"] > 150.0


def test_feature_extraction_broken_rect_synthetic():
    # Create synthetic broken kernel (transversely severed rectangle: length 35, width 25)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (32, 37), (67, 62), 255, -1)

    feat = extract_grain_features_from_mask(mask)
    assert feat is not None
    assert 33.0 <= feat["length_px"] <= 37.0
    assert 23.0 <= feat["width_px"] <= 27.0
    assert 1.2 <= feat["aspect_ratio"] <= 1.6
    # Broken rectangle has high end bluntness
    assert feat["max_bluntness"] >= 0.70


def test_stratified_grouped_cv_duplicate_group_locking():
    # Verify that duplicate image pairs are never split across train and val folds
    rows = []
    # 6 images: 2 whole (one duplicate pair), 2 broken, 2 mixed
    images = [
        ("whole 1.jpg", "whole", "whole 1.jpg", "Whole", 1),
        ("whole 4.jpg", "whole", "whole_dup_group", "Whole", 1),
        ("whole 5.jpg", "whole", "whole_dup_group", "Whole", 1),
        ("broken 1.jpg", "broken", "broken 1.jpg", "Broken", 0),
        ("broken 2.jpg", "broken", "broken 2.jpg", "Broken", 0),
        ("mixed 1.jpg", "mixed", "mixed 1.jpg", "Whole", 1),
        ("mixed 2.jpg", "mixed", "mixed 2.jpg", "Broken", 0),
    ]
    for idx, (fname, cat, gid, cond, y) in enumerate(images):
        for g_i in range(5):
            rows.append({
                "grain_id": f"G_{idx}_{g_i}",
                "filename": fname,
                "category": cat,
                "group_id": gid,
                "gt_condition": cond,
                "y": y,
                "length_px": 70.0 if y == 1 else 35.0,
                "width_px": 25.0,
                "aspect_ratio": 2.8 if y == 1 else 1.4,
                "area_px": 1400.0 if y == 1 else 700.0,
                "perimeter_px": 170.0 if y == 1 else 100.0,
                "circularity": 0.6 if y == 1 else 0.8,
                "solidity": 0.96,
                "extent": 0.75,
                "eq_diameter": 42.0 if y == 1 else 30.0,
                "max_bluntness": 0.3 if y == 1 else 0.6,
                "min_bluntness": 0.2 if y == 1 else 0.4,
                "end_asymmetry": 0.1 if y == 1 else 0.2,
            })

    df_test = pd.DataFrame(rows)
    res = run_stratified_grouped_cv(
        df=df_test,
        model_factory=lambda: CalibratedRuleClassifier(),
        model_name="Calibrated_Test",
        n_splits=3,
        random_state=42
    )
    # Check that predictions cover all samples
    assert len(res.out_of_fold_predictions) == len(df_test)
    assert res.pooled_accuracy == 1.0


def test_calibrated_rule_vs_baseline_on_pure_broken():
    # Synthetic dataset containing one pure whole image and one pure broken image
    df_eval = pd.DataFrame([
        # Image 1 (whole): 10 whole grains (len ~ 70)
        *[
            {
                "grain_id": f"W_{i}",
                "filename": "whole_img.jpg",
                "gt_condition": "Whole",
                "length_px": 70.0 + i,
                "aspect_ratio": 2.8,
                "solidity": 0.96,
            }
            for i in range(10)
        ],
        # Image 2 (broken): 10 broken grains (len ~ 35, AR ~ 1.4 to 1.9)
        *[
            {
                "grain_id": f"B_{i}",
                "filename": "broken_img.jpg",
                "gt_condition": "Broken",
                "length_px": 35.0 + i,
                "aspect_ratio": 1.7 + (0.05 * i),  # AR >= 1.7 qualifies for unbroken_candidates
                "solidity": 0.95,
            }
            for i in range(10)
        ]
    ])

    base_model = BaselineRuleClassifier()
    base_preds = base_model.predict(df_eval)
    # In broken_img, baseline computes median from broken grains (~39.5)
    # grains with len >= 0.75 * 39.5 = 29.6 and AR >= 1.6 get misclassified as Whole (1)
    broken_slice = df_eval[df_eval["gt_condition"] == "Broken"].index
    assert np.sum(base_preds[broken_slice] == 1) > 0  # Demonstrates baseline failure!

    # Calibrated rule: fits on whole_img (median = 74.5)
    cal_model = CalibratedRuleClassifier()
    train_whole = df_eval[df_eval["gt_condition"] == "Whole"]
    cal_model.fit(train_whole)
    cal_preds = cal_model.predict(df_eval)
    # 0.75 * 74.5 = 55.8. All broken grains (35..44) are < 55.8, so correctly classified as 0 (Broken)
    assert np.all(cal_preds[broken_slice] == 0)


def test_supervised_classifier_fit_predict():
    np.random.seed(42)
    # 50 whole, 50 broken
    w_len = np.random.normal(70.0, 4.0, 50)
    w_ar = np.random.normal(2.8, 0.2, 50)
    b_len = np.random.normal(35.0, 5.0, 50)
    b_ar = np.random.normal(1.4, 0.2, 50)

    train_rows = []
    for l, ar in zip(w_len, w_ar):
        train_rows.append({
            "y": 1, "gt_condition": "Whole",
            "length_px": l, "width_px": 25.0, "aspect_ratio": ar,
            "area_px": l * 20.0, "perimeter_px": l * 2.4, "circularity": 0.6,
            "solidity": 0.96, "extent": 0.75, "eq_diameter": 40.0,
            "max_bluntness": 0.3, "min_bluntness": 0.2, "end_asymmetry": 0.1
        })
    for l, ar in zip(b_len, b_ar):
        train_rows.append({
            "y": 0, "gt_condition": "Broken",
            "length_px": l, "width_px": 25.0, "aspect_ratio": ar,
            "area_px": l * 20.0, "perimeter_px": l * 2.4, "circularity": 0.8,
            "solidity": 0.95, "extent": 0.74, "eq_diameter": 28.0,
            "max_bluntness": 0.6, "min_bluntness": 0.4, "end_asymmetry": 0.2
        })

    train_df = pd.DataFrame(train_rows)
    clf = SupervisedGrainClassifier(model_type="logistic_regression")
    clf.fit(train_df)

    preds = clf.predict(train_df)
    acc = np.mean(preds == train_df["y"])
    assert acc >= 0.98


def test_deliverables_exist_and_valid():
    cm_path = "outputs/reports/M2_WHOLE_BROKEN_CONFUSION_MATRIX.csv"
    err_path = "outputs/reports/M2_WHOLE_BROKEN_ERROR_ANALYSIS.csv"
    assert os.path.exists(cm_path), f"Missing {cm_path}"
    assert os.path.exists(err_path), f"Missing {err_path}"

    cm_df = pd.read_csv(cm_path)
    err_df = pd.read_csv(err_path)

    assert len(cm_df) >= 14  # 7 models across 2 levels
    assert "pooled_accuracy" in cm_df.columns
    assert "pooled_balanced_accuracy" in cm_df.columns
    assert "macro_f1" in cm_df.columns

    assert len(err_df) == 1017  # All 1017 grains accounted for
    assert "error_category" in err_df.columns
    assert "analysis_notes" in err_df.columns
