# Automated tests for RiceAge M3 quality classification module (src/quality_classification.py).

import numpy as np
import pandas as pd
import pytest
import cv2

from src.quality_classification import (
    GrainQualityFeatures,
    extract_quality_features_from_crop,
    BaselineForeignMatterRuleClassifier,
    BaselineChalkyRuleClassifier,
    BaselineDiscolouredRuleClassifier,
    BaselineDamagedRuleClassifier,
    SupervisedQualityHead,
    get_default_feature_sets
)


def test_feature_extraction_synthetic_crop():
    """Verify feature extraction from a synthetic rice grain mask and BGR crop."""
    # Synthetic white ellipse on dark background
    crop = np.zeros((60, 40, 3), dtype=np.uint8)
    mask = np.zeros((60, 40), dtype=np.uint8)
    cv2.ellipse(crop, (20, 30), (10, 25), 0, 0, 360, (200, 200, 200), -1)
    cv2.ellipse(mask, (20, 30), (10, 25), 0, 0, 360, 255, -1)

    feats = extract_quality_features_from_crop(crop, mask, grain_id="SYN_01")
    assert feats is not None
    assert feats.length_px > 40.0
    assert feats.width_px > 15.0
    assert feats.solidity > 0.90
    assert feats.mean_L > 100.0


def test_baseline_rule_classifiers():
    """Verify classical baseline rule classifications on synthetic dataframes."""
    data = {
        "solidity": [0.95, 0.70],
        "roughness": [0.40, 3.50],
        "yellow_ratio": [0.01, 0.80],
        "aspect_ratio": [2.5, 4.5],
        "area_px": [1500, 3500],
        "mean_L": [160.0, 100.0],
        "chalky_ratio": [0.10, 0.80],
        "high_whiteness": [0.05, 0.60],
        "mean_b": [131.0, 136.0],
        "mean_sat": [12.0, 30.0],
        "dark_spot_ratio": [0.0, 0.05],
        "edge_density": [0.05, 0.12],
        "num_defects": [0, 3]
    }
    df = pd.DataFrame(data)

    # Foreign matter
    fm_clf = BaselineForeignMatterRuleClassifier()
    fm_preds = fm_clf.predict(df)
    assert fm_preds[0] == 0  # Sound rice
    assert fm_preds[1] == 1  # Foreign matter

    # Chalky
    ch_clf = BaselineChalkyRuleClassifier()
    ch_preds = ch_clf.predict(df)
    assert ch_preds[0] == 0
    assert ch_preds[1] == 1

    # Discoloured
    disc_clf = BaselineDiscolouredRuleClassifier()
    disc_preds = disc_clf.predict(df)
    assert disc_preds[0] == 0
    assert disc_preds[1] == 1

    # Damaged
    dmg_clf = BaselineDamagedRuleClassifier()
    dmg_preds = dmg_clf.predict(df)
    assert dmg_preds[0] == 0
    assert dmg_preds[1] == 1


def test_supervised_head_fit_predict():
    """Verify fitting and prediction of supervised quality head."""
    feature_sets = get_default_feature_sets()
    cols = feature_sets["foreign_matter"]

    # Simple training dataframe
    np.random.seed(42)
    train_data = {c: np.random.randn(20) for c in cols}
    train_df = pd.DataFrame(train_data)
    y_train = np.array([0]*10 + [1]*10)

    head = SupervisedQualityHead(feature_cols=cols)
    head.fit(train_df, y_train)

    test_data = {c: np.random.randn(5) for c in cols}
    test_df = pd.DataFrame(test_data)
    preds = head.predict(test_df)
    probs = head.predict_proba(test_df)

    assert len(preds) == 5
    assert len(probs) == 5
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
