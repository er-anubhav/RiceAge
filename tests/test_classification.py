import pytest
import pandas as pd

from src.classification import classify_grains_rule_based, RealDataTaxonomy


def test_classify_grains_rule_based_whole_and_broken():
    df = pd.DataFrame([
        {
            "grain_id": 1,
            "length_px": 50.0,
            "aspect_ratio": 2.2,
            "area_px": 800,
            "solidity": 0.95,
            "is_boundary": False,
            "separation_status": "isolated_single"
        },
        {
            "grain_id": 2,
            "length_px": 25.0, # significantly shorter than 50.0 -> < 75%
            "aspect_ratio": 1.4,
            "area_px": 350,
            "solidity": 0.94,
            "is_boundary": False,
            "separation_status": "isolated_single"
        },
        {
            "grain_id": 3,
            "length_px": 55.0,
            "aspect_ratio": 2.1,
            "area_px": 850,
            "solidity": 0.96,
            "is_boundary": True, # touches border
            "separation_status": "isolated_single"
        },
        {
            "grain_id": 4,
            "length_px": 80.0,
            "aspect_ratio": 1.5,
            "area_px": 1500,
            "solidity": 0.80,
            "is_boundary": False,
            "separation_status": "under_segmented_cluster" # unresolved cluster
        }
    ])

    res = classify_grains_rule_based(df, broken_length_ratio_thresh=0.75, broken_aspect_ratio_thresh=1.6)
    classes = res["quality_class"].tolist()
    assert classes[0] == RealDataTaxonomy.WHOLE_RICE.value
    assert classes[1] == RealDataTaxonomy.BROKEN_RICE.value
    assert classes[2] == "BOUNDARY_TRUNCATED"
    assert classes[3] == "UNRESOLVED_CLUSTER"


def test_classify_grains_with_reference_anchor():
    # If a sample only has small grains, but reference whole length is provided
    df = pd.DataFrame([
        {
            "grain_id": 1,
            "length_px": 30.0,
            "aspect_ratio": 2.0,
            "area_px": 400,
            "solidity": 0.95,
            "is_boundary": False,
            "separation_status": "isolated_single"
        }
    ])

    # Reference length = 6.0 mm with scale = 0.1 mm/px -> reference whole length is 60 px
    # 30 px < 0.75 * 60 px (45 px), so it should be classified as BROKEN
    res = classify_grains_rule_based(
        df,
        reference_whole_length_mm=6.0,
        pixel_scale_mm=0.1
    )
    assert res.iloc[0]["quality_class"] == RealDataTaxonomy.BROKEN_RICE.value
