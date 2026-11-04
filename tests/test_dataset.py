import pytest
import os
import numpy as np

from src.dataset import (
    load_dataset,
    validate_dataset,
    rasterize_instance_mask,
    rasterize_semantic_mask,
    AnnotatedGrain,
)


def test_validate_dataset():
    val = validate_dataset("data/dataset")
    assert val["is_valid"]
    assert val["total_images"] == 119
    assert val["total_grains"] == 2529
    assert val["missing_raw_images"] == 0
    assert val["missing_annotated_images"] == 0


def test_load_dataset_and_filtering():
    cat = load_dataset("data/dataset")
    assert len(cat) == 119
    assert cat.total_grains == 2529

    whole_imgs = cat.filter_by_category("whole")
    assert len(whole_imgs) == 20

    broken_imgs = cat.filter_by_category("broken")
    assert len(broken_imgs) == 20


def test_dataset_split_no_leakage():
    cat = load_dataset("data/dataset")
    train, val, test = cat.split(seed=42)
    assert len(train) + len(val) + len(test) == 119

    # Verify whole 4.jpg and whole 5.jpg are in the same split
    w4_split = "train" if any(i.filename == "whole 4.jpg" for i in train) else ("val" if any(i.filename == "whole 4.jpg" for i in val) else "test")
    w5_split = "train" if any(i.filename == "whole 5.jpg" for i in train) else ("val" if any(i.filename == "whole 5.jpg" for i in val) else "test")
    assert w4_split == w5_split, "Duplicate images whole 4 and whole 5 must be in the same split"


def test_rasterize_instance_mask():
    g1 = AnnotatedGrain(
        grain_id="G1", image_id=1, image_name="test.jpg", category="whole",
        condition="Whole", points=[(10, 10), (30, 10), (30, 40), (10, 40)],
        bbox_xyxy=(10, 10, 30, 40), polygon_area_px=600.0
    )
    g2 = AnnotatedGrain(
        grain_id="G2", image_id=1, image_name="test.jpg", category="whole",
        condition="Whole", points=[(50, 50), (80, 50), (80, 70), (50, 70)],
        bbox_xyxy=(50, 50, 80, 70), polygon_area_px=600.0
    )

    mask = rasterize_instance_mask([g1, g2], shape=(100, 100))
    assert mask.shape == (100, 100)
    assert mask[20, 20] == 1
    assert mask[60, 60] == 2
    assert mask[0, 0] == 0
