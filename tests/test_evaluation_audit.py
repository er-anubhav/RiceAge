# Comprehensive unit tests for RiceAge evaluation framework audit (Milestone M1.1).

import pytest
import numpy as np
import pandas as pd

from src.evaluation import (
    compute_pairwise_overlap_matrices,
    evaluate_image_instances,
    DatasetImageRecord,
    AnnotatedGrain,
)
from src.dataset import DatasetCatalog


def test_hungarian_one_to_one_matching_strictness():
    # Verify that Hungarian matching strictly enforces 1-to-1 bipartite assignment.
    # Case: 2 GT grains, 1 Pred grain that overlaps both.
    # Exactly one GT should be matched (TP=1), the other remains unmatched (FN=1).
    pred_mask = np.zeros((100, 100), dtype=np.int32)
    pred_mask[20:60, 20:40] = 1 # Covers rows 20-60 (height 40, width 20)

    gt_mask = np.zeros((100, 100), dtype=np.int32)
    gt_mask[20:38, 20:40] = 1 # GT1: rows 20-38 (area 360)
    gt_mask[42:60, 20:40] = 2 # GT2: rows 42-60 (area 360)

    iou_mat, cont_mat = compute_pairwise_overlap_matrices(
        pred_instance_mask=pred_mask,
        gt_instance_mask=gt_mask,
        pred_ids=[1],
        gt_ids=[1, 2]
    )

    g1 = AnnotatedGrain(
        grain_id="G1", image_id=1, image_name="test.jpg", category="whole",
        condition="Whole", points=[(20, 20), (40, 20), (40, 38), (20, 38)],
        bbox_xyxy=(20, 20, 40, 38), polygon_area_px=360.0
    )
    g2 = AnnotatedGrain(
        grain_id="G2", image_id=1, image_name="test.jpg", category="whole",
        condition="Whole", points=[(20, 42), (40, 42), (40, 60), (20, 60)],
        bbox_xyxy=(20, 42, 40, 60), polygon_area_px=360.0
    )
    rec = DatasetImageRecord(
        image_id=1, filename="test.jpg", category="whole",
        width=100, height=100, total_grains=2, counts_by_condition={"Whole": 2},
        image_path="", annotated_image_path="", annotation_xml_path="", grains=[g1, g2]
    )

    pred_df = pd.DataFrame([{"grain_id": 1, "quality_class": "WHOLE_RICE"}])

    metrics = evaluate_image_instances(
        pred_instance_mask=pred_mask,
        pred_classified_df=pred_df,
        gt_record=rec,
        matching_mode="containment",
        containment_threshold=0.30
    )

    assert metrics.gt_count == 2
    assert metrics.pred_count == 1
    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 0
    assert metrics.under_segmented_merges == 1
    assert pytest.approx(metrics.precision, 0.01) == 1.0
    assert pytest.approx(metrics.recall, 0.01) == 0.50


def test_duplicate_predictions_penalized_as_false_positives():
    # Verify that multiple predictions covering a single ground-truth grain
    # do NOT inflate recall. Exactly one is TP, and surplus predictions are FP.
    gt_mask = np.zeros((100, 100), dtype=np.int32)
    gt_mask[20:50, 20:50] = 1 # Area 900 px

    pred_mask = np.zeros((100, 100), dtype=np.int32)
    pred_mask[20:34, 20:50] = 1 # Pred 1 (inside GT)
    pred_mask[36:50, 20:50] = 2 # Pred 2 (inside GT)

    g1 = AnnotatedGrain(
        grain_id="G1", image_id=1, image_name="test.jpg", category="whole",
        condition="Whole", points=[(20, 20), (50, 20), (50, 50), (20, 50)],
        bbox_xyxy=(20, 20, 50, 50), polygon_area_px=900.0
    )
    rec = DatasetImageRecord(
        image_id=1, filename="test.jpg", category="whole",
        width=100, height=100, total_grains=1, counts_by_condition={"Whole": 1},
        image_path="", annotated_image_path="", annotation_xml_path="", grains=[g1]
    )

    pred_df = pd.DataFrame([
        {"grain_id": 1, "quality_class": "WHOLE_RICE"},
        {"grain_id": 2, "quality_class": "WHOLE_RICE"}
    ])

    metrics = evaluate_image_instances(
        pred_instance_mask=pred_mask,
        pred_classified_df=pred_df,
        gt_record=rec,
        matching_mode="containment",
        containment_threshold=0.50
    )

    assert metrics.gt_count == 1
    assert metrics.pred_count == 2
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.over_segmented_splits == 1
    assert pytest.approx(metrics.precision, 0.01) == 0.50
    assert pytest.approx(metrics.recall, 0.01) == 1.00


def test_sub_threshold_overlap_rejection():
    # Verify that predictions overlapping GT below threshold are rejected as false positives.
    gt_mask = np.zeros((100, 100), dtype=np.int32)
    gt_mask[20:40, 20:40] = 1 # 400 px

    pred_mask = np.zeros((100, 100), dtype=np.int32)
    # 20% containment overlap
    pred_mask[36:44, 20:40] = 1 # 160 px total, rows 36-39 overlap GT (80 px -> 50% containment)
    # Make it 20% containment: rows 36-39 is 4x20=80 px, rows 40-59 is 20x20=400 px -> total 480 px, overlap 80 px (16.7%)
    pred_mask_sub = np.zeros((100, 100), dtype=np.int32)
    pred_mask_sub[36:60, 20:40] = 1 # 480 px, overlap 80 px (16.7% containment)

    g1 = AnnotatedGrain(
        grain_id="G1", image_id=1, image_name="test.jpg", category="whole",
        condition="Whole", points=[(20, 20), (40, 20), (40, 40), (20, 40)],
        bbox_xyxy=(20, 20, 40, 40), polygon_area_px=400.0
    )
    rec = DatasetImageRecord(
        image_id=1, filename="test.jpg", category="whole",
        width=100, height=100, total_grains=1, counts_by_condition={"Whole": 1},
        image_path="", annotated_image_path="", annotation_xml_path="", grains=[g1]
    )

    pred_df = pd.DataFrame([{"grain_id": 1, "quality_class": "WHOLE_RICE"}])

    metrics = evaluate_image_instances(
        pred_instance_mask=pred_mask_sub,
        pred_classified_df=pred_df,
        gt_record=rec,
        matching_mode="containment",
        containment_threshold=0.50
    )

    assert metrics.true_positives == 0
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


def test_iou_vs_containment_semantics_on_bounding_boxes():
    # Coarse GT bounding box vs tight grain mask:
    # A tight ellipse inside a bounding box has high containment (~1.0) but low IoU (~0.25).
    gt_mask = np.zeros((100, 100), dtype=np.int32)
    gt_mask[10:90, 20:80] = 1 # 80 x 60 = 4800 px box

    pred_mask = np.zeros((100, 100), dtype=np.int32)
    # Tight elliptical mask centered in box (approx area 1200 px)
    pred_mask[30:70, 40:60] = 1 # 40 x 20 = 800 px

    iou_mat, cont_mat = compute_pairwise_overlap_matrices(
        pred_instance_mask=pred_mask,
        gt_instance_mask=gt_mask,
        pred_ids=[1],
        gt_ids=[1]
    )

    # IoU is intersection (800) / union (4800) = 0.1667
    assert pytest.approx(iou_mat[0, 0], 0.01) == 800.0 / 4800.0
    # Containment is intersection (800) / pred_area (800) = 1.00
    assert pytest.approx(cont_mat[0, 0], 0.01) == 1.0


def test_dataset_split_no_duplicate_leakage(tmp_path):
    # Verify DatasetCatalog group splitting keeps identical duplicate images
    # in the same partition.
    r1 = DatasetImageRecord(1, "whole 4.jpg", "whole", 100, 100, 1, {"Whole": 1}, "", "", "")
    r2 = DatasetImageRecord(2, "whole 5.jpg", "whole", 100, 100, 1, {"Whole": 1}, "", "", "")
    r3 = DatasetImageRecord(3, "whole 1.jpg", "whole", 100, 100, 1, {"Whole": 1}, "", "", "")
    r4 = DatasetImageRecord(4, "whole 2.jpg", "whole", 100, 100, 1, {"Whole": 1}, "", "", "")

    cat = DatasetCatalog(images=[r1, r2, r3, r4], dataset_dir=str(tmp_path))
    train_set, val_set, test_set = cat.split(train_ratio=0.5, val_ratio=0.5, test_ratio=0.0, seed=42)

    train_names = {img.filename for img in train_set}
    val_names = {img.filename for img in val_set}

    # whole 4 and whole 5 must NEVER be split between train and val
    assert not (("whole 4.jpg" in train_names and "whole 5.jpg" in val_names) or
                ("whole 5.jpg" in train_names and "whole 4.jpg" in val_names))
