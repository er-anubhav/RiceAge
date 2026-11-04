import pytest
import numpy as np
import pandas as pd

from src.evaluation import (
    compute_pairwise_overlap_matrices,
    evaluate_image_instances,
    DatasetImageRecord,
    AnnotatedGrain,
)


def test_compute_pairwise_overlap_matrices():
    pred_mask = np.zeros((100, 100), dtype=np.int32)
    pred_mask[20:40, 20:40] = 1 # 400 px

    gt_mask = np.zeros((100, 100), dtype=np.int32)
    gt_mask[20:40, 20:40] = 1 # Exactly overlapping: 400 px
    gt_mask[60:80, 60:80] = 2

    iou_mat, cont_mat = compute_pairwise_overlap_matrices(
        pred_instance_mask=pred_mask,
        gt_instance_mask=gt_mask,
        pred_ids=[1],
        gt_ids=[1, 2]
    )

    assert pytest.approx(iou_mat[0, 0], 0.01) == 1.0
    assert pytest.approx(cont_mat[0, 0], 0.01) == 1.0
    assert iou_mat[0, 1] == 0.0
    assert cont_mat[0, 1] == 0.0


def test_evaluate_image_instances():
    # 2 GT grains
    g1 = AnnotatedGrain(
        grain_id="G1", image_id=1, image_name="test.jpg", category="whole",
        condition="Whole", points=[(10, 10), (30, 10), (30, 40), (10, 40)],
        bbox_xyxy=(10, 10, 30, 40), polygon_area_px=600.0
    )
    g2 = AnnotatedGrain(
        grain_id="G2", image_id=1, image_name="test.jpg", category="whole",
        condition="Whole", points=[(50, 50), (70, 50), (70, 80), (50, 80)],
        bbox_xyxy=(50, 50, 70, 80), polygon_area_px=600.0
    )

    rec = DatasetImageRecord(
        image_id=1, filename="test.jpg", category="whole",
        width=100, height=100, total_grains=2,
        counts_by_condition={"Whole": 2},
        image_path="", annotated_image_path="", annotation_xml_path="",
        grains=[g1, g2]
    )

    # Pred mask with 2 perfectly matching grains
    pred_mask = np.zeros((100, 100), dtype=np.int32)
    pred_mask[10:40, 10:30] = 1
    pred_mask[50:80, 50:70] = 2

    pred_df = pd.DataFrame([
        {"grain_id": 1, "quality_class": "WHOLE_RICE"},
        {"grain_id": 2, "quality_class": "WHOLE_RICE"},
    ])

    metrics = evaluate_image_instances(
        pred_instance_mask=pred_mask,
        pred_classified_df=pred_df,
        gt_record=rec,
        matching_mode="containment",
        containment_threshold=0.50
    )

    assert metrics.gt_count == 2
    assert metrics.pred_count == 2
    assert metrics.abs_count_error == 0
    assert metrics.true_positives == 2
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert len(metrics.matches) == 2
    assert all(m.is_correct_class for m in metrics.matches)
