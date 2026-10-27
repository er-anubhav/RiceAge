import numpy as np

from scripts.build_segmentation_candidates import (
    build_yolo_annotations,
    make_image_id,
    polygon_for_instance,
    remap_instance_ids,
)


def test_polygon_for_instance_returns_normalized_yolo_polygon():
    mask = np.zeros((80, 120), dtype=np.int32)
    mask[12:30, 15:55] = 7

    polygon = polygon_for_instance(mask, 7)

    assert polygon is not None
    assert polygon["line"].startswith("0 ")
    values = [float(value) for value in polygon["line"].split()[1:]]
    assert len(values) >= 6
    assert len(values) % 2 == 0
    assert all(0.0 <= value <= 1.0 for value in values)
    assert polygon["pixel_area"] == 18 * 40


def test_builder_excludes_boundary_and_unresolved_instances():
    mask = np.zeros((100, 140), dtype=np.int32)
    mask[10:30, 10:45] = 1
    mask[35:55, 55:90] = 2
    mask[60:80, 100:130] = 3
    measurements = [
        {
            "grain_id": 1,
            "separation_status": "isolated_single",
            "is_boundary": False,
            "quality_class": "WHOLE_RICE",
        },
        {
            "grain_id": 2,
            "separation_status": "under_segmented_cluster",
            "is_boundary": False,
            "quality_class": "UNRESOLVED_CLUSTER",
        },
        {
            "grain_id": 3,
            "separation_status": "isolated_single",
            "is_boundary": True,
            "quality_class": "BOUNDARY_TRUNCATED",
        },
    ]

    lines, audit = build_yolo_annotations(mask, measurements)

    assert len(lines) == 1
    assert lines[0].startswith("0 ")
    assert audit[0]["included_in_candidate_labels"] is True
    assert audit[1]["exclusion_reason"].startswith("untrusted_m0_separation_status")
    assert audit[2]["exclusion_reason"] == "boundary_truncated"


def test_polygon_is_rejected_for_missing_or_tiny_instance():
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[4:6, 4:6] = 9

    assert polygon_for_instance(mask, 9, minimum_mask_area_px=20) is None
    assert polygon_for_instance(mask, 99) is None


def test_instance_map_remap_is_compact_and_lossless():
    mask = np.array([[0, 17, 17], [0, 200, 0]], dtype=np.int32)

    remapped, id_map = remap_instance_ids(mask)

    assert set(id_map.values()) == {17, 200}
    assert remapped[0, 0] == 0
    assert remapped[0, 1] == remapped[0, 2]
    assert remapped[0, 1] != remapped[1, 1]
    assert id_map[str(int(remapped[0, 1]))] == 17
    assert id_map[str(int(remapped[1, 1]))] == 200


def test_image_id_is_stable_and_path_sensitive():
    first = make_image_id("batch_a/scan_01.jpg")
    second = make_image_id("batch_a/scan_01.jpg")
    third = make_image_id("batch_b/scan_01.jpg")

    assert first == second
    assert first != third
