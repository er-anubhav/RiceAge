import json

import pytest

from src.feedback_store import FeedbackStore
from src.threshold_calibration import ThresholdBandit


def test_feedback_store_uses_server_cached_features_and_upserts(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.db")
    store.cache_scan_grains("scan-1", [{
        "id": "grain-1",
        "predicted_label": "DISCOLORED_RICE",
        "bbox": [1, 2, 30, 12],
        "probabilities": {"discoloured": 0.71},
        "features": {"length_px": 30.0, "solidity": 0.94},
    }])

    first = store.record_correction(
        "scan-1", "grain-1", "WHOLE_RICE",
        error_type="FALSE_POSITIVE", affected_head="discoloured",
        threshold_before=0.5, threshold_after=0.515,
    )
    second = store.record_correction(
        "scan-1", "grain-1", "BROKEN_RICE",
        error_type="LABEL_CORRECTION", affected_head="discoloured",
    )

    assert first["id"] == second["id"]
    assert store.get_pending_count() == 1
    record = store.get_pending_records()[0]
    assert record["corrected_label"] == "BROKEN_RICE"
    assert record["features"] == {"length_px": 30.0, "solidity": 0.94}
    assert store.get_scan_grains("scan-1")[0]["corrected_label"] == "BROKEN_RICE"


def test_feedback_store_rejects_unknown_grain_and_invalid_error(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.db")
    with pytest.raises(LookupError):
        store.record_correction(
            "missing-scan", "grain-1", "WHOLE_RICE",
            error_type="FALSE_POSITIVE",
        )
    store.cache_scan_grains("scan-1", [{"id": "grain-1", "predicted_label": "WHOLE_RICE"}])
    with pytest.raises(ValueError):
        store.record_correction(
            "scan-1", "grain-1", "BROKEN_RICE",
            error_type="GUESS",
        )


def test_threshold_direction_and_atomic_persistence(tmp_path):
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"thresholds": {"discoloured": 0.5}}))
    bandit = ThresholdBandit(weights)

    fp = bandit.record_feedback("discoloured", "FALSE_POSITIVE")
    assert fp["threshold_after"] == pytest.approx(0.515)
    fn = bandit.record_feedback("discoloured", "FALSE_NEGATIVE")
    assert fn["threshold_after"] == pytest.approx(0.5)
    assert bandit.get_threshold("discoloured") == pytest.approx(0.5)
    assert json.loads(weights.read_text())["thresholds"]["discoloured"] == pytest.approx(0.5)


def test_threshold_clamps_to_safety_bounds(tmp_path):
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"thresholds": {"damaged": 0.895, "chalky": 0.105}}))
    bandit = ThresholdBandit(weights)

    assert bandit.record_feedback("damaged", "FALSE_POSITIVE")["threshold_after"] == 0.9
    assert bandit.record_feedback("chalky", "FALSE_NEGATIVE")["threshold_after"] == 0.1


def test_threshold_supports_heads_weight_schema(tmp_path):
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"heads": {"discoloured": {"threshold": 0.4, "feature_cols": ["x"]}}}))
    bandit = ThresholdBandit(weights)
    result = bandit.record_feedback("discoloured", "FALSE_NEGATIVE")

    data = json.loads(weights.read_text())
    assert result["threshold_after"] == pytest.approx(0.385)
    assert data["heads"]["discoloured"]["threshold"] == pytest.approx(0.385)
    assert data["heads"]["discoloured"]["feature_cols"] == ["x"]
