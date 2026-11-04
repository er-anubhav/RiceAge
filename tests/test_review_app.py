from src.review_app import create_app


def test_review_api_serves_scans_and_records_audited_corrections(tmp_path):
    db_path = tmp_path / "review.db"
    app = create_app(db_path)
    app.config.update(TESTING=True)
    client = app.test_client()
    store = app.config["FEEDBACK_STORE"]

    store.cache_scan_grains("scan-1", [{
        "id": "grain_0001",
        "predicted_label": "WHOLE_RICE",
        "bbox": [10, 20, 40, 12],
        "probabilities": {},
        "measurements": {"aspect_ratio": 3.2, "review_flags": []},
        "features": {"aspect_ratio": 3.2, "secret_training_feature": 99},
    }])

    page = client.get("/review")
    assert page.status_code == 200
    assert b"Grain quality audit" in page.data

    scans = client.get("/api/scans")
    assert scans.status_code == 200
    assert scans.get_json()["scans"][0]["scan_id"] == "scan-1"

    grains_response = client.get("/api/scan_grains/scan-1")
    assert grains_response.status_code == 200
    grain = grains_response.get_json()["grains"][0]
    assert grain["grain_id"] == "grain_0001"
    assert grain["bbox"] == [10, 20, 40, 12]
    assert "features" not in grain
    assert "secret_training_feature" not in grains_response.get_data(as_text=True)

    corrected = client.post("/api/feedback/correct", json={
        "scan_id": "scan-1",
        "grain_id": "grain_0001",
        "corrected_label": "BROKEN_RICE",
    })
    assert corrected.status_code == 200
    assert corrected.get_json()["threshold_update"] is None
    assert store.get_pending_count() == 1

    duplicate = client.post("/api/feedback/correct", json={
        "scan_id": "scan-1",
        "grain_id": "grain_0001",
        "corrected_label": "BROKEN_RICE",
    })
    assert duplicate.status_code == 200
    assert store.get_pending_count() == 1

    unsupported = client.post("/api/feedback/correct", json={
        "scan_id": "scan-1",
        "grain_id": "grain_0001",
        "corrected_label": "DISC",
    })
    assert unsupported.status_code == 400

    unavailable = client.post("/api/feedback/retrain")
    assert unavailable.status_code == 501
    assert store.get_total_corrections() == 1


def test_review_api_rejects_unknown_scan_and_malformed_payload(tmp_path):
    app = create_app(tmp_path / "review.db")
    app.config.update(TESTING=True)
    client = app.test_client()

    unknown = client.get("/api/scan_grains/not-here")
    assert unknown.status_code == 404

    missing = client.post("/api/feedback/correct", json={"scan_id": "s"})
    assert missing.status_code == 400

    non_object = client.post("/api/feedback/correct", json=["not", "an", "object"])
    assert non_object.status_code == 400
