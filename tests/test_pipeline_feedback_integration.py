from pathlib import Path

from configs.default_config import ExportConfig, PipelineConfig
from src.feedback_store import FeedbackStore
from src.pipeline import RiceQualityPipeline


def test_m0_pipeline_caches_grains_without_changing_existing_outputs(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    image_path = repo_root / "data" / "samples" / "sample_1489x2048.jpg"
    assert image_path.exists(), "The repository sample image is required for this integration test"

    db_path = tmp_path / "feedback.db"
    config = PipelineConfig(
        export=ExportConfig(
            output_dir=str(tmp_path),
            feedback_db_path=str(db_path),
            persist_feedback=True,
        )
    )

    result = RiceQualityPipeline(config).run(str(image_path), generate_outputs=False)

    assert result["summary"]["scan_id"] == result["scan_id"]
    assert result["summary"]["feedback"]["persisted"] is True
    assert len(result["grains_data"]) == result["summary"]["total_grains_detected"]
    assert "measurements_df" in result
    assert "instance_mask" in result
    assert "clean_mask" in result

    store = FeedbackStore(db_path)
    cached = store.get_scan_grains(result["scan_id"])
    assert len(cached) == result["summary"]["total_grains_detected"]
    if cached:
        grain = cached[0]
        assert grain["grain_id"].startswith("grain_")
        assert len(grain["bbox"]) == 4
        assert grain["original_pred"] in {
            "WHOLE_RICE", "BROKEN_RICE", "BOUNDARY_TRUNCATED", "UNRESOLVED_CLUSTER"
        }
        # M0 is a rule-based classifier, not the planned probabilistic defect model.
        assert grain["probabilities"] == {}
        assert "quality_class" not in grain["features"]
