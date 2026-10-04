import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.default_config import PipelineConfig
from src.pipeline import RiceQualityPipeline


def run_experiment():
    sample_img = os.path.join(PROJECT_ROOT, "data/samples/sample_1489x2048.jpg")
    raw_img = os.path.join(PROJECT_ROOT, "data/raw/laserjet_1200dpi_sample.jpg")

    pipeline = RiceQualityPipeline(config=PipelineConfig())
    results = pipeline.run(sample_img, raw_reference_path=raw_img if os.path.exists(raw_img) else None)
    summary = results["summary"]
    sep = summary["separation"]

    print(f"EXP-001 Complete | Detected: {summary['total_grains_detected']} | "
          f"Raw CC: {sep['raw_connected_components']} | "
          f"Clusters Split: {sep['clusters_successfully_split']}/{sep['clusters_detected']} | "
          f"Unresolved: {sep['clusters_unresolved']}")

    exp_log = {
        "experiment_id": "EXP-001",
        "date": summary["timestamp"],
        "detected_count": summary["total_grains_detected"],
        "separation": sep,
        "quality_counts": summary["quality_counts"],
    }
    with open(os.path.join(PROJECT_ROOT, "experiments/exp001_log.json"), "w", encoding="utf-8") as f:
        json.dump(exp_log, f, indent=2)


if __name__ == "__main__":
    run_experiment()
