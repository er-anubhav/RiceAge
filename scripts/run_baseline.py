#!/usr/bin/env python3
import argparse
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.default_config import PipelineConfig
from src.pipeline import RiceQualityPipeline


def main():
    parser = argparse.ArgumentParser(description="Rice Grain Quality Analyzer - Baseline Pipeline")
    parser.add_argument("--image", type=str, default="data/samples/sample_1489x2048.jpg")
    parser.add_argument("--raw-reference", type=str, default="data/raw/laserjet_1200dpi_sample.jpg")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--max-crops", type=int, default=1000)
    args = parser.parse_args()

    cfg = PipelineConfig()
    cfg.export.output_dir = args.output_dir
    cfg.export.visualizations_dir = os.path.join(args.output_dir, "visualizations")
    cfg.export.crops_dir = os.path.join(args.output_dir, "crops")
    cfg.export.masks_dir = os.path.join(args.output_dir, "masks")
    cfg.export.reports_dir = os.path.join(args.output_dir, "reports")
    cfg.export.max_crops_to_save = args.max_crops

    pipeline = RiceQualityPipeline(config=cfg)
    results = pipeline.run(
        image_path=args.image,
        raw_reference_path=args.raw_reference if os.path.exists(args.raw_reference) else None,
        generate_outputs=True
    )
    s = results["summary"]
    print(f"Done | Grains: {s['total_grains_detected']} | Split: {s['separation']['clusters_successfully_split']}/{s['separation']['clusters_detected']}")


if __name__ == "__main__":
    main()
