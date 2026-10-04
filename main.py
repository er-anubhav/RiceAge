#!/usr/bin/env python3
# Entrypoint for running the Rice Grain Quality Analyzer pipeline.

import argparse
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.default_config import PipelineConfig
from src.pipeline import RiceQualityPipeline


def main():
    parser = argparse.ArgumentParser(description="Rice Grain Quality Analyzer - Main Pipeline")
    parser.add_argument("--image", type=str, default="data/samples/sample_1489x2048.jpg", help="Path to input rice grain image")
    parser.add_argument("--raw-reference", type=str, default="data/raw/laserjet_1200dpi_sample.jpg", help="Path to reference raw scan for DPI derivation")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory where artifacts are exported")
    parser.add_argument("--max-crops", type=int, default=1000, help="Maximum individual grain crops to save")
    parser.add_argument("--clean-output", action="store_true", help="Clean output directory before writing new run results")
    parser.add_argument("--no-save", action="store_true", help="Run analysis in memory and print results without saving disk artifacts")
    args = parser.parse_args()

    import shutil
    if args.clean_output and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

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
        generate_outputs=not args.no_save
    )
    s = results["summary"]
    print(f"Pipeline Complete | Grains: {s['total_grains_detected']} | Split: {s['separation']['clusters_successfully_split']}/{s['separation']['clusters_detected']}")


if __name__ == "__main__":
    main()
