#!/usr/bin/env python3
"""CLI script to run the baseline rice grain analysis pipeline."""

import argparse
import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.default_config import PipelineConfig
from src.pipeline import RiceQualityPipeline


def main():
    parser = argparse.ArgumentParser(description="Rice Grain Quality Analyzer - Baseline Pipeline")
    parser.add_argument(
        "--image",
        type=str,
        default="data/samples/sample_1489x2048.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--raw-reference",
        type=str,
        default="data/raw/laserjet_1200dpi_sample.jpg",
        help="Optional path to unscaled raw scan for verified physical calibration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=1000,
        help="Maximum individual crops and masks to save"
    )
    args = parser.parse_args()

    cfg = PipelineConfig()
    cfg.export.output_dir = args.output_dir
    cfg.export.visualizations_dir = os.path.join(args.output_dir, "visualizations")
    cfg.export.crops_dir = os.path.join(args.output_dir, "crops")
    cfg.export.masks_dir = os.path.join(args.output_dir, "masks")
    cfg.export.reports_dir = os.path.join(args.output_dir, "reports")
    cfg.export.max_crops_to_save = args.max_crops

    pipeline = RiceQualityPipeline(config=cfg)
    print(f"Running Rice Grain Quality Pipeline on: {args.image}")
    results = pipeline.run(
        image_path=args.image,
        raw_reference_path=args.raw_reference if os.path.exists(args.raw_reference) else None,
        generate_outputs=True
    )

    summary = results["summary"]
    print("\n--- Pipeline Completed Successfully! ---")
    print(f"Total Detected Grains (estimate): {summary['total_grains_detected']}")
    print(f"Calibration: {summary['calibration']['source_info']}")
    print(f"Separation: {summary['separation']['clusters_successfully_split']} of {summary['separation']['clusters_detected']} clusters separated.")
    print(f"Visualizations saved to: {cfg.export.visualizations_dir}")
    print(f"CSV saved to: {os.path.join(cfg.export.output_dir, 'grain_measurements.csv')}")
    print(f"Summary saved to: {os.path.join(cfg.export.output_dir, 'summary.json')}")
    print(f"Report saved to: {os.path.join(cfg.export.reports_dir, 'M0_BASELINE_REPORT.md')}")


if __name__ == "__main__":
    main()
