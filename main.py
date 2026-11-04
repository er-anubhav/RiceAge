#!/usr/bin/env python3
# Entrypoint for running the Rice Grain Quality Analyzer pipeline.

import os
import sys
import shutil

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flags import parse_args, build_config_from_args
from src.pipeline import RiceQualityPipeline


def main():
    args = parse_args()

    if args.clean_output and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    cfg = build_config_from_args(args)
    pipeline = RiceQualityPipeline(config=cfg)

    results = pipeline.run(
        image_path=args.image,
        raw_reference_path=args.raw_reference if os.path.exists(args.raw_reference) else None,
        generate_outputs=not args.no_save
    )

    s = results["summary"]
    q_counts = s.get("quality_counts", {})
    mode = s.get("engine_mode", {})

    print("\n=======================================================")
    print("        RICE QUALITY ANALYZER - RUN COMPLETE           ")
    print("=======================================================")
    print(f"  Image:                {s.get('image_path')}")
    print(f"  Segmentation Engine:  {mode.get('segmentation')}")
    print(f"  Classifier Engine:    {mode.get('classification')}")
    print(f"  Total Grains Found:   {s.get('total_grains_detected')}")
    print(f"  Commercial Grade:     {s.get('commercial_grade')}")
    print("-------------------------------------------------------")
    print("  Quality Breakdown:")
    for k, v in q_counts.items():
        pct = (v / max(1, s.get('total_grains_detected', 1))) * 100
        print(f"    - {k:20s}: {v:4d}  ({pct:5.1f}%)")
    print("-------------------------------------------------------")
    print(f"  Artifacts saved to:   {args.output_dir}/")
    print(f"    -> Visualizations:  {args.output_dir}/visualizations/")
    print(f"    -> Measurements:    {args.output_dir}/grain_measurements.csv")
    print(f"    -> Summary JSON:    {args.output_dir}/summary.json")
    print(f"    -> Audit Report:    {args.output_dir}/reports/")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
