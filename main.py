#!/usr/bin/env python3
# Entrypoint for running the Rice Grain Quality Analyzer pipeline.

import os
import sys
import shutil

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
    print(f"Pipeline Complete | Grains: {s['total_grains_detected']} | Split: {s['separation']['clusters_successfully_split']}/{s['separation']['clusters_detected']}")


if __name__ == "__main__":
    main()
