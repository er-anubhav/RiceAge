#!/usr/bin/env python3
# CLI entrypoint for evaluating RiceAge pipeline against annotated datasets.

import os
import sys
import argparse
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.default_config import PipelineConfig
from src.pipeline import RiceQualityPipeline
from src.dataset import load_dataset
from src.evaluation import evaluate_catalog, DatasetEvaluationReport


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RiceAge Dataset Evaluation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=str, default="data/dataset", help="Path to ground truth dataset")
    parser.add_argument("--category", type=str, default="all", choices=["all", "whole", "broken", "chalky", "damaged", "foreign_matter", "mixed"], help="Filter by category")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to evaluate")
    parser.add_argument("--matching-mode", type=str, default="containment", choices=["containment", "iou"], help="Instance matching mode: containment (point/mask containment in coarse box) or strict iou")
    parser.add_argument("--iou-thresh", type=float, default=0.50, help="Intersection over Union threshold for instance matching")
    parser.add_argument("--containment-thresh", type=float, default=0.50, help="Containment threshold for instance matching")
    parser.add_argument("--disable-hybrid", action="store_true", help="Disable concave neck fallback (run pure watershed only)")
    parser.add_argument("--ref-whole-length-mm", type=float, default=None, help="Optional physical reference cultivar length in mm")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation", help="Directory where evaluation artifacts are saved")
    parser.add_argument("--save-vis", action="store_true", help="Save diagnostic evaluation visual overlays for each image")
    return parser.parse_args()


def run_evaluation():
    args = parse_eval_args()
    print("=" * 80)
    print("RICEAGE DATASET EVALUATION BENCHMARK")
    print(f"Dataset: {args.dataset_dir} | Category: {args.category} | Mode: {args.matching_mode}")
    print(f"Thresholds: IoU={args.iou_thresh}, Containment={args.containment_thresh}")
    print(f"Separation Mode: {'Pure Watershed' if args.disable_hybrid else 'Hybrid (Watershed + Concave Neck)'}")
    print("=" * 80)

    # 1. Load catalog
    catalog = load_dataset(dataset_dir=args.dataset_dir, repo_root=PROJECT_ROOT)
    if args.category != "all":
        filtered_imgs = catalog.filter_by_category(args.category)
        from src.dataset import DatasetCatalog
        catalog = DatasetCatalog(images=filtered_imgs, dataset_dir=catalog.dataset_dir)

    if args.max_images is not None and args.max_images > 0:
        catalog.images = catalog.images[:args.max_images]

    print(f"Evaluating {len(catalog)} images containing {catalog.total_grains} ground-truth instances...")

    # 2. Configure pipeline
    cfg = PipelineConfig()
    cfg.separation.enable_concave_fallback = not args.disable_hybrid
    if args.ref_whole_length_mm is not None:
        cfg.classification.reference_whole_length_mm = args.ref_whole_length_mm

    pipeline = RiceQualityPipeline(config=cfg)

    # 3. Execute evaluation
    vis_dir = os.path.join(args.output_dir, "visualizations") if args.save_vis else None
    report: DatasetEvaluationReport = evaluate_catalog(
        pipeline=pipeline,
        catalog=catalog,
        iou_threshold=args.iou_thresh,
        matching_mode=args.matching_mode,
        containment_threshold=args.containment_thresh,
        save_visualizations=args.save_vis,
        vis_output_dir=vis_dir
    )

    # 4. Print results
    print("\n" + "=" * 80)
    print("INSTANCE EXTRACTION & COUNTING SUMMARY")
    print("=" * 80)
    print(f"Total Images Evaluated : {report.total_images_evaluated}")
    print(f"Ground-Truth Grains    : {report.overall_gt_count}")
    print(f"Predicted Grains       : {report.overall_pred_count}")
    print(f"Mean Abs Count Error   : {report.mean_abs_count_error:.2f} grains / image")
    print(f"Mean Rel Count Error   : {report.mean_rel_count_error * 100:.2f}%")
    print(f"Instance Precision     : {report.mean_precision * 100:.2f}%")
    print(f"Instance Recall        : {report.mean_recall * 100:.2f}%")
    print(f"Instance F1 Score      : {report.mean_f1 * 100:.2f}%")
    print(f"Mean Matched IoU       : {report.mean_iou:.4f}")

    print("\n" + "-" * 80)
    print("PER-CATEGORY BREAKDOWN")
    print("-" * 80)
    cat_df = pd.DataFrame.from_dict(report.category_breakdown, orient="index")
    print(cat_df.to_string())

    print("\n" + "-" * 80)
    print("CLASSIFICATION EVALUATION (SUPPORTED VS UNSUPPORTED)")
    print("-" * 80)
    cb = report.classification_breakdown
    print(f"Supported Classes Evaluated  : {cb.get('supported_classes_evaluated')}")
    print(f"Unsupported Defect Classes   : {cb.get('unsupported_classes_present')}")
    print(f"Supported Instances Matched  : {cb.get('supported_evaluated_instances')}")
    print(f"Supported Correct Predictions: {cb.get('supported_correct_predictions')}")
    print(f"Supported Class Accuracy     : {cb.get('supported_classification_accuracy', 0) * 100:.2f}%")

    print("\nCondition Breakdown Matrix (Ground-Truth vs Pipeline Prediction):")
    cond_matrix = pd.DataFrame.from_dict(cb.get("condition_prediction_matrix", {}), orient="index")
    print(cond_matrix.to_string())

    # 5. Export structured reports
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Per-image CSV
    per_img_rows = [m.to_dict() for m in report.per_image_metrics]
    df_per_img = pd.DataFrame(per_img_rows)
    per_img_csv_path = os.path.join(args.output_dir, "per_image_metrics.csv")
    df_per_img.to_csv(per_img_csv_path, index=False)

    # Matched instances CSV
    all_matches = []
    for m in report.per_image_metrics:
        for match in m.matches:
            all_matches.append({
                "image_id": m.image_id,
                "filename": m.filename,
                "category": m.category,
                "pred_id": match.pred_id,
                "gt_id": match.gt_id,
                "gt_condition": match.gt_condition,
                "pred_quality_class": match.pred_quality_class,
                "iou": match.iou,
                "is_correct_class": match.is_correct_class,
                "is_class_supported": match.is_class_supported,
            })
    df_matches = pd.DataFrame(all_matches)
    matches_csv_path = os.path.join(args.output_dir, "matched_instances.csv")
    df_matches.to_csv(matches_csv_path, index=False)

    # Summary JSON
    import json
    summary_json_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_summary_dict(), f, indent=2)

    print(f"\nSaved evaluation artifacts to: {args.output_dir}")
    print(f"  - {per_img_csv_path}")
    print(f"  - {matches_csv_path}")
    print(f"  - {summary_json_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
