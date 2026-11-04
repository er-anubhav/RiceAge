# CLI flags and runtime configuration options for Rice Grain Quality Analyzer.

import argparse
from configs.default_config import PipelineConfig


def parse_args() -> argparse.Namespace:
    # Build and parse command line flags for pipeline execution.
    parser = argparse.ArgumentParser(
        description="Rice Grain Quality Analyzer - Flag & Runtime Configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # I/O and File Paths
    parser.add_argument("--image", type=str, default="RiceAge/data/samples/sample_1489x2048.jpg", help="Path to input rice grain image")
    parser.add_argument("--raw-reference", type=str, default="RiceAge/data/raw/laserjet_1200dpi_sample.jpg", help="Path to reference raw scan for DPI derivation")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory where artifacts are exported")

    # Output and Execution Controls
    parser.add_argument("--max-crops", type=int, default=1000, help="Maximum individual grain crops to save")
    parser.add_argument("--clean-output", action="store_true", help="Clean output directory before writing new run results")
    parser.add_argument("--no-save", action="store_true", help="Run analysis in memory and print results without saving disk artifacts")

    # Algorithmic & AI Engine Flags
    parser.add_argument("--seg-engine", type=str, default="yolo", choices=["yolo", "classic"], help="Segmentation engine (yolo deep learning vs classic watershed)")
    parser.add_argument("--seg-method", type=str, default="otsu", choices=["otsu", "adaptive", "triangle"], help="Segmentation thresholding method for classic engine")
    parser.add_argument("--use-ai", action="store_true", default=True, help="Enable Stage 2 AI defect classifier")
    parser.add_argument("--no-ai", dest="use_ai", action="store_false", help="Disable Stage 2 AI defect classifier")
    parser.add_argument("--runtime", type=str, default="auto", choices=["auto", "onnx", "pytorch"], help="Inference runtime backend (auto-detect, onnx runtime, or pytorch)")
    parser.add_argument("--onnx", action="store_true", help="Shortcut flag: run both Stage 1 and Stage 2 models using ONNX Runtime")
    parser.add_argument("--yolo-weights", type=str, default="runs/rice_seg_yolo11s_1280/weights/best.pt", help="Path to Stage 1 YOLO segmenter weights (.pt or .onnx)")
    parser.add_argument("--classifier-weights", type=str, default="runs/stage2_resnet50/best_resnet50.pth", help="Path to Stage 2 ResNet-50 defect classifier weights (.pth or .onnx)")

    # Calibration Flags
    parser.add_argument("--assumed-dpi", type=float, default=None, help="Explicit DPI value to assume if metadata is unverified")
    parser.add_argument("--cluster-area-thresh", type=int, default=650, help="Area threshold (px) to scrutinize as cluster")
    parser.add_argument("--solidity-thresh", type=float, default=0.92, help="Solidity threshold below which component is considered touching")
    parser.add_argument("--broken-ratio-thresh", type=float, default=0.75, help="Length fraction relative to median whole grain to consider broken")

    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    # Map command line arguments into structured PipelineConfig.
    cfg = PipelineConfig()

    # Export paths
    cfg.export.output_dir = args.output_dir
    cfg.export.visualizations_dir = f"{args.output_dir}/visualizations"
    cfg.export.crops_dir = f"{args.output_dir}/crops"
    cfg.export.masks_dir = f"{args.output_dir}/masks"
    cfg.export.reports_dir = f"{args.output_dir}/reports"
    cfg.export.max_crops_to_save = args.max_crops

    # Calibration & Preprocessing
    if args.assumed_dpi is not None:
        cfg.calibration.assumed_dpi = args.assumed_dpi

    # Segmentation & AI Models
    cfg.segmentation.engine = args.seg_engine
    cfg.segmentation.method = args.seg_method
    cfg.separation.cluster_area_threshold_px = args.cluster_area_thresh
    cfg.separation.solidity_threshold = args.solidity_thresh

    # Classification & Defect AI
    cfg.classification.use_ai_classifier = args.use_ai
    cfg.classification.broken_length_ratio_thresh = args.broken_ratio_thresh

    # Runtime backend (ONNX vs PyTorch)
    runtime = "onnx" if args.onnx else args.runtime
    cfg.segmentation.runtime = runtime
    cfg.classification.runtime = runtime

    if runtime == "onnx" and args.yolo_weights == "runs/rice_seg_yolo11s_1280/weights/best.pt":
        cfg.segmentation.yolo_weights_path = cfg.segmentation.yolo_onnx_path
    else:
        cfg.segmentation.yolo_weights_path = args.yolo_weights

    if runtime == "onnx" and args.classifier_weights == "runs/stage2_resnet50/best_resnet50.pth":
        cfg.classification.classifier_weights_path = cfg.classification.classifier_onnx_path
    else:
        cfg.classification.classifier_weights_path = args.classifier_weights

    return cfg
