"""
Rice Crop Analyzer - Model Training Pipeline
Trains YOLO11 Instance Segmentation on the normalized 7-class rice dataset.
"""

import sys
import os
from pathlib import Path

def main():
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            print("WARNING: No CUDA GPU detected. Training will run on CPU.")
    except ImportError:
        print("PyTorch is not installed. Please install it first.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is not installed. Please install it first.")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent
    data_yaml = project_root / "dataset" / "segment" / "data.yaml"
    runs_dir = project_root / "runs" / "train"

    if not data_yaml.exists():
        print(f"Error: Dataset configuration file not found at {data_yaml}")
        sys.exit(1)

    print(f"\nDataset Config: {data_yaml}")
    print(f"Output Directory: {runs_dir}\n")

    # Initialize model: YOLO11s-seg (Small segmentation baseline: fast, accurate, fits in 12GB VRAM)
    model = YOLO("yolo11s-seg.pt")

    # Hyperparameters tuned for scanner images:
    # - imgsz=1280: High resolution to preserve grain contours and defect spots
    # - batch=4: Stable for 12GB RTX 3060 at 1280px
    # - workers=4: Fast Windows DataLoader
    # - epochs=100 with patience=20 (early stopping if no improvement)
    # - save_period=10: Checkpoint every 10 epochs
    results = model.train(
        data=str(data_yaml),
        epochs=100,
        patience=20,
        imgsz=1280,
        batch=4,
        workers=2,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(project_root / "runs"),
        name="rice_seg_yolo11s_1280",
        save=True,
        save_period=10,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        plots=True,
    )

    print("\nTraining completed successfully!")
    print(f"Best model weights saved to: {model.trainer.best if hasattr(model, 'trainer') else 'runs/rice_seg_yolo11s_1280/weights/best.pt'}")

if __name__ == "__main__":
    main()
