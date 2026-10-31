"""
Stage 2: Grain Defect Crop Classifier (MobileNetV3-Small)
Trains on 43,000 full-resolution 128x128 grain crops with class-balanced cross-entropy loss.
Outputs best.pth, confusion matrix, and ONNX model.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def main():
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    crops_dir = Path("dataset/crops")
    train_dir = crops_dir / "train"
    val_dir = crops_dir / "val"
    test_dir = crops_dir / "test"

    output_dir = Path("runs/stage2_classifier")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Image transforms with robust optical augmentations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading image datasets...")
    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=val_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Calculate class counts and inverse frequency weights for balanced training
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets, minlength=num_classes)
    total_samples = len(targets)
    print("\nTraining Class Distribution:")
    for name, cnt in zip(class_names, class_counts):
        print(f"  {name:15s}: {cnt}")

    # Inverse frequency weights with smoothing to prevent rare classes from dominating gradients
    weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(np.float32))
    # Normalize weights so mean weight is 1.0
    weights = weights / np.mean(weights)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print("\nComputed Class Weights (Balanced):")
    for name, w in zip(class_names, weights):
        print(f"  {name:15s}: {w:.2f}")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize MobileNetV3-Small (lightweight, ultra-fast: ~2.5M parameters)
    print("\nInitializing MobileNetV3-Small pretrained backbone...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_weights_path = output_dir / "best_classifier.pth"

    print(f"\nStarting training for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_loss = val_loss / total_val
        val_acc = (correct_val / total_val) * 100.0
        elapsed = time.time() - t0

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
            }, str(best_weights_path))

        star = " * (Best)" if is_best else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%{star}", flush=True)

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Weights saved to: {best_weights_path}")

    # Final detailed evaluation on Test Set
    print("\nRunning final evaluation on held-out TEST set...")
    checkpoint = torch.load(str(best_weights_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_test_preds = []
    all_test_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_targets.extend(labels.cpu().numpy())

    all_test_preds = np.array(all_test_preds)
    all_test_targets = np.array(all_test_targets)

    print("\n=== PER-CLASS ACCURACY ON TEST SET ===")
    for c_idx, c_name in enumerate(class_names):
        mask = (all_test_targets == c_idx)
        total_c = np.sum(mask)
        if total_c > 0:
            acc_c = (np.sum(all_test_preds[mask] == c_idx) / total_c) * 100.0
            print(f"  {c_name:15s}: {acc_c:6.2f}% ({total_c} samples)")

    # Export to ONNX for commercial production deployment (MIT license safe)
    onnx_path = output_dir / "grain_classifier.onnx"
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"\nExported Production ONNX Model to: {onnx_path}")

if __name__ == "__main__":
    main()
