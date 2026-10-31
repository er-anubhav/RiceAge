"""
Stage 2: High-Resolution ResNet-50 Grain Defect Classifier (224x224)
Trained with Mixed Precision (FP16) and Focal Loss to maximize texture distinction
across subtle defect classes (chalky, damaged, discoloured, dehusked).
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Multi-Class Focal Loss for subtle defects and class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def main():
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    crops_dir = Path("dataset/crops_224")
    train_dir = crops_dir / "train"
    val_dir = crops_dir / "val"
    test_dir = crops_dir / "test"

    output_dir = Path("runs/stage2_resnet50")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Augmentations designed for circular/random grain orientations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading 224x224 image datasets...", flush=True)
    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=val_transform)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}", flush=True)

    # Class balance weights
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets, minlength=num_classes)
    total_samples = len(targets)

    # Balanced alpha weights with smoothing power 0.5
    smoothed_counts = np.power(np.maximum(class_counts, 1), 0.5)
    weights = total_samples / (num_classes * smoothed_counts)
    weights = weights / np.mean(weights)
    alpha_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    print("\nClass Counts & Focal Loss Alpha Weights:", flush=True)
    for name, cnt, w in zip(class_names, class_counts, weights):
        print(f"  {name:15s}: count={cnt:5d} | alpha={w:.2f}", flush=True)

    # DataLoader with batch_size 64 and pinned memory
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize Pretrained ResNet-50
    print("\nLoading pretrained ResNet-50 backbone...", flush=True)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    # Add dropout layer before final classification for regularization
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)

    criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    best_val_acc = 0.0
    best_weights_path = output_dir / "best_resnet50.pth"

    print(f"\nStarting training for {epochs} epochs with Mixed Precision FP16...\n", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

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

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)
    print(f"Weights saved to: {best_weights_path}", flush=True)

    # Detailed evaluation on Test Set
    print("\nRunning final evaluation on held-out TEST set...", flush=True)
    checkpoint = torch.load(str(best_weights_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_test_preds = []
    all_test_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_test_preds.extend(preds.cpu().numpy())
            all_test_targets.extend(labels.cpu().numpy())

    all_test_preds = np.array(all_test_preds)
    all_test_targets = np.array(all_test_targets)

    print("\n=== RESNET-50 PER-CLASS ACCURACY ON TEST SET ===", flush=True)
    for c_idx, c_name in enumerate(class_names):
        mask = (all_test_targets == c_idx)
        total_c = np.sum(mask)
        if total_c > 0:
            acc_c = (np.sum(all_test_preds[mask] == c_idx) / total_c) * 100.0
            print(f"  {c_name:15s}: {acc_c:6.2f}% ({total_c} samples)", flush=True)

    # Export to ONNX (MIT license safe)
    onnx_path = output_dir / "grain_resnet50.onnx"
    print(f"\nExporting to ONNX: {onnx_path}...", flush=True)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    # Strip dropout for inference export
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14,
    )
    print(f"Exported Production ResNet-50 ONNX Model to: {onnx_path}", flush=True)

if __name__ == "__main__":
    main()
