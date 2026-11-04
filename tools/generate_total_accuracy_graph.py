"""
Total Graph Accuracy Benchmark:
Evaluates the entire 7,501 held-out test grains using the Two-Stage Calibrated Optical Fusion Engine.
Computes Precision, Recall, F1-Score, Overall Accuracy, and saves a high-resolution graph.
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "broken",
    "chalky",
    "damaged",
    "dehusked",
    "discoloured",
    "foreign_matter",
    "whole"
]

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running Total Accuracy Benchmark on {device}...")

    # Load ResNet-50
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(CLASS_NAMES))
    )
    checkpoint = torch.load("runs/stage2_resnet50/best_resnet50.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dir = Path("dataset/crops_224/test")
    all_files = []
    ground_truth = []

    for c_idx, c_name in enumerate(CLASS_NAMES):
        c_folder = test_dir / c_name
        if c_folder.exists():
            files = list(c_folder.glob("*.jpg"))
            for f in files:
                all_files.append((f, c_idx))
                ground_truth.append(c_idx)

    total_samples = len(all_files)
    print(f"Total Held-Out Test Samples: {total_samples}")

    batch_size = 64
    predictions = []

    IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

    # Process in batches
    for b_start in range(0, total_samples, batch_size):
        b_files = all_files[b_start:b_start + batch_size]
        tensors = []
        opt_features = []

        for p, _ in b_files:
            img = cv2.imread(str(p))
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(transform(rgb))

            # Optical forensic metrics
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            inner = cv2.erode(otsu, kernel) > 0
            if np.sum(inner) < 30:
                inner = otsu > 0

            core_bgr = img[inner]
            core_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[inner]

            r_vals = core_bgr[:, 2].astype(np.float32)
            g_vals = core_bgr[:, 1].astype(np.float32)
            rg_ratio = float(np.mean(r_vals / np.maximum(g_vals, 1.0)))

            v_vals = core_hsv[:, 2].astype(np.float32)
            s_vals = core_hsv[:, 1].astype(np.float32)
            core_std_v = float(np.std(v_vals))
            median_v = float(np.median(v_vals))

            pinhole_ratio = float(np.sum(v_vals < 0.60 * median_v)) / max(float(len(v_vals)), 1.0)
            chalky_ratio = float(np.sum((v_vals >= 170) & (s_vals <= 35))) / max(float(len(v_vals)), 1.0)

            # Aspect ratio check on crop contour
            cnts, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            aspect_ratio = 2.5
            if cnts:
                rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
                w_box, h_box = rect[1]
                aspect_ratio = max(w_box, h_box) / max(min(w_box, h_box), 1e-3)

            opt_features.append({
                "rg_ratio": rg_ratio,
                "core_std_v": core_std_v,
                "pinhole_ratio": pinhole_ratio,
                "chalky_ratio": chalky_ratio,
                "aspect_ratio": aspect_ratio
            })

        batch_t = torch.stack(tensors).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda") if "cuda" in str(device) else torch.no_grad():
                logits = model(batch_t)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()

        for prob_vec, opt in zip(probs, opt_features):
            rg_ratio = opt["rg_ratio"]
            core_std_v = opt["core_std_v"]
            pinhole_ratio = opt["pinhole_ratio"]
            chalky_ratio = opt["chalky_ratio"]
            aspect_ratio = opt["aspect_ratio"]

            p_broken = prob_vec[IDX["broken"]]
            p_chalky = prob_vec[IDX["chalky"]]
            p_damaged = prob_vec[IDX["damaged"]]
            p_dehusked = prob_vec[IDX["dehusked"]]
            p_discoloured = prob_vec[IDX["discoloured"]]
            p_foreign = prob_vec[IDX["foreign_matter"]]
            p_whole = prob_vec[IDX["whole"]]

            # Calibrated Decision Fusion
            if p_foreign > 0.40 or rg_ratio > 1.15:
                pred_cls = "foreign_matter"
            elif (p_broken > 0.35) or (aspect_ratio < 1.70):
                pred_cls = "broken"
            elif p_dehusked > 0.40:
                pred_cls = "dehusked"
            elif (p_damaged > p_whole) or (p_damaged > 0.28 and (rg_ratio >= 1.018 or core_std_v >= 14.0 or pinhole_ratio > 0.04)):
                pred_cls = "damaged"
            elif (p_chalky > p_whole) or (p_chalky > 0.32 and chalky_ratio > 0.20):
                pred_cls = "chalky"
            elif (p_discoloured > p_whole) or (p_discoloured > 0.30 and rg_ratio >= 1.022):
                pred_cls = "discoloured"
            else:
                pred_cls = "whole"

            predictions.append(IDX[pred_cls])

    y_true = np.array(ground_truth)
    y_pred = np.array(predictions)

    # Compute overall accuracy
    overall_acc = (np.sum(y_true == y_pred) / float(len(y_true))) * 100.0

    # 7x7 Confusion Matrix
    cm = np.zeros((7, 7), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1.0) * 100.0

    # Per-Class Precision, Recall, F1
    precisions = []
    recalls = []
    f1_scores = []

    print("\n" + "=" * 90)
    print(f"{'CLASS':<16} | {'SAMPLES':<8} | {'PRECISION (%)':<15} | {'RECALL (%)':<12} | {'F1-SCORE (%)'}")
    print("-" * 90)

    for i, c_name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        prec = (tp / max(tp + fp, 1)) * 100.0
        rec = (tp / max(tp + fn, 1)) * 100.0
        f1 = (2 * prec * rec / max(prec + rec, 1e-4))

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

        print(f"{c_name:<16} | {np.sum(y_true == i):<8} | {prec:13.2f}% | {rec:10.2f}% | {f1:11.2f}%")

    macro_f1 = np.mean(f1_scores)
    print("=" * 90)
    print(f"TOTAL SYSTEM ACCURACY: {overall_acc:.2f}% (Across {total_samples} Held-Out Test Grains)")
    print(f"MACRO F1-SCORE:       {macro_f1:.2f}%")

    # Plot Visual Accuracy Graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#ffffff')

    # Panel 1: Bar Chart of Precision, Recall, F1
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    ax1.bar(x - width, precisions, width, label='Precision', color='#1f77b4', alpha=0.9)
    ax1.bar(x, recalls, width, label='Recall', color='#2ca02c', alpha=0.9)
    ax1.bar(x + width, f1_scores, width, label='F1-Score', color='#ff7f0e', alpha=0.9)

    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Per-Class Accuracy Benchmark\nTotal Test Accuracy: {overall_acc:.1f}% | Macro F1: {macro_f1:.1f}%', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace('_', ' ').capitalize() for c in CLASS_NAMES], rotation=30, ha='right', fontsize=10)
    ax1.set_ylim([0, 105])
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right', frameon=True)

    for i in range(len(CLASS_NAMES)):
        ax1.text(x[i] + width, f1_scores[i] + 1.5, f"{f1_scores[i]:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel 2: Normalized Confusion Matrix Heatmap
    im = ax2.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title('Normalized Confusion Matrix (% per class)', fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(CLASS_NAMES))
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels([c.replace('_', ' ').capitalize() for c in CLASS_NAMES], rotation=35, ha='right', fontsize=9)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels([c.replace('_', ' ').capitalize() for c in CLASS_NAMES], fontsize=9)

    fmt = '.1f'
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax2.text(j, i, format(cm_norm[i, j], fmt) + '%',
                     ha="center", va="center",
                     fontsize=8,
                     color="white" if cm_norm[i, j] > thresh else "black")

    ax2.set_ylabel('True Ground Truth Label', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Predicted Label by Pipeline', fontsize=11, fontweight='bold')

    plt.tight_layout()
    out_graph = Path("runs/total_graph_accuracy.png")
    plt.savefig(out_graph, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nTotal Graph Accuracy visualization successfully saved to: {out_graph.resolve()}")

if __name__ == "__main__":
    main()
