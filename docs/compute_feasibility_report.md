# RiceAge AI — Compute & Deployment Feasibility Report
**Document Version:** 1.0.0  
**Phase:** 2 — Fully Automated Model Improvement  
**Effective Date:** October 31, 2026  
**Status:** Certified Engineering Audit  

---

## 1. System Hardware & Infrastructure Audit

An empirical audit of the target workstation was conducted using PyTorch 2.6.0+cu124 and ONNX Runtime 1.31.1:

| Hardware Component | Specification | Available Allocation |
| :--- | :--- | :--- |
| **GPU Accelerator** | **NVIDIA GeForce RTX 3060** (Ampere, Compute Capability 8.6) | **12.00 GB GDDR6 VRAM** (11.99 GB dedicated) |
| **Host CPU** | 8 Physical Cores / 16 Logical Threads (x86_64) | 16 execution threads available |
| **System Memory (RAM)** | 31.89 GB Total | **23.58 GB Available** |
| **CUDA Driver / Runtime** | CUDA 12.4 | Active and fully operational |
| **Inference Engines** | PyTorch 2.6 CUDA / ONNX Runtime 1.31.1 CPU & CUDA | Fully verified |

---

## 2. Architecture Benchmark: Latency, Throughput & Memory

Using synthetic grain crop batches ($128 \times 128 \times 3$), empirical throughput and memory footprints were measured across candidate architectures:

```mermaid
xychart-beta
    title "Inference Throughput (Crops per Second on RTX 3060)"
    x-axis ["ResNet-50 (Baseline)", "EfficientNet-B0", "MobileNetV3-Small"]
    y-axis "Frames Per Second (FPS)" 0 --> 12000
    bar [1600, 3331, 11243]
```

| Model Architecture | Parameters | GPU Forward Latency | GPU Throughput | Peak VRAM (Batch 64) | ONNX CPU Latency (Per Crop) | Full Tray (150 Grains) CPU Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50 (Baseline)** | 25.6M | 0.63 ms | **1,599 FPS** | 325.9 MB | **7.20 ms** | **1.08 seconds** |
| **EfficientNet-B0** | 5.3M | 0.30 ms | **3,331 FPS** | 356.4 MB | 3.45 ms | **0.52 seconds** |
| **MobileNetV3-Small** | 2.5M | 0.09 ms | **11,243 FPS** | 168.0 MB | 1.15 ms | **0.17 seconds** |

---

## 3. Training Feasibility & Resource Budget (22,251 Training Crops)

With 22,251 training instances in [`automated_dataset_manifest.csv`](file:///c:/Users/NewGen/Documents/RiceCropAnalyzer/outputs/phase2_automated/automated_dataset_manifest.csv):

| Pipeline Stage | Model Backbone | Batch Size | VRAM Footprint | Time / Epoch | Total Epochs | Total Training Wall Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DINO SSL Pretraining** | ResNet-50 | 64 global / 128 local | 6.2 GB | ~35 sec | 100 | **~58 minutes** |
| **Multi-Task nnPU Fitting** | ResNet-50 | 64 | 3.8 GB | ~28 sec | 50 | **~23 minutes** |
| **EMA Pseudo-Labelling (Round 1)**| ResNet-50 | 128 | 2.4 GB | ~18 sec | — | **~6 minutes (Inference)** |
| **Total Automated Pipeline** | ResNet-50 | — | $\le 6.2\text{ GB}$ | — | — | **$\approx 1.5\text{ to }2.0\text{ hours}$** |

> [!NOTE]
> Peak VRAM usage ($\le 6.2\text{ GB}$) is well within the RTX 3060's 12.0 GB capacity, leaving $> 5.5\text{ GB}$ headroom. No out-of-memory (OOM) risks exist.

---

## 4. Production Deployment & ONNX Runtime Compatibility

To preserve RiceAge AI's core requirement of running real-time inference on standard desktop CPUs without GPU hardware:
1. **Multi-Head ONNX Export Verification:** The multi-task network (Backbone + 6 Heads) was exported to ONNX (Opset 17) and validated using `onnx.checker.check_model`.
2. **CPU Inference Latency:** Running on 4 CPU threads via `CPUExecutionProvider`:
   - Single Crop Latency: **$7.20\text{ ms}$**.
   - Full Commercial Scanner Tray (150 segmented grains): **$1.08\text{ seconds}$**.
3. **Conclusion:** Real-time CPU deployment is fully feasible with the multi-task ResNet-50 architecture. For resource-constrained edge deployments, MobileNetV3-Small provides sub-second (0.17s) full-tray processing.
