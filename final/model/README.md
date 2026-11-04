# RiceAge Quality Classification ML Model (Milestone M4)

## Architecture Overview

This directory contains the machine learning models and standalone inference engine for the Rice Grain Quality Analyzer.

### Files
- **`quality_classifier_weights.json`**: Production weights for the M4 Multi-Head Quality Classifier.
  Includes calibrated regularized Logistic Regression heads with z-score normalization matrices for:
  1. `foreign_matter`: Non-rice contaminants (stones, weed seeds, chaff).
  2. `whole_broken`: Whole grains (Head Rice $\ge 75\%$ length) vs. broken fragments.
  3. `chalky`: Milky/chalky endosperm defect ($\ge 50\%$ opaque area).
  4. `discoloured`: Peck, enzymatic browning, heat discoloration, and red striations.
  5. `damaged`: Insect-chewed cavitation, cracks, and structural deformities.
- **`wb_classifier_weights.json`**: High-accuracy Whole vs. Broken baseline binary classifier ($99.6\%$ accuracy).
- **`inference.py`**: Pure OpenCV + NumPy inference engine (`QualityInferenceEngine`).
  - No heavy PyTorch/TensorFlow runtimes needed.
  - Zero latency: processes hundreds of grains per second on ARM Raspberry Pi.
  - Segments grains, calculates morphology, photometry, CIELAB & HSV color metrics, and computes exact sigmoid probabilities.

## Mathematical Formulation

Each head computes predictions using calibrated logistic regression logits:
$$z = \sum_{i=1}^{D} w_i \cdot \frac{x_i - \mu_i}{\sigma_i} + b$$
$$P(\text{class} = 1) = \frac{1}{1 + e^{-z}}$$

Where:
- $\mathbf{x}$: Extracted physical feature vector (morphology, CIELAB $L^*, a^*, b^*$, HSV, contour convexity defects).
- $\boldsymbol{\mu}, \boldsymbol{\sigma}$: Training feature mean and standard deviation vectors (`scaler_mean`, `scaler_scale`).
- $\mathbf{w}$: Calibrated head weights (`coef`).
- $b$: Head intercept (`intercept`).
- Prediction is positive if $P \ge \text{threshold}$ (typically $0.50$ to $0.60$).

## Usage Example

```python
from final.model.inference import QualityInferenceEngine

engine = QualityInferenceEngine("final/model/quality_classifier_weights.json")
results = engine.analyze_image(
    image_path="sample.jpg",
    output_annotated_path="annotated.jpg",
    commodity="Raw Rice",
    sample_serial="Sample-0001"
)

print("Total grains detected:", results["total_grains"])
print("Broken grain %:", results["stats"][1]["pct"])
print("Assigned grade:", results["disposition"])
```
