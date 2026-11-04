# RiceAge Final Production Bundle

Integrated package containing the **Raspberry Pi Kiosk Application** and the **Production ML Model**.

---

## Directory Structure

```text
RiceAge/final/
├── app.py                      # Main Raspberry Pi script (Tkinter kiosk + Flask server + Real ML)
├── run.sh                      # One-click launcher script
├── requirements.txt            # Python dependencies (OpenCV, NumPy, Flask, Pillow, QRCode)
├── model/                      # Production Machine Learning Models
│   ├── quality_classifier_weights.json  # M4 Multi-Head weights (FM, Whole/Broken, Chalky, Damaged, Discoloured)
│   ├── wb_classifier_weights.json       # M2.1 Whole vs. Broken baseline weights
│   ├── inference.py                     # High-performance OpenCV + NumPy inference engine
│   └── README.md                        # Documentation of weights and mathematical formulation
├── static/
│   ├── uploads/                # Stores raw and annotated scans (latest_scan.jpg, scan_<ts>.jpg)
│   └── history.json            # Persistent scan records and historical results
└── templates/
    └── index.html              # Responsive Laboratory Web Dashboard (served live by Flask)
```

---

## Architecture Highlights

1. **Integrated Real Machine Learning**:
   - `app.py` runs [`QualityInferenceEngine`](file:///home/anubhavtripathi/Documents/RiceAge/final/model/inference.py) from `model/inference.py`.
   - Every grain contour is segmented via OpenCV, features are extracted (Morphology, CIELAB $L^*, a^*, b^*$, HSV, Convexity Defects), and predictions are evaluated using calibrated multi-head logistic regressions from `quality_classifier_weights.json`.
   - Bounding boxes are drawn with genuine color codes:
     - 🟩 **`OK`** (Whole / Sound Head Rice)
     - 🟨 **`BRK`** (Broken Rice)
     - 🟦 **`CHK`** (Chalky Endosperm)
     - 🟥 **`DMG`** (Insect Chewed / Cavitated)
     - 🟪 **`DISC`** (Discoloured / Peck)
     - 🟧 **`FM`** (Foreign Matter / Non-Rice)

2. **Self-Contained Web Dashboard**:
   - Flask on port 5000 serves `templates/index.html`.
   - Displays the 4 colored KPI cards and analysis parameter table.
   - Live updates via `/api/results` every 3 seconds.
   - Anyone scanning the on-screen QR code can view the dashboard on mobile or laptop.

3. **Touchscreen Kiosk**:
   - Fullscreen scaling for any display resolution.
   - Wi-Fi manager (`nmcli` + on-screen keyboard `wvkbd-mobintl`).
   - Administrator PIN lock (`1234`) to exit.
   - Animated StepRow spinners and countdown timers.

---

## How to Run

### On Raspberry Pi
```bash
cd final
pip install -r requirements.txt
./run.sh
```

### On Desktop / Development Linux
```bash
cd final
python3 app.py
```
Or with virtual environment:
```bash
../.venv/bin/python app.py
```
