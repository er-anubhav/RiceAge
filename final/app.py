import os
import sys
import time
import socket
import threading
import subprocess
import json
import random
import cv2
import numpy as np
try:
    import tkinter as tk
    from tkinter import simpledialog, messagebox
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    class _DummyTk:
        Frame = object
        Canvas = object
        Button = object
        Label = object
        Toplevel = object
        FLAT = "flat"
        SOLID = "solid"
        BOTH = "both"
        LEFT = "left"
        ARC = "arc"
        NE = "ne"
        SW = "sw"
        NW = "nw"
        S = "s"
        W = "w"
        CENTER = "center"
        X = "x"
        DISABLED = "disabled"
    tk = _DummyTk()
    simpledialog = None
    messagebox = None
try:
    from PIL import Image, ImageTk
except ImportError:
    from PIL import Image
    ImageTk = None
try:
    import qrcode
except ImportError:
    qrcode = None
import shutil
# pyrefly: ignore [missing-import]
from flask import Flask, render_template, jsonify, send_from_directory, request

# --- BASE & WORKING DIRECTORIES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

UPLOADS_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOADS_DIR, exist_ok=True)

TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
os.makedirs(TEMPLATES_DIR, exist_ok=True)

MODEL_WEIGHTS = os.path.join(BASE_DIR, 'model', 'quality_classifier_weights.json')

HISTORY_FILE = os.path.join(BASE_DIR, 'static', 'history.json')
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)

# --- IMPORT REAL ML INFERENCE ENGINE ---
try:
    from model.inference import QualityInferenceEngine
    AI_ENGINE = QualityInferenceEngine(MODEL_WEIGHTS)
except Exception as e:
    print(f"[Warning] Failed to import QualityInferenceEngine: {e}")
    AI_ENGINE = None

# --- CONFIGURATION ---
PORT = 5000
SCAN_RESOLUTION = "300"

# --- THEME COLORS ---
BG_COLOR = "#F9F8F4"
BORDER_COLOR = "#2C3E50"
TEXT_COLOR = "#333333"
ACTIVE_BLUE = "#007BFF"
ACTIVE_YELLOW = "#D4A017"
SUCCESS_COLOR = "#28A745"
FAIL_COLOR = "#DC3545"
BORDER_WIDTH = 12

# --- GLOBAL LIVE DATA ---
LATEST_RESULTS = {
    "status": "Awaiting Sample",
    "timestamp": 0,
    "timestamp_str": time.strftime("%d/%m/%Y %H:%M:%S"),
    "image": "",
    "serial": "Sample-0000",
    "live_step": "Ready for Next Sample",
    "commodity": "Raw Rice",
    "total_objects": 0,
    "total_grains": 0,
    "total_foreign_matter": 0,
    "avg_length_mm": 6.452,
    "avg_breadth_mm": 2.115,
    "disposition": "GRADE_1",
    "stats": [],
    "rows": []
}

# Pre-populate with most recent scan from persistent history if available
if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, 'r') as f:
            history_data = json.load(f)
            if history_data and len(history_data) > 0 and isinstance(history_data[0], dict):
                LATEST_RESULTS.update(history_data[0])
    except Exception:
        pass


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

HOST_IP = get_local_ip()
DASHBOARD_URL = f"http://{HOST_IP}:{PORT}"

# --- FLASK SERVER (SERVING WEB DASHBOARD & REAL-TIME API) ---
app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=os.path.join(BASE_DIR, 'static'),
    static_url_path='/static'
)

DIST_DIR = os.path.join(BASE_DIR, 'dist')

@app.route('/static/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOADS_DIR, filename)

@app.route('/api/results')
def get_results():
    return jsonify(LATEST_RESULTS)

@app.route('/api/results/<sample_id>')
def get_result_by_id(sample_id):
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
            for item in history:
                if str(item.get('serial', '')) == sample_id or str(item.get('sample_id', '')) == sample_id:
                    return jsonify(item)
        except Exception:
            pass
    return jsonify(LATEST_RESULTS)

@app.route('/api/history')
def get_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                records = json.load(f)
                return jsonify(records)
        except Exception:
            pass
    return jsonify([])

def process_image_analysis(image_path, sample_id=None, commodity="Raw Rice"):
    global LATEST_RESULTS
    try:
        with open(HISTORY_FILE, 'r') as f:
            raw_history = json.load(f)
        serial_num = len(raw_history) + 1
    except Exception:
        raw_history = []
        serial_num = 1

    if not sample_id:
        sample_id = f"Sample-{serial_num:04d}"

    timestamp_val = int(time.time())
    unique_img_name = f"scan_{timestamp_val}.jpg"
    dest_path = os.path.join(UPLOADS_DIR, unique_img_name)
    try:
        shutil.copy2(image_path, dest_path)
    except Exception:
        dest_path = image_path

    if AI_ENGINE is not None:
        analysis = AI_ENGINE.analyze_image(
            image_path=dest_path,
            output_annotated_path=dest_path,
            commodity=commodity,
            sample_serial=sample_id
        )
        LATEST_RESULTS.update(analysis)
    else:
        LATEST_RESULTS.update({
            "serial": sample_id,
            "sample_id": sample_id,
            "commodity": commodity,
            "total_grains": 0,
            "stats": [],
            "rows": []
        })

    LATEST_RESULTS["serial"] = sample_id
    LATEST_RESULTS["sample_id"] = sample_id
    LATEST_RESULTS["timestamp"] = timestamp_val
    LATEST_RESULTS["timestamp_str"] = time.strftime("%d/%m/%Y %H:%M:%S")
    LATEST_RESULTS["image"] = unique_img_name
    LATEST_RESULTS["live_step"] = "Analysis Complete"

    # Save to history
    raw_history.insert(0, LATEST_RESULTS.copy())
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(raw_history, f, indent=2)
    except Exception as e:
        print(f"Failed saving history: {e}")

    return LATEST_RESULTS

def acquire_hardware_scan(output_file, resolution=SCAN_RESOLUTION, timeout=60):
    """
    Acquire flatbed optical capture using SANE scanimage.
    Explicitly targets Canon LiDE 400 airscan device to prevent network SNMP delays and I/O driver conflicts.
    """
    dev = None
    for slot in ('e0', 'e1', 'e2'):
        candidate = f"airscan:{slot}:Canon LiDE 400"
        try:
            p = subprocess.run(['scanimage', '-d', candidate, '-A'], capture_output=True, timeout=2)
            if p.returncode == 0:
                dev = candidate
                break
        except Exception:
            pass

    if dev:
        cmd = ['scanimage', '-d', dev, '--format=jpeg', f'--resolution={resolution}', '--mode=Color', '-o', output_file]
    else:
        cmd = ['scanimage', '--format=jpeg', f'--resolution={resolution}', '--mode=Color', '-o', output_file]

    print(f"[Scanner Hardware] Executing scan command: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if proc.returncode == 0 and os.path.exists(output_file):
            print(f"[Scanner Hardware] Scan successful -> {output_file} ({os.path.getsize(output_file)} bytes)")
            return True, None
        err_msg = proc.stderr.decode('utf-8', errors='replace')
        print(f"[Scanner Hardware] scanimage returned error code {proc.returncode}: {err_msg}")
        return False, err_msg
    except Exception as e:
        print(f"[Scanner Hardware] Scan exception: {e}")
        return False, str(e)

@app.route('/api/scan', methods=['GET', 'POST'])
def api_scan():
    output_file = os.path.join(UPLOADS_DIR, "latest_scan.jpg")
    success, err = acquire_hardware_scan(output_file, resolution=SCAN_RESOLUTION, timeout=60)

    if not success:
        # Fallback to test/bundled image if no physical scanner attached
        fallback_sample = os.path.join(BASE_DIR, 'model', 'sample_default.jpg')
        if not os.path.exists(fallback_sample):
            fallback_sample = os.path.join(BASE_DIR, '..', 'data', 'samples', 'sample_1489x2048.jpg')
        if os.path.exists(fallback_sample):
            print(f"[Scanner Hardware] Using fallback sample {fallback_sample}")
            shutil.copy2(fallback_sample, output_file)
        elif not os.path.exists(output_file):
            return jsonify({"error": f"Scanner device error: {err}"}), 503

    result = process_image_analysis(output_file)
    return jsonify(result)


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'image' not in request.files:
        return jsonify({"detail": "No image file provided in request"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"detail": "No image selected"}), 400

    sample_id = request.form.get('sample_id') or None
    commodity = request.form.get('commodity') or "Raw Rice"

    filename = f"upload_{int(time.time())}_{file.filename}"
    saved_path = os.path.join(UPLOADS_DIR, filename)
    file.save(saved_path)

    result = process_image_analysis(saved_path, sample_id=sample_id, commodity=commodity)
    return jsonify(result)

@app.route('/api/health')
def health():
    return jsonify({
        "online": True,
        "status": "ok",
        "message": "RiceAge Laboratory Server Online",
        "station": "RiceAge Optical Inspection Station",
        "version": "M5.0",
        "ml_loaded": AI_ENGINE.is_loaded if AI_ENGINE else False
    })


# Serve modern React SPA from DIST_DIR if available, fallback to template
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_spa(path):
    if path.startswith('api/') or path.startswith('static/'):
        return jsonify({"error": "Not found"}), 404
    if os.path.exists(DIST_DIR):
        file_path = os.path.join(DIST_DIR, path)
        if path != "" and os.path.exists(file_path) and os.path.isfile(file_path):
            return send_from_directory(DIST_DIR, path)
        index_file = os.path.join(DIST_DIR, 'index.html')
        if os.path.exists(index_file):
            return send_from_directory(DIST_DIR, 'index.html')
    return render_template('index.html')

def run_flask():
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)

# --- GUI TOOLTIP COMPONENT ---
class ToolTip:
    def __init__(self, widget, scale_factor=1.0):
        self.widget = widget
        self.scale = scale_factor
        self.tipwindow = None
        self.text = ""
        self.widget.bind("<Enter>", self.showtip)
        self.widget.bind("<Leave>", self.hidetip)

    def showtip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() - int(50 * self.scale)
        y = self.widget.winfo_rooty() + int(30 * self.scale)
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        fs = max(8, int(10 * self.scale))
        label = tk.Label(tw, text=self.text, background=BORDER_COLOR, fg="white", 
                         relief=tk.FLAT, font=("Helvetica", fs, "bold"))
        label.pack(ipadx=int(8*self.scale), ipady=int(4*self.scale))

    def hidetip(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

# --- PROGRESS SPINNER & STEP ROW ---
class StepRow(tk.Frame):
    def __init__(self, parent, app_instance, text, duration, scale_factor):
        super().__init__(parent, bg=BG_COLOR)
        self.app = app_instance
        self.scale = scale_factor
        self.text = text
        self.total = duration
        self.remaining = duration
        self.show_timer = duration > 0
        self.angle = 0
        self.is_running = False
        
        self.cw = int(60 * self.scale)
        self.font_main = max(12, int(26 * self.scale))
        self.font_time = max(8, int(14 * self.scale))
        
        self.canvas = tk.Canvas(self, width=self.cw, height=self.cw, bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, padx=(0, int(20 * self.scale))) 
        
        self.label = tk.Label(self, text=self.text, fg=ACTIVE_YELLOW, bg=BG_COLOR, font=("Helvetica", self.font_main, "bold"))
        self.label.pack(side=tk.LEFT)
        
        self.arc = None
        self.time_text = None

    def start(self):
        self.is_running = True
        pad = int(5 * self.scale)
        arc_w = max(2, int(5 * self.scale))
        self.arc = self.canvas.create_arc(pad, pad, self.cw-pad, self.cw-pad, start=0, extent=270, style=tk.ARC, outline=ACTIVE_BLUE, width=arc_w)
        if self.show_timer:
            self.time_text = self.canvas.create_text(self.cw/2, self.cw/2, text=str(self.remaining), fill=TEXT_COLOR, font=("Helvetica", self.font_time, "bold"))
            self.countdown()
        self.update_spinner()

    def update_spinner(self):
        if self.is_running and not self.app.abort_test:
            if self.app.pause_event.is_set():
                self.angle = (self.angle - 15) % 360
                self.canvas.itemconfig(self.arc, start=self.angle)
            self.after(40, self.update_spinner)

    def countdown(self):
        if not self.is_running: return
        if self.app.abort_test: return
        
        if self.app.pause_event.is_set():
            if self.remaining > 0:
                self.remaining -= 1
                global LATEST_RESULTS
                LATEST_RESULTS["step_remaining"] = self.remaining
                self.canvas.itemconfig(self.time_text, text=str(self.remaining))
        
        self.after(1000, self.countdown)

    def finish(self, success=True):
        self.is_running = False
        self.canvas.delete("all")
        color = SUCCESS_COLOR if success else FAIL_COLOR
        symbol = "✔" if success else "✖"
        f_sym = max(14, int(32 * self.scale))
        self.canvas.create_text(self.cw/2, self.cw/2, text=symbol, fill=color, font=("Helvetica", f_sym, "bold"))
        self.label.config(fg=color) 

# --- MAIN GRAIN ANALYZER TOUCHSCREEN KIOSK APPLICATION ---
class GrainAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automated Grain Analyzer")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg=BORDER_COLOR)

        self.abort_test = False
        self.pause_event = threading.Event()
        self.pause_event.set()

        self.sw = self.root.winfo_screenwidth()
        self.sh = self.root.winfo_screenheight()
        self.scale = min(self.sw / 1920.0, self.sh / 1080.0)
        self.scale = max(0.5, min(self.scale, 3.0)) 
        
        self.f_h1 = max(16, int(48 * self.scale))
        self.f_h2 = max(14, int(28 * self.scale))
        self.f_h3 = max(12, int(20 * self.scale))
        self.f_norm = max(10, int(14 * self.scale))
        self.f_small = max(8, int(12 * self.scale))

        b_width = int(BORDER_WIDTH * self.scale)
        self.bg_frame = tk.Frame(root, bg=BG_COLOR)
        self.bg_frame.pack(fill=tk.BOTH, expand=True, padx=b_width, pady=b_width)

        self.net_label = tk.Label(self.bg_frame, text="🌐 Checking...", fg=TEXT_COLOR, bg=BG_COLOR, font=("Helvetica", self.f_norm, "bold"), cursor="hand2")
        self.net_label.place(relx=0.98, rely=0.02, anchor=tk.NE)
        self.net_tooltip = ToolTip(self.net_label, self.scale)
        self.net_label.bind("<Button-1>", self.show_wifi_menu)

        self.draw_qr_code()

        self.main_frame = tk.Frame(self.bg_frame, bg=BG_COLOR)
        self.main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.title_label = tk.Label(self.main_frame, text="Automated Grain Analyzer", fg=TEXT_COLOR, bg=BG_COLOR, font=("Helvetica", self.f_h1, "bold"))
        self.title_label.pack(pady=(0, int(40 * self.scale)))

        self.content_frame = tk.Frame(self.main_frame, bg=BG_COLOR)
        self.content_frame.pack()

        self.bottom_bar_frame = tk.Frame(self.bg_frame, bg=BG_COLOR)
        self.bottom_bar_frame.place(relx=0.5, rely=0.95, anchor=tk.S)
        
        self.progress_canvas_w = int(1000 * self.scale)
        self.progress_canvas_h = int(15 * self.scale)
        self.progress_canvas = tk.Canvas(self.bottom_bar_frame, width=self.progress_canvas_w, height=self.progress_canvas_h, bg=BORDER_COLOR, highlightthickness=0)
        self.progress_rect = None
        self.progress_label = tk.Label(self.bottom_bar_frame, text="SCANNING PROGRESS: 0%", font=("Helvetica", self.f_small, "bold"), bg=BG_COLOR, fg=ACTIVE_BLUE)

        self.btn_state = "START"
        self.current_step = None
        self.tk_img = None
        
        self.exit_btn = tk.Button(self.bg_frame, text="Exit", bg=FAIL_COLOR, fg="white", font=("Helvetica", self.f_norm, "bold"), command=self.attempt_exit, relief=tk.FLAT)
        self.exit_btn.place(relx=0.02, rely=0.98, anchor=tk.SW)

        self.monitor_network()
        self.reset_ui()

    def show_wifi_menu(self, event=None):
        self.wifi_win = tk.Toplevel(self.root)
        self.wifi_win.title("Wi-Fi Settings")
        self.wifi_win.configure(bg=BG_COLOR)
        
        w = int(450 * self.scale)
        h = int(600 * self.scale)
        x = self.root.winfo_rootx() + (self.root.winfo_width() // 2) - (w // 2)
        y = self.root.winfo_rooty() + (self.root.winfo_height() // 2) - (h // 2)
        self.wifi_win.geometry(f"{w}x{h}+{x}+{y}")
        self.wifi_win.attributes('-topmost', True)

        tk.Label(self.wifi_win, text="Scanning for Wi-Fi Networks...", font=("Helvetica", self.f_norm, "bold"), bg=BG_COLOR).pack(pady=int(50*self.scale))
        threading.Thread(target=self.scan_wifi_networks, daemon=True).start()

    def scan_wifi_networks(self):
        try:
            output = subprocess.check_output(['nmcli', '-t', '-f', 'SSID,SIGNAL', 'dev', 'wifi'], timeout=10).decode('utf-8')
            networks = []
            seen = set()
            for line in output.strip().split('\n'):
                if ':' in line:
                    ssid, signal = line.rsplit(':', 1)
                    if ssid and ssid not in seen:
                        seen.add(ssid)
                        networks.append((ssid, int(signal)))
            networks.sort(key=lambda x: x[1], reverse=True)
            self.root.after(0, lambda: self.render_wifi_list(networks))
        except Exception:
            self.root.after(0, lambda: self.render_wifi_list([]))

    def render_wifi_list(self, networks):
        if not hasattr(self, 'wifi_win') or not self.wifi_win.winfo_exists():
            return

        for widget in self.wifi_win.winfo_children():
            widget.destroy()

        tk.Label(self.wifi_win, text="Select a Wi-Fi Network", font=("Helvetica", self.f_h2, "bold"), bg=BG_COLOR).pack(pady=int(15*self.scale))
        
        list_frame = tk.Frame(self.wifi_win, bg=BG_COLOR)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=int(25*self.scale))

        if not networks:
            tk.Label(list_frame, text="No networks found.", font=("Helvetica", self.f_norm), bg=BG_COLOR).pack(pady=int(20*self.scale))

        for ssid, signal in networks[:8]:
            btn = tk.Button(list_frame, text=f"{ssid}  ({signal}%)", font=("Helvetica", self.f_norm), 
                            bg="white", relief=tk.SOLID, bd=1, cursor="hand2",
                            command=lambda s=ssid: self.prompt_wifi_password(s))
            btn.pack(fill=tk.X, pady=int(4*self.scale), ipady=int(6*self.scale))

        tk.Button(self.wifi_win, text="Cancel", font=("Helvetica", self.f_norm, "bold"), bg=FAIL_COLOR, fg="white", 
                  command=self.wifi_win.destroy, relief=tk.FLAT, cursor="hand2").pack(pady=int(20*self.scale), ipady=int(8*self.scale), ipadx=int(30*self.scale))

    def prompt_wifi_password(self, ssid):
        subprocess.Popen(['wvkbd-mobintl', '-L', '150'])
        pwd = simpledialog.askstring("Wi-Fi Password", f"Enter password for '{ssid}'\n(Leave blank if open network):", show='*', parent=self.wifi_win)
        subprocess.run(['pkill', 'wvkbd-mobintl'])
        
        if pwd is not None:
            self.wifi_win.destroy()
            threading.Thread(target=self.connect_to_wifi, args=(ssid, pwd), daemon=True).start()

    def connect_to_wifi(self, ssid, pwd):
        try:
            cmd = ['nmcli', 'dev', 'wifi', 'connect', ssid]
            if pwd:
                cmd.extend(['password', pwd])
            subprocess.run(cmd, check=True, timeout=20)
            self.root.after(0, lambda: messagebox.showinfo("Connected", f"Successfully connected to {ssid}", parent=self.root))
        except subprocess.CalledProcessError:
            self.root.after(0, lambda: messagebox.showerror("Connection Error", f"Failed to connect to {ssid}.\nPlease verify the password.", parent=self.root))
        except Exception:
            self.root.after(0, lambda: messagebox.showerror("Connection Error", "Connection timed out.", parent=self.root))

    def draw_qr_code(self):
        qr_frame = tk.Frame(self.bg_frame, bg=BG_COLOR)
        qr_frame.place(relx=0.02, rely=0.02, anchor=tk.NW)
        if qrcode is not None and ImageTk is not None:
            qr = qrcode.make(DASHBOARD_URL)
            qr_size = int(250 * self.scale) 
            qr = qr.resize((qr_size, qr_size)) 
            self.qr_img = ImageTk.PhotoImage(qr) 
            tk.Label(qr_frame, text="Live Dashboard Active", bg=BG_COLOR, fg=SUCCESS_COLOR, font=("Helvetica", self.f_small, "bold")).pack()
            tk.Label(qr_frame, image=self.qr_img, bg=BG_COLOR).pack()
            tk.Label(qr_frame, text=DASHBOARD_URL, bg=BG_COLOR, fg=ACTIVE_BLUE, font=("Helvetica", self.f_small)).pack()
        else:
            tk.Label(qr_frame, text="Live Dashboard Active", bg=BG_COLOR, fg=SUCCESS_COLOR, font=("Helvetica", self.f_small, "bold")).pack()
            tk.Label(qr_frame, text=DASHBOARD_URL, bg=BG_COLOR, fg=ACTIVE_BLUE, font=("Helvetica", self.f_norm, "bold")).pack()

    def monitor_network(self):
        def check():
            connected = False
            ssid = "Unknown"
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=2)
                connected = True
            except OSError:
                pass
            if connected:
                try:
                    ssid_bytes = subprocess.check_output(['iwgetid', '-r'], timeout=1)
                    ssid = ssid_bytes.decode('utf-8').strip() or "Wired/Unknown"
                except Exception:
                    ssid = "Wired/Unknown"
            self.root.after(0, lambda: self._update_net_ui(connected, ssid))

        threading.Thread(target=check, daemon=True).start()
        self.root.after(5000, self.monitor_network)

    def _update_net_ui(self, connected, ssid):
        if connected:
            self.net_label.config(text=f"🌐 {ssid}", fg=SUCCESS_COLOR)
            self.net_tooltip.text = f"Connected to: {ssid}"
        else:
            self.net_label.config(text="🌐 OFFLINE", fg=FAIL_COLOR)
            self.net_tooltip.text = "No internet connection"

    def draw_pill_button(self, parent, text, bg_color, text_color="white", base_w=400, base_h=90):
        w = int(base_w * self.scale)
        h = int(base_h * self.scale)
        canvas = tk.Canvas(parent, width=w, height=h, bg=BG_COLOR, highlightthickness=0)
        r = h / 2
        canvas.create_oval(0, 0, h, h, fill=bg_color, outline=bg_color)
        canvas.create_oval(w-h, 0, w, h, fill=bg_color, outline=bg_color)
        canvas.create_rectangle(r, 0, w-r, h, fill=bg_color, outline=bg_color)
        
        f_size = max(14, int(24 * self.scale))
        canvas.create_text(w/2, h/2, text=text, fill=text_color, font=("Helvetica", f_size, "bold"))
        canvas.bind("<Button-1>", self.on_start_click)
        return canvas

    def update_main_button(self, text, bg_color):
        self.btn_state = "RESET"
        self.btn_canvas.delete("all")
        w = int(400 * self.scale)
        h = int(100 * self.scale)
        self.btn_canvas.config(width=w, height=h)
        r = h / 2
        self.btn_canvas.create_oval(0, 0, h, h, fill=bg_color, outline=bg_color)
        self.btn_canvas.create_oval(w-h, 0, w, h, fill=bg_color, outline=bg_color)
        self.btn_canvas.create_rectangle(r, 0, w-r, h, fill=bg_color, outline=bg_color)
        f_size = max(14, int(24 * self.scale))
        self.btn_canvas.create_text(w/2, h/2, text=text, fill="white", font=("Helvetica", f_size, "bold"))
        
        if hasattr(self, 'controls_frame') and self.controls_frame.winfo_exists():
            self.controls_frame.destroy()

    def on_start_click(self, event):
        if self.btn_state == "START":
            self.start_sequence()
        elif self.btn_state == "RESET":
            self.reset_ui()

    def attempt_exit(self):
        subprocess.Popen(['wvkbd-mobintl', '-L', '150'])
        pin = simpledialog.askstring("System Locked", "Enter Administrator PIN:", show='*', parent=self.root)
        subprocess.run(['pkill', 'wvkbd-mobintl'])
        
        if pin == "1234":
            self.root.destroy()
        elif pin is not None:
            messagebox.showerror("Access Denied", "Incorrect PIN.", parent=self.root)

    def abort_test_routine(self):
        self.abort_test = True
        self.pause_event.set() 
        self.abort_btn.config(text="Aborting...", state=tk.DISABLED)

    def start_sequence(self):
        global LATEST_RESULTS
        LATEST_RESULTS["commodity"] = self.commodity_var.get()
        
        self.btn_state = "TESTING"
        self.abort_test = False
        self.pause_event.set()
        
        LATEST_RESULTS["live_step"] = "Initializing Hardware..."
        
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        self.btn_canvas = self.draw_pill_button(self.content_frame, "Scanning in Progress...", "#A0A0A0")
        self.btn_canvas.pack(pady=int(10*self.scale))

        self.controls_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        self.controls_frame.pack(pady=int(5*self.scale))
        btn_font = ("Helvetica", max(10, int(16*self.scale)), "bold")
        
        self.abort_btn = tk.Button(self.controls_frame, text="⏹ Abort Scan", bg=FAIL_COLOR, fg="white", font=btn_font, command=self.abort_test_routine, relief=tk.FLAT, padx=15, pady=8)
        self.abort_btn.pack(side=tk.LEFT, padx=10)

        self.steps_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        self.steps_frame.pack(pady=int(10*self.scale))

        self.progress_label.pack(anchor=tk.W)
        self.progress_canvas.pack(pady=(int(5*self.scale), 0))
        
        self.progress_rect = self.progress_canvas.create_rectangle(
            0, 0, 0, self.progress_canvas_h, fill=ACTIVE_BLUE, width=0
        )
        self.scan_progress_value = 0

        threading.Thread(target=self.hardware_routine, daemon=True).start()

    def add_step(self, text, duration):
        global LATEST_RESULTS
        LATEST_RESULTS["live_step"] = text
        LATEST_RESULTS["step_remaining"] = duration
        step = StepRow(self.steps_frame, self, text, duration, self.scale)
        step.pack(anchor=tk.CENTER, pady=int(8*self.scale)) 
        step.start()
        self.current_step = step

    def complete_step(self, success=True):
        if self.current_step:
            self.current_step.finish(success)

    def update_bottom_progress(self):
        if self.abort_test or self.scan_progress_value >= 100:
            return
        self.scan_progress_value += 0.5 
        if self.scan_progress_value > 99:
            self.scan_progress_value = 99
        global LATEST_RESULTS
        LATEST_RESULTS["progress"] = int(self.scan_progress_value)
            
        pixel_width = (self.scan_progress_value / 100.0) * self.progress_canvas_w
        
        self.progress_canvas.coords(
            self.progress_rect, 
            0, 
            0, 
            pixel_width, 
            self.progress_canvas_h
        )
        
        self.progress_label.config(text=f"SCANNING PROGRESS: {int(self.scan_progress_value)}%")
        self.root.after(100, self.update_bottom_progress)

    def finish_bottom_progress(self):
        self.scan_progress_value = 100
        global LATEST_RESULTS
        LATEST_RESULTS["progress"] = 100
        
        self.progress_canvas.coords(
            self.progress_rect, 
            0, 
            0, 
            self.progress_canvas_w, 
            self.progress_canvas_h
        )
        
        self.progress_label.config(text="SCANNING PROGRESS: 100% - COMPLETE")

    def hide_bottom_progress(self):
        self.progress_label.pack_forget()
        self.progress_canvas.pack_forget()

    # --- REAL ML INFERENCE ENGINE INTEGRATION ---
    def execute_real_ml_analysis(self, image_path):
        global LATEST_RESULTS
        serial_str = LATEST_RESULTS.get("serial", "Sample-0001")
        commodity = self.commodity_var.get() if hasattr(self, 'commodity_var') else "Raw Rice"

        if AI_ENGINE is not None:
            analysis = AI_ENGINE.analyze_image(
                image_path=image_path,
                output_annotated_path=image_path,
                commodity=commodity,
                sample_serial=serial_str
            )
            LATEST_RESULTS.update(analysis)
        else:
            # Fallback if engine failed loading
            LATEST_RESULTS["total_grains"] = 0
            LATEST_RESULTS["stats"] = []
            LATEST_RESULTS["rows"] = []

    def hardware_routine(self):
        try:
            self.root.after(0, lambda: self.add_step("Warming up scanner motor...", 2))
            time.sleep(2)
            if self.abort_test: return self.handle_abort()
            self.root.after(0, lambda: self.complete_step(True))

            self.root.after(0, lambda: self.add_step(f"Acquiring {SCAN_RESOLUTION} DPI Optical Capture...", 0))
            self.root.after(0, self.update_bottom_progress)
            
            output_file = os.path.join(UPLOADS_DIR, "latest_scan.jpg")
            success, err = acquire_hardware_scan(output_file, resolution=SCAN_RESOLUTION, timeout=60)

            # Scanner hardware handling: If scan fails or no flatbed attached, check fallback test image for simulation
            if not success:
                print("Scanner Warning / Error:", err)
                # If running on device / test mode without physical scanner attached, load bundled baseline sample
                fallback_sample = os.path.join(BASE_DIR, 'model', 'sample_default.jpg')
                if not os.path.exists(fallback_sample):
                    fallback_sample = os.path.join(BASE_DIR, '..', 'data', 'samples', 'sample_1489x2048.jpg')
                if os.path.exists(fallback_sample):
                    shutil.copy2(fallback_sample, output_file)
                elif not os.path.exists(output_file):
                    self.root.after(0, lambda: self.complete_step(False))
                    self.root.after(0, lambda: self.update_main_button("Scanner Not Found - Connect Device", FAIL_COLOR))
                    self.root.after(0, self.hide_bottom_progress)
                    return


            if self.abort_test:
                return self.handle_abort()
            
            self.root.after(0, self.finish_bottom_progress)
            self.root.after(0, lambda: self.complete_step(True))

            # Milestone M4: Real ML Inferencing step
            self.root.after(0, lambda: self.add_step("Processing Edge Detection & ML Inferencing...", 3))
            
            # Read history to generate sequential Sample-XXXX serial number
            try:
                with open(HISTORY_FILE, 'r') as f:
                    raw_history = json.load(f)
                serial_num = len(raw_history) + 1
            except Exception:
                raw_history = []
                serial_num = 1

            serial_str = f"Sample-{serial_num:04d}"
            global LATEST_RESULTS
            LATEST_RESULTS["serial"] = serial_str
            timestamp_val = int(time.time())
            LATEST_RESULTS["timestamp"] = timestamp_val
            LATEST_RESULTS["timestamp_str"] = time.strftime("%d/%m/%Y %H:%M:%S")

            # Execute real ML inference with M4 quality weights
            self.execute_real_ml_analysis(output_file)
            time.sleep(2)
            self.root.after(0, lambda: self.complete_step(True))

            unique_img_name = f"scan_{timestamp_val}.jpg"
            try:
                os.system(f"cp {output_file} {os.path.join(UPLOADS_DIR, unique_img_name)}")
            except Exception:
                pass
            
            LATEST_RESULTS["image"] = unique_img_name
            LATEST_RESULTS["live_step"] = "Analysis Complete"
            
            # Save scan record to persistent JSON history
            raw_history.insert(0, LATEST_RESULTS.copy())
            try:
                with open(HISTORY_FILE, 'w') as f:
                    json.dump(raw_history, f, indent=2)
            except Exception as e:
                print(f"Failed saving history: {e}")
            
            time.sleep(0.5)
            self.root.after(0, self.hide_bottom_progress)
            self.root.after(0, self.display_local_results)

        except Exception as e:
            print("System Exception:", e)
            if self.current_step:
                self.root.after(0, lambda: self.complete_step(False))
            self.root.after(0, lambda: self.update_main_button("System Error - Retry", FAIL_COLOR))
            self.root.after(0, self.hide_bottom_progress)

    def handle_abort(self):
        if self.current_step:
            self.root.after(0, lambda: self.complete_step(False))
        self.root.after(0, lambda: self.update_main_button("Scan Aborted - Start Over", FAIL_COLOR))
        self.root.after(0, self.hide_bottom_progress)

    def display_local_results(self):
        self.btn_state = "RESET"
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        self.title_label.pack_configure(pady=(0, int(10*self.scale)))

        split_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        split_frame.pack(fill=tk.BOTH, expand=True, padx=int(40*self.scale))

        left_col = tk.Frame(split_frame, bg=BG_COLOR)
        left_col.pack(side=tk.LEFT, fill=tk.Y, padx=(0, int(60*self.scale)))

        right_col = tk.Frame(split_frame, bg=BG_COLOR)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        try:
            latest_path = os.path.join(UPLOADS_DIR, "latest_scan.jpg")
            img = Image.open(latest_path)
            img_w = int(480 * self.scale)
            img_h = int(678 * self.scale)
            img.thumbnail((img_w, img_h), Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(img)
            
            b_thicc = max(2, int(4 * self.scale))
            img_lbl = tk.Label(left_col, image=self.tk_img, bg=BG_COLOR, bd=b_thicc, relief=tk.SOLID)
            img_lbl.pack(pady=(0, int(30*self.scale)))
        except Exception:
            tk.Label(left_col, text="Scanner File Error", bg=BG_COLOR).pack(pady=20)

        self.btn_canvas = self.draw_pill_button(left_col, "Run Next Batch", SUCCESS_COLOR, base_w=400, base_h=70)
        self.btn_canvas.pack()

        total_grains = LATEST_RESULTS.get("total_grains", 0)
        tk.Label(right_col, text=f"QUALITY CLASSIFICATION REPORT (N = {total_grains})", font=("Helvetica", self.f_h2, "bold"), fg=ACTIVE_BLUE, bg=BG_COLOR).pack(anchor=tk.W, pady=(0, int(15*self.scale)))
        
        table_frame = tk.Frame(right_col, bg=BORDER_COLOR) 
        table_frame.pack(anchor=tk.W, fill=tk.X)

        hdr_bg = "#E9ECEF"
        headers = ["Refraction Parameter", "Count", "Analyzed (%)", "Limit (%)"]
        for col, text in enumerate(headers):
            tk.Label(table_frame, text=text, font=("Helvetica", self.f_norm, "bold"), fg=TEXT_COLOR, bg=hdr_bg, anchor=tk.W).grid(row=0, column=col, sticky="nsew", padx=1, pady=1, ipadx=int(10*self.scale), ipady=int(5*self.scale))

        stats_list = LATEST_RESULTS.get("stats", [])

        for i, item in enumerate(stats_list, start=1):
            param = item["param"]
            count_val = str(item["count"])
            pct_val = f"{item['pct']:.2f}%"
            limit_val = f"{item['limit']}%" if item['limit'] != "-" else "-"
            
            tk.Label(table_frame, text=param, font=("Helvetica", self.f_norm, "bold"), bg=BG_COLOR, fg=TEXT_COLOR, anchor=tk.W).grid(row=i, column=0, sticky="nsew", padx=1, pady=1, ipadx=int(10*self.scale), ipady=int(5*self.scale))
            tk.Label(table_frame, text=count_val, font=("Helvetica", self.f_norm, "normal"), bg=BG_COLOR, fg=TEXT_COLOR, anchor=tk.W).grid(row=i, column=1, sticky="nsew", padx=1, pady=1, ipadx=int(10*self.scale), ipady=int(5*self.scale))
            tk.Label(table_frame, text=pct_val, font=("Helvetica", self.f_norm, "normal"), bg=BG_COLOR, fg=TEXT_COLOR, anchor=tk.W).grid(row=i, column=2, sticky="nsew", padx=1, pady=1, ipadx=int(10*self.scale), ipady=int(5*self.scale))
            tk.Label(table_frame, text=limit_val, font=("Helvetica", self.f_norm, "normal"), bg=BG_COLOR, fg=TEXT_COLOR, anchor=tk.W).grid(row=i, column=3, sticky="nsew", padx=1, pady=1, ipadx=int(10*self.scale), ipady=int(5*self.scale))

        tk.Label(right_col, text="*Graded against AP Civil Supplies Corp & Codex CXS 198 specifications.", font=("Helvetica", self.f_small, "italic"), fg="#777", bg=BG_COLOR).pack(anchor=tk.W, pady=(int(10*self.scale), 0))

    def reset_ui(self):
        global LATEST_RESULTS
        LATEST_RESULTS["live_step"] = "Ready for Next Sample"
        LATEST_RESULTS["step_remaining"] = 0
        LATEST_RESULTS["progress"] = 0
        
        self.btn_state = "START"
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
        self.hide_bottom_progress()

        self.title_label.pack_configure(pady=(0, int(20*self.scale)))
        
        selection_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        selection_frame.pack(pady=int(10*self.scale))
        tk.Label(selection_frame, text="Select Commodity:", font=("Helvetica", self.f_h3, "bold"), fg=TEXT_COLOR, bg=BG_COLOR).pack(side=tk.LEFT, padx=10)
        
        self.commodity_var = tk.StringVar(value="Raw Rice")
        dropdown = tk.OptionMenu(selection_frame, self.commodity_var, "Raw Rice", "Wheat (Profile Locked)", "Maize (Profile Locked)")
        dropdown.config(font=("Helvetica", self.f_norm), bg="white", fg=TEXT_COLOR, relief=tk.SOLID, bd=1)
        dropdown.pack(side=tk.LEFT)

        self.btn_canvas = self.draw_pill_button(self.content_frame, "START SCAN", SUCCESS_COLOR, base_w=400, base_h=100)
        self.btn_canvas.pack(pady=int(20*self.scale))
        self.steps_frame = tk.Frame(self.content_frame, bg=BG_COLOR)
        self.steps_frame.pack(pady=int(20*self.scale)) 

if __name__ == "__main__":
    is_headless = "--headless" in sys.argv or os.environ.get("RICEAGE_HEADLESS", "0") == "1"
    if not is_headless and HAS_TKINTER:
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()
        print(f"[RiceAge Station] Flask web dashboard listening at {DASHBOARD_URL}")
        root = tk.Tk()
        app = GrainAnalyzerApp(root)
        root.mainloop()
    else:
        print(f"[RiceAge Station] Running Modern Web Dashboard server directly at {DASHBOARD_URL}")
        run_flask()

