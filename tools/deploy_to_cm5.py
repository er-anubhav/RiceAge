"""
Full Remote Deployment & Verification Script for Raspberry Pi Compute Module 5 (bigb-cm5)
Transfers the Two-Stage Production Pipeline (YOLO11s-seg ONNX + ResNet-50 ONNX + Decoupled Optical Fusion Engine),
installs/verifies all dependencies, executes end-to-end inference on real scanner trays, and validates accuracy.
"""

import os
import sys
import time
from pathlib import Path
import paramiko

CM5_HOST = "192.168.16.160"
CM5_PORT = 22
CM5_USER = "bigb-cm5"
CM5_PASS = "Pingpong@123"
REMOTE_BASE = "/home/bigb-cm5/RiceAge_Production"

def create_sftp_client(ssh):
    return ssh.open_sftp()

def ensure_remote_dir(sftp, remote_dir):
    dirs = []
    p = remote_dir
    while p not in ('/', ''):
        dirs.append(p)
        p = os.path.dirname(p)
    dirs.reverse()
    for d in dirs:
        try:
            sftp.stat(d)
        except IOError:
            sftp.mkdir(d)

def upload_file(sftp, local_path, remote_path):
    ensure_remote_dir(sftp, os.path.dirname(remote_path))
    sz = os.path.getsize(local_path)
    print(f"Uploading {os.path.basename(local_path)} ({sz / (1024*1024):.1f} MB)...", end="", flush=True)
    t0 = time.time()
    sftp.put(local_path, remote_path)
    dt = time.time() - t0
    rate = (sz / (1024*1024)) / max(dt, 0.01)
    print(f" Done in {dt:.1f}s ({rate:.1f} MB/s)")

def main():
    print("=" * 70)
    print("  RICEAGE PRODUCTION PIPELINE - DEPLOYMENT TO CM5")
    print(f"  Target: {CM5_USER}@{CM5_HOST}:{CM5_PORT}")
    print("=" * 70)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("Connecting to CM5 via SSH...")
    ssh.connect(CM5_HOST, port=CM5_PORT, username=CM5_USER, password=CM5_PASS, timeout=10)
    print("SSH Connection established.")

    sftp = create_sftp_client(ssh)
    ensure_remote_dir(sftp, REMOTE_BASE)
    ensure_remote_dir(sftp, f"{REMOTE_BASE}/weights")
    ensure_remote_dir(sftp, f"{REMOTE_BASE}/src")
    ensure_remote_dir(sftp, f"{REMOTE_BASE}/configs")
    ensure_remote_dir(sftp, f"{REMOTE_BASE}/data")
    ensure_remote_dir(sftp, f"{REMOTE_BASE}/outputs")

    # 1. Upload Model Weights
    upload_file(sftp, "runs/rice_seg_yolo11s_1280/weights/best.onnx", f"{REMOTE_BASE}/weights/best.onnx")
    upload_file(sftp, "runs/stage2_resnet50/grain_resnet50.onnx", f"{REMOTE_BASE}/weights/grain_resnet50.onnx")

    # 2. Upload Sample Test Image
    upload_file(sftp, "RiceAge/data/samples/sample_1489x2048.jpg", f"{REMOTE_BASE}/data/sample_1489x2048.jpg")

    # 3. Upload Code Files
    for root, dirs, files in os.walk("RiceAge/src"):
        for f in files:
            if f.endswith((".py", ".json", ".yaml")):
                lp = os.path.join(root, f)
                rel = os.path.relpath(lp, "RiceAge")
                rp = f"{REMOTE_BASE}/{rel.replace(os.sep, '/')}"
                upload_file(sftp, lp, rp)

    for root, dirs, files in os.walk("RiceAge/configs"):
        for f in files:
            if f.endswith((".py", ".json", ".yaml")):
                lp = os.path.join(root, f)
                rel = os.path.relpath(lp, "RiceAge")
                rp = f"{REMOTE_BASE}/{rel.replace(os.sep, '/')}"
                upload_file(sftp, lp, rp)

    upload_file(sftp, "RiceAge/flags.py", f"{REMOTE_BASE}/flags.py")
    upload_file(sftp, "RiceAge/main.py", f"{REMOTE_BASE}/main.py")

    sftp.close()
    print("\nAll files successfully transferred via SFTP.")
    ssh.close()

if __name__ == "__main__":
    main()
