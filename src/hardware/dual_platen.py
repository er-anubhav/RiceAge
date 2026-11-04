# Dual-Platen Scanner Controller and Image Processing Architecture
# Eliminates aspect-ratio distortion and manages memory pressure across physical platens.

import os
import time
import subprocess
import shutil
from typing import Dict, Any, Tuple, Optional, List
import cv2
import numpy as np
import pandas as pd

from src.classification import compute_quality_metrics, estimate_grain_mass, FutureDefectTaxonomy


# Hardware Configuration (Defaulting to 300 DPI for safe ARM memory and fast 12s cycle)
DEFAULT_SCAN_DPI = int(os.environ.get("RICEAGE_SCAN_DPI", "300"))
SCANNER_BOTTOM_ID = os.environ.get("SCANNER_BOTTOM_ID", "pixma:04A91912_50C8B2")
SCANNER_TOP_ID = os.environ.get("SCANNER_TOP_ID", "pixma:04A91912_53012D")
SETTLE_DELAY = int(os.environ.get("RICEAGE_SETTLE_DELAY", "5"))


def acquire_hardware_scan(
    output_file: str,
    device_id: str,
    resolution: int = DEFAULT_SCAN_DPI,
    timeout: int = 60
) -> Tuple[bool, Optional[str]]:
    """
    Acquire flatbed optical capture using SANE scanimage.
    Executes sequentially on the target device ID.
    """
    cmd = [
        "scanimage",
        "-d", device_id,
        "--format=jpeg",
        f"--resolution={resolution}",
        "--mode=Color",
        "-o", output_file
    ]
    print(f"[Scanner Hardware] Executing scan: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if proc.returncode == 0 and os.path.exists(output_file):
            sz = os.path.getsize(output_file)
            print(f"[Scanner Hardware] Scan successful -> {output_file} ({sz} bytes)")
            return True, None
        err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
        print(f"[Scanner Hardware] Scan failed (code {proc.returncode}): {err_msg}")
        return False, err_msg
    except Exception as e:
        print(f"[Scanner Hardware] Scan exception: {e}")
        return False, str(e)


def create_distortion_free_composite(
    img_b: np.ndarray,
    img_t: np.ndarray,
    output_path: Optional[str] = None,
    divider_width: int = 8,
    divider_color: Tuple[int, int, int] = (44, 62, 80)
) -> np.ndarray:
    """
    Combines bottom and top scanner platen images horizontally WITHOUT resizing or
    distorting grain aspect ratios. Heights are unified by vertical padding (letterboxing)
    with the background median color, preserving 1:1 metric grain geometry.
    """
    h_b, w_b = img_b.shape[:2]
    h_t, w_t = img_t.shape[:2]
    target_h = max(h_b, h_t)

    # Pad bottom if needed
    if h_b < target_h:
        pad_top = (target_h - h_b) // 2
        pad_bot = target_h - h_b - pad_top
        bg_col = np.median(img_b, axis=(0, 1)).astype(int).tolist()
        img_b_padded = cv2.copyMakeBorder(
            img_b, pad_top, pad_bot, 0, 0,
            cv2.BORDER_CONSTANT, value=bg_col
        )
    else:
        img_b_padded = img_b

    # Pad top if needed
    if h_t < target_h:
        pad_top = (target_h - h_t) // 2
        pad_bot = target_h - h_t - pad_top
        bg_col = np.median(img_t, axis=(0, 1)).astype(int).tolist()
        img_t_padded = cv2.copyMakeBorder(
            img_t, pad_top, pad_bot, 0, 0,
            cv2.BORDER_CONSTANT, value=bg_col
        )
    else:
        img_t_padded = img_t

    # Create solid separation divider bar
    divider = np.full((target_h, divider_width, 3), divider_color, dtype=np.uint8)

    # Concatenate horizontally
    composite = cv2.hconcat([img_b_padded, divider, img_t_padded])

    # Overlay Platen Header Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, target_h / 2400.0)
    thick = max(1, int(round(scale * 2)))

    # Label Bottom Platen (Left)
    cv2.rectangle(composite, (20, 20), (int(360 * scale), int(70 * scale)), (30, 30, 30), -1)
    cv2.putText(composite, "PLATEN 1 (BOTTOM)", (30, int(52 * scale)), font, scale * 0.75, (0, 255, 255), thick, cv2.LINE_AA)

    # Label Top Platen (Right)
    right_x = w_b + divider_width + 20
    cv2.rectangle(composite, (right_x, 20), (right_x + int(360 * scale), int(70 * scale)), (30, 30, 30), -1)
    cv2.putText(composite, "PLATEN 2 (TOP)", (right_x + 10, int(52 * scale)), font, scale * 0.75, (0, 255, 255), thick, cv2.LINE_AA)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, composite)

    return composite


class DualPlatenManager:
    """
    Orchestrates dual flatbed scanner acquisition and independent platen processing.
    Ensures memory efficiency, platen isolation, and unified metric reporting.
    """

    def __init__(self, pipeline=None, scan_resolution: int = DEFAULT_SCAN_DPI):
        self.pipeline = pipeline
        self.scan_resolution = scan_resolution

    def run_dual_analysis(
        self,
        bottom_image_path: Optional[str],
        top_image_path: Optional[str],
        sample_id: str = "Sample-0001",
        commodity: str = "Raw Rice",
        composite_output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes independent platen pipeline runs, merges measurements without cross-platen
        coordinate distortion, and generates consolidated Count & Mass Basis statistics.
        """
        dfs = []
        platens_info = {}
        rendered_images = []

        # 1. Process Bottom Platen
        if bottom_image_path and os.path.exists(bottom_image_path):
            print(f"[DualPlatenManager] Processing Platen 1 (Bottom): {bottom_image_path}")
            res_b = self.pipeline.run(bottom_image_path, generate_outputs=False)
            df_b = res_b["measurements_df"].copy()
            df_b["platen"] = "bottom"
            dfs.append(df_b)
            platens_info["bottom"] = {
                "image_path": bottom_image_path,
                "grains_detected": len(df_b),
                "summary": res_b["summary"]
            }
            img_b = cv2.imread(bottom_image_path)
            rendered_images.append(img_b)
        else:
            img_b = None

        # 2. Process Top Platen
        if top_image_path and os.path.exists(top_image_path):
            print(f"[DualPlatenManager] Processing Platen 2 (Top): {top_image_path}")
            res_t = self.pipeline.run(top_image_path, generate_outputs=False)
            df_t = res_t["measurements_df"].copy()
            df_t["platen"] = "top"
            dfs.append(df_t)
            platens_info["top"] = {
                "image_path": top_image_path,
                "grains_detected": len(df_t),
                "summary": res_t["summary"]
            }
            img_t = cv2.imread(top_image_path)
            rendered_images.append(img_t)
        else:
            img_t = None

        # 3. Handle Composite Rendering
        if img_b is not None and img_t is not None:
            composite_img = create_distortion_free_composite(img_b, img_t, output_path=composite_output_path)
        elif img_b is not None:
            composite_img = img_b
            if composite_output_path:
                cv2.imwrite(composite_output_path, composite_img)
        elif img_t is not None:
            composite_img = img_t
            if composite_output_path:
                cv2.imwrite(composite_output_path, composite_img)
        else:
            raise FileNotFoundError("Neither bottom nor top platen images could be found.")

        # 4. Merge Grain Measurements into Single Unified Dataset
        if dfs:
            df_unified = pd.concat(dfs, ignore_index=True)
            df_unified["grain_id"] = list(range(1, len(df_unified) + 1))
        else:
            df_unified = pd.DataFrame()

        # Calibration pixel scale
        calib_scale = None
        for pinfo in platens_info.values():
            scale_val = pinfo["summary"].get("calibration", {}).get("pixel_scale_mm")
            if scale_val:
                calib_scale = scale_val
                break

        # Compute Consolidated Count and Mass Basis Metrics
        consolidated_metrics = compute_quality_metrics(df_unified, pixel_scale_mm=calib_scale)

        # Build Overall Dashboard and Certificate Payload
        mb = consolidated_metrics["mass_basis"]
        cb = consolidated_metrics["count_basis"]

        # Mean physical dimensions
        avg_len_mm = float(df_unified["length_mm"].mean()) if "length_mm" in df_unified.columns and not df_unified.empty else float(df_unified.get("length_px", pd.Series([0.0])).mean() * (calib_scale or 0.08467))
        avg_brd_mm = float(df_unified["width_mm"].mean()) if "width_mm" in df_unified.columns and not df_unified.empty else float(df_unified.get("width_px", pd.Series([0.0])).mean() * (calib_scale or 0.08467))

        stats = [
            {"param": "Good / Head Rice", "count": cb["whole_count"], "pct": cb["whole_pct"], "mass_g": mb["whole_mass_g"], "mass_pct": mb["whole_pct"], "limit": "Min 75.00%"},
            {"param": "Big Brokens (25-75%)", "count": cb["broken_count"], "pct": cb["broken_pct"], "mass_g": mb["broken_mass_g"], "mass_pct": mb["broken_pct"], "limit": "Max 15.00%"},
            {"param": "Small Brokens (<25%)", "count": cb["small_broken_count"], "pct": cb["small_broken_pct"], "mass_g": mb["small_broken_mass_g"], "mass_pct": mb["small_broken_pct"], "limit": "Max 1.00%"},
            {"param": "Foreign Matter", "count": cb["foreign_matter_count"], "pct": cb["foreign_matter_pct"], "mass_g": mb["foreign_matter_mass_g"], "mass_pct": mb["foreign_matter_pct"], "limit": "Max 0.50%"},
            {"param": "Damaged Grains", "count": cb["damaged_count"], "pct": cb["damaged_pct"], "mass_g": mb["damaged_mass_g"], "mass_pct": mb["damaged_pct"], "limit": "Max 2.00%"},
            {"param": "Discoloured Grains", "count": cb["discoloured_count"], "pct": cb["discoloured_pct"], "mass_g": mb["discoloured_mass_g"], "mass_pct": mb["discoloured_pct"], "limit": "Max 1.00%"},
            {"param": "Chalky Grains", "count": cb["chalky_count"], "pct": cb["chalky_pct"], "mass_g": mb["chalky_mass_g"], "mass_pct": mb["chalky_pct"], "limit": "Max 3.00%"},
        ]

        rows = []
        for s in stats:
            rows.append({
                "name": s["param"],
                "limit": s["limit"],
                "resultPct": f"{s['mass_pct']:.2f}% (mass) / {s['pct']:.2f}% (cnt)",
                "resultCountAndGms": f"{s['count']} grains ({s['mass_g']:.3f} g)",
                "status": "acceptable" if "Min" in s["limit"] and s["mass_pct"] >= float(s["limit"].replace("Min ", "").replace("%", "")) or ("Max" in s["limit"] and s["mass_pct"] <= float(s["limit"].replace("Max ", "").replace("%", ""))) else "substandard"
            })

        return {
            "status": "Complete",
            "commodity": commodity,
            "serial": sample_id,
            "sample_id": sample_id,
            "timestamp": int(time.time()),
            "timestamp_str": time.strftime("%d/%m/%Y %H:%M:%S"),
            "total_objects": cb["total_objects"],
            "total_grains": cb["total_rice_grains"],
            "total_sample_mass_g": consolidated_metrics["total_mass_g"],
            "avg_length_mm": round(avg_len_mm, 2),
            "avg_breadth_mm": round(avg_brd_mm, 2),
            "length_breadth_ratio": round(avg_len_mm / max(avg_brd_mm, 0.01), 2),
            "commercial_grade": consolidated_metrics["commercial_grade"],
            "disposition": "GRADE_1" if "Grade 1" in consolidated_metrics["commercial_grade"] else "SUBSTANDARD",
            "count_basis": cb,
            "mass_basis": mb,
            "stats": stats,
            "rows": rows,
            "platens": platens_info,
            "grain_records": df_unified.to_dict(orient="records"),
            "composite_image": os.path.basename(composite_output_path) if composite_output_path else ""
        }
