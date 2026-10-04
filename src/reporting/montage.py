# Cluster side-by-side comparison and diagnostic montage generator.

import os
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import pandas as pd


def colorize_roi(roi_labels: np.ndarray, base_crop_bgr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    # Colorize a labeled sub-mask onto a base crop.
    h, w = roi_labels.shape[:2]
    vis = base_crop_bgr.copy()
    
    unique_lbls = np.unique(roi_labels)
    unique_lbls = unique_lbls[unique_lbls > 0]
    
    palette = [
        (0, 255, 0),    # Green
        (0, 200, 255),  # Yellow-Orange
        (255, 100, 0),  # Blue
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (100, 255, 100) # Lime
    ]

    for idx, lbl in enumerate(unique_lbls):
        col = palette[idx % len(palette)]
        mask = (roi_labels == lbl)
        vis_pixels = vis[mask].astype(np.float32)
        blended = vis_pixels * (1.0 - alpha) + np.array(col, dtype=np.float32) * alpha
        vis[mask] = blended.astype(np.uint8)
        # Draw contour boundary
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, col, 1)

    return vis


def create_cluster_comparison_panel(
    original_bgr: np.ndarray,
    cluster_roi_info: Dict[str, Any],
    padding: int = 6
) -> np.ndarray:
    # Generate a 4-panel side-by-side comparison for a single cluster. Panels: [ 1. Original RGB Crop | 2. Watershed Result | 3. Concave-Neck Result | 4. Final Hybrid Result ]
    comp_x, comp_y, comp_w, comp_h = cluster_roi_info["bbox"]
    img_h, img_w = original_bgr.shape[:2]

    # Bounded crop with padding
    x1 = max(0, comp_x - padding)
    y1 = max(0, comp_y - padding)
    x2 = min(img_w, comp_x + comp_w + padding)
    y2 = min(img_h, comp_y + comp_h + padding)

    crop_bgr = original_bgr[y1:y2, x1:x2].copy()
    crop_h, crop_w = crop_bgr.shape[:2]

    # Map cluster ROI masks to crop coordinate frame
    off_x = comp_x - x1
    off_y = comp_y - y1

    # Base crop for panels
    panel1 = crop_bgr.copy()
    
    # Panel 2: Watershed
    ws_crop_lbl = np.zeros((crop_h, crop_w), dtype=np.int32)
    ws_roi = cluster_roi_info["watershed_roi"]
    ws_crop_lbl[off_y:off_y + comp_h, off_x:off_x + comp_w] = ws_roi
    panel2 = colorize_roi(ws_crop_lbl, crop_bgr)

    # Panel 3: Concave
    cnc_crop_lbl = np.zeros((crop_h, crop_w), dtype=np.int32)
    cnc_roi = cluster_roi_info["concave_roi"]
    cnc_crop_lbl[off_y:off_y + comp_h, off_x:off_x + comp_w] = cnc_roi
    panel3 = colorize_roi(cnc_crop_lbl, crop_bgr)
    # Draw cutlines on Panel 3
    for p1, p2 in cluster_roi_info.get("cnc_cuts", []):
        pt1_crop = (p1[0] + off_x, p1[1] + off_y)
        pt2_crop = (p2[0] + off_x, p2[1] + off_y)
        cv2.line(panel3, pt1_crop, pt2_crop, (0, 0, 255), 2)
        cv2.circle(panel3, pt1_crop, 2, (0, 255, 255), -1)
        cv2.circle(panel3, pt2_crop, 2, (0, 255, 255), -1)

    # Panel 4: Final Hybrid
    hyb_crop_lbl = np.zeros((crop_h, crop_w), dtype=np.int32)
    hyb_roi = cluster_roi_info["hybrid_roi"]
    hyb_crop_lbl[off_y:off_y + comp_h, off_x:off_x + comp_w] = hyb_roi
    panel4 = colorize_roi(hyb_crop_lbl, crop_bgr)

    # Add header bars to each panel
    panels = [panel1, panel2, panel3, panel4]
    titles = [
        "1. Original",
        f"2. Watershed (N={len(np.unique(ws_roi[ws_roi > 0]))})",
        f"3. Concave-Neck (N={len(np.unique(cnc_roi[cnc_roi > 0]))})",
        f"4. Hybrid Final (N={len(np.unique(hyb_roi[hyb_roi > 0]))})"
    ]

    header_h = 24
    titled_panels = []
    for p, title in zip(panels, titles):
        framed = np.zeros((crop_h + header_h, crop_w, 3), dtype=np.uint8)
        framed[:header_h, :] = (35, 35, 35)
        framed[header_h:, :] = p
        cv2.putText(framed, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.rectangle(framed, (0, 0), (crop_w - 1, crop_h + header_h - 1), (70, 70, 70), 1)
        titled_panels.append(framed)

    # Concatenate horizontally
    combined = np.hstack(titled_panels)
    return combined


def create_unresolved_clusters_montage(
    original_bgr: np.ndarray,
    cluster_rois: Dict[int, Dict[str, Any]],
    benchmark_df: pd.DataFrame,
    output_path: str = "outputs/visualizations/unresolved_clusters_montage.jpg"
) -> str:
    # Create a structured diagnostic montage of all clusters repaired by concave neck.
    repaired_records = benchmark_df[benchmark_df["confidence_status"] == "repaired_by_concave"]
    if repaired_records.empty:
        # Fallback to any clusters if none repaired
        repaired_records = benchmark_df.head(8)

    panels = []
    target_panel_w = 480
    target_panel_h = 100

    for _, row in repaired_records.iterrows():
        cid = int(row["cluster_id"])
        roi_info = cluster_rois.get(cid)
        if not roi_info:
            continue

        panel = create_cluster_comparison_panel(original_bgr, roi_info)
        # Resize to uniform width maintaining aspect ratio
        cur_h, cur_w = panel.shape[:2]
        scaled_w = target_panel_w
        scaled_h = int(round(cur_h * (scaled_w / float(cur_w))))
        panel_resized = cv2.resize(panel, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

        # Add cluster label banner
        banner = np.zeros((26, scaled_w, 3), dtype=np.uint8)
        banner[:, :] = (20, 20, 20)
        status_txt = f"Cluster #{cid} | Area: {row['original_area']}px | Watershed: {row['watershed_count']} -> Concave Cut: {row['concave_split_count']} -> Hybrid: {row['final_count']}"
        cv2.putText(banner, status_txt, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(banner, (0, 0), (scaled_w - 1, 25), (100, 100, 100), 1)

        panel_block = np.vstack([banner, panel_resized])
        panels.append(panel_block)

    if not panels:
        return output_path

    # Tile panels vertically (or in 2 columns if > 6)
    if len(panels) <= 4:
        montage = np.vstack(panels)
    else:
        # Split into 2 columns
        mid = (len(panels) + 1) // 2
        col1 = panels[:mid]
        col2 = panels[mid:]
        
        # Match heights
        max_h1 = sum(p.shape[0] for p in col1)
        max_h2 = sum(p.shape[0] for p in col2)
        total_h = max(max_h1, max_h2)

        def pad_col(col_panels, target_h):
            stack = np.vstack(col_panels)
            if stack.shape[0] < target_h:
                diff = target_h - stack.shape[0]
                pad = np.zeros((diff, stack.shape[1], 3), dtype=np.uint8)
                stack = np.vstack([stack, pad])
            return stack

        c1 = pad_col(col1, total_h)
        c2 = pad_col(col2, total_h)
        montage = np.hstack([c1, c2])

    # Add master title header
    title_banner = np.zeros((45, montage.shape[1], 3), dtype=np.uint8)
    title_banner[:, :] = (15, 15, 15)
    cv2.putText(
        title_banner,
        "EXP-002: DIAGNOSTIC MONTAGE OF UNRESOLVED CLUSTERS REPAIRED BY CONCAVE-NECK CUT-LINES",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    montage_final = np.vstack([title_banner, montage])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, montage_final, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return output_path
