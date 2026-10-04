# Visualization module for segmentation, instances, measurements, and separation diagnostics.

import os
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2


def generate_instance_colormap(num_classes: int, seed: int = 42) -> np.ndarray:
    # Generate high-contrast random colors for instance visualization.
    np.random.seed(seed)
    # Bright saturated colors
    colors = np.random.randint(60, 255, size=(num_classes + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # background is black
    return colors


def create_segmentation_overlay(
    original_bgr: np.ndarray,
    binary_mask: np.ndarray,
    alpha: float = 0.35
) -> np.ndarray:
    # Create a semi-transparent color overlay of the binary segmentation on original image.
    overlay = original_bgr.copy()
    # Color mask pixels with bright green
    overlay[binary_mask > 0] = [0, 220, 0]
    blended = cv2.addWeighted(overlay, alpha, original_bgr, 1.0 - alpha, 0)
    
    # Draw contour boundaries in solid green
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 0), 1)
    return blended


def create_instances_visualization(
    original_bgr: np.ndarray,
    instance_mask: np.ndarray,
    measurements: List[Dict[str, Any]],
    alpha: float = 0.5
) -> np.ndarray:
    # Generate color-coded instance segmentation map with labeled grain IDs.
    max_id = int(instance_mask.max())
    if max_id == 0:
        return original_bgr.copy()

    colors = generate_instance_colormap(max_id)
    colored_instances = colors[instance_mask]

    # Blend with original
    blended = cv2.addWeighted(colored_instances, alpha, original_bgr, 1.0 - alpha, 0)

    # Draw instance boundaries and label IDs at centroids
    for row in measurements:
        gid = row["grain_id"]
        cx = int(round(row["centroid_x"]))
        cy = int(round(row["centroid_y"]))

        # Draw a small center dot
        cv2.circle(blended, (cx, cy), 2, (255, 255, 255), -1)
        
        # Label text
        label = str(gid)
        # Choose text position slightly offset
        tx, ty = cx + 4, cy - 2
        
        # Text shadow for readability
        cv2.putText(blended, label, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(blended, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

    return blended


def create_measurements_visualization(
    original_bgr: np.ndarray,
    measurements: List[Dict[str, Any]],
    instance_mask: np.ndarray
) -> np.ndarray:
    # Annotate grains with rotated minimum-area bounding boxes, major axes, and metrics.
    vis = original_bgr.copy()

    for row in measurements:
        gid = row["grain_id"]
        cx = int(round(row["centroid_x"]))
        cy = int(round(row["centroid_y"]))
        status = row.get("separation_status", "")
        is_boundary = row.get("is_boundary", False)

        # Select color based on status
        if is_boundary:
            box_color = (255, 255, 0)  # Cyan for boundary truncated
        elif status == "under_segmented_cluster":
            box_color = (0, 0, 255)    # Red for unresolved cluster
        elif status == "separated_from_cluster":
            box_color = (0, 255, 255)  # Yellow for separated
        elif status == "over_segmented_fragment":
            box_color = (200, 100, 255) # Magenta for fragment
        else:
            box_color = (0, 255, 0)    # Green for isolated whole

        # Draw contour of this specific grain
        grain_mask = (instance_mask == gid).astype(np.uint8)
        contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = contours[0]
            # Draw rotated bounding box
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(vis, [box], 0, box_color, 1)

            # Draw major axis line
            length = row["length_px"]
            angle_rad = np.radians(row["orientation_deg"])
            half_len = length / 2.0
            dx = half_len * np.cos(angle_rad)
            dy = half_len * np.sin(angle_rad)
            pt1 = (int(round(cx - dx)), int(round(cy - dy)))
            pt2 = (int(round(cx + dx)), int(round(cy + dy)))
            cv2.line(vis, pt1, pt2, (255, 200, 50), 1, cv2.LINE_AA)

        # Label grain ID
        cv2.putText(vis, str(gid), (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, str(gid), (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw legend in top-left corner
    legend_items = [
        ("Isolated Single Grain", (0, 255, 0)),
        ("Separated from Cluster", (0, 255, 255)),
        ("Unresolved Cluster", (0, 0, 255)),
        ("Boundary Truncated", (255, 255, 0)),
        ("Fragment / Small", (200, 100, 255)),
    ]
    
    cv2.rectangle(vis, (15, 15), (280, 145), (20, 20, 20), -1)
    cv2.rectangle(vis, (15, 15), (280, 145), (100, 100, 100), 1)
    cv2.putText(vis, "Legend / Diagnostic Status", (25, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    
    for idx, (text, col) in enumerate(legend_items):
        ly = 52 + idx * 18
        cv2.rectangle(vis, (25, ly - 8), (40, ly + 2), col, -1)
        cv2.putText(vis, text, (48, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (220, 220, 220), 1, cv2.LINE_AA)

    return vis


def save_visualization(image: np.ndarray, output_path: str) -> None:
    # Save visualization image ensuring parent directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 92])
