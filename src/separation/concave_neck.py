"""Concave points / neck detection and contour-based splitting for touching rice clusters."""

from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
import cv2


def detect_concave_points(
    contour: np.ndarray,
    min_defect_depth_px: float = 2.0,
    min_contour_dist_px: int = 5
) -> List[Dict[str, Any]]:
    """Detect concave pinch points (necks) along the cluster contour using convexity defects.
    
    Args:
        contour: (N, 1, 2) or (N, 2) contour array.
        min_defect_depth_px: Minimum depth inside convex hull to consider as a pinch point.
        min_contour_dist_px: Spatial suppression distance between neighboring concave points.
        
    Returns:
        List of dicts with 'point': (x, y), 'depth': float, 'start': (x, y), 'end': (x, y).
    """
    if len(contour) < 5:
        return []

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return []

    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except Exception:
        return []

    if defects is None:
        return []

    candidates = []
    # OpenCV 5.0 defects may have shape (K, 4) or (K, 1, 4)
    for d in defects:
        row = d.flatten()
        s_idx, e_idx, f_idx, depth_raw = row
        depth = float(depth_raw / 256.0)

        if depth >= min_defect_depth_px:
            far_pt = tuple(int(v) for v in contour[f_idx].flatten())
            start_pt = tuple(int(v) for v in contour[s_idx].flatten())
            end_pt = tuple(int(v) for v in contour[e_idx].flatten())

            candidates.append({
                "point": far_pt,
                "depth": depth,
                "start": start_pt,
                "end": end_pt,
                "contour_idx": int(f_idx)
            })

    # Sort candidates by depth descending
    candidates.sort(key=lambda x: x["depth"], reverse=True)

    # Non-maximum suppression by Euclidean distance between defect points
    suppressed = []
    for cand in candidates:
        pt = cand["point"]
        too_close = False
        for s in suppressed:
            s_pt = s["point"]
            dist = math.hypot(pt[0] - s_pt[0], pt[1] - s_pt[1])
            if dist < min_contour_dist_px:
                too_close = True
                break
        if not too_close:
            suppressed.append(cand)

    return suppressed


def is_cutline_valid(
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    cluster_mask: np.ndarray,
    num_samples: int = 25
) -> bool:
    """Verify that the straight cut line between pt1 and pt2 lies strictly inside foreground."""
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = cluster_mask.shape[:2]

    # Sample points along the segment
    xs = np.linspace(x1, x2, num_samples)
    ys = np.linspace(y1, y2, num_samples)

    for x, y in zip(xs, ys):
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return False
        # If segment crosses background, line is invalid (cuts outside grain)
        if cluster_mask[iy, ix] == 0:
            return False

    return True


def pair_concave_points(
    concave_candidates: List[Dict[str, Any]],
    cluster_mask: np.ndarray,
    max_neck_width_px: float = 60.0,
    min_neck_width_px: float = 3.0
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Pair opposing concave pinch points to form candidate cut-lines.
    
    Rules for valid pair:
    1. Distance between points is within plausible grain neck bounds.
    2. The straight cut-line segment lies entirely inside the cluster foreground.
    3. Greedily pairs deepest, closest mutually opposing points.
    """
    if len(concave_candidates) < 2:
        return []

    pairs = []
    used_indices = set()

    # Generate all candidate pairs with scoring
    pair_candidates = []
    n = len(concave_candidates)
    for i in range(n):
        for j in range(i + 1, n):
            p1 = concave_candidates[i]["point"]
            p2 = concave_candidates[j]["point"]
            dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])

            if min_neck_width_px <= dist <= max_neck_width_px:
                if is_cutline_valid(p1, p2, cluster_mask):
                    # Score: prioritize high defect depths and short neck distance
                    depth_sum = concave_candidates[i]["depth"] + concave_candidates[j]["depth"]
                    score = depth_sum / (dist + 1.0)
                    pair_candidates.append({
                        "i": i,
                        "j": j,
                        "p1": p1,
                        "p2": p2,
                        "dist": dist,
                        "score": score
                    })

    # Sort pair candidates by score descending
    pair_candidates.sort(key=lambda x: x["score"], reverse=True)

    for cand in pair_candidates:
        i, j = cand["i"], cand["j"]
        if i not in used_indices and j not in used_indices:
            pairs.append((cand["p1"], cand["p2"]))
            used_indices.add(i)
            used_indices.add(j)

    return pairs


def split_cluster_concave_neck(
    cluster_mask: np.ndarray,
    min_defect_depth_px: float = 2.0,
    max_neck_width_px: float = 60.0,
    cut_thickness: int = 2,
    min_piece_area_px: int = 40
) -> Tuple[np.ndarray, int, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Separate touching grains in a binary cluster mask using concave neck cut-lines.
    
    Returns:
        labeled_mask: int32 array where 0=bg, 1..K=separated grains.
        num_pieces: Number of valid grain instances extracted.
        cut_lines: List of applied (p1, p2) line tuples.
    """
    h, w = cluster_mask.shape[:2]
    binary_uint8 = (cluster_mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((h, w), dtype=np.int32), 0, []

    c = max(contours, key=cv2.contourArea)
    concave_pts = detect_concave_points(c, min_defect_depth_px=min_defect_depth_px)
    pairs = pair_concave_points(concave_pts, binary_uint8, max_neck_width_px=max_neck_width_px)

    if not pairs:
        # No valid concave pairs found
        labeled = (binary_uint8 > 0).astype(np.int32)
        return labeled, 1, []

    # Apply cut lines on a copy of the mask
    cut_mask = binary_uint8.copy()
    applied_cuts = []
    for p1, p2 in pairs:
        cv2.line(cut_mask, p1, p2, 0, cut_thickness)
        applied_cuts.append((p1, p2))

    # Connected components on sliced mask
    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(cut_mask, connectivity=8)

    # Filter out tiny slivers created by cut width
    valid_labels = []
    for lbl in range(1, num_cc):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_piece_area_px:
            valid_labels.append(lbl)

    if len(valid_labels) <= 1:
        # Cut did not produce multiple valid pieces: keep original
        labeled = (binary_uint8 > 0).astype(np.int32)
        return labeled, 1, []

    # Remap valid pieces to 1, 2, ... K
    result_labeled = np.zeros((h, w), dtype=np.int32)
    for new_idx, old_lbl in enumerate(valid_labels, start=1):
        result_labeled[labels == old_lbl] = new_idx

    return result_labeled, len(valid_labels), applied_cuts
