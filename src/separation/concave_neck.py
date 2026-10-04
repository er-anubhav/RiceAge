"""Concave neck detection and contour-based splitting for touching rice grain clusters."""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
import cv2


@dataclass(frozen=True)
class ConcavePoint:
    point: Tuple[int, int]
    depth: float
    start: Tuple[int, int]
    end: Tuple[int, int]
    contour_idx: int

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point": self.point,
            "depth": self.depth,
            "start": self.start,
            "end": self.end,
            "contour_idx": self.contour_idx,
        }


@dataclass(frozen=True)
class CutCandidate:
    idx_a: int
    idx_b: int
    pt_a: Tuple[int, int]
    pt_b: Tuple[int, int]
    distance: float
    score: float


def _extract_convexity_defects(
    contour: np.ndarray,
    min_depth_px: float
) -> List[ConcavePoint]:
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return []

    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return []

    if defects is None:
        return []

    candidates = []
    for defect_row in defects:
        s_idx, e_idx, f_idx, raw_depth = defect_row.flatten()
        depth = float(raw_depth / 256.0)

        if depth >= min_depth_px:
            candidates.append(
                ConcavePoint(
                    point=tuple(int(v) for v in contour[f_idx].flatten()),
                    depth=depth,
                    start=tuple(int(v) for v in contour[s_idx].flatten()),
                    end=tuple(int(v) for v in contour[e_idx].flatten()),
                    contour_idx=int(f_idx),
                )
            )

    return sorted(candidates, key=lambda cp: cp.depth, reverse=True)


def _suppress_neighboring_points(
    points: List[ConcavePoint],
    min_distance_px: float
) -> List[ConcavePoint]:
    suppressed: List[ConcavePoint] = []
    for candidate in points:
        cx, cy = candidate.point
        is_isolated = all(
            math.hypot(cx - s.point[0], cy - s.point[1]) >= min_distance_px
            for s in suppressed
        )
        if is_isolated:
            suppressed.append(candidate)
    return suppressed


def detect_concave_points(
    contour: np.ndarray,
    min_defect_depth_px: float = 2.0,
    min_contour_dist_px: int = 5
) -> List[ConcavePoint]:
    """Detect concave pinch points (necks) along the cluster contour using convexity defects."""
    if len(contour) < 5:
        return []

    raw_candidates = _extract_convexity_defects(contour, min_defect_depth_px)
    return _suppress_neighboring_points(raw_candidates, min_contour_dist_px)


def is_cutline_valid(
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    cluster_mask: np.ndarray,
    num_samples: int = 25
) -> bool:
    """Verify that the straight cut line between pt1 and pt2 lies strictly inside foreground."""
    h, w = cluster_mask.shape[:2]
    xs = np.linspace(pt1[0], pt2[0], num_samples)
    ys = np.linspace(pt1[1], pt2[1], num_samples)

    for x, y in zip(xs, ys):
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return False
        if cluster_mask[iy, ix] == 0:
            return False

    return True


def pair_concave_points(
    concave_candidates: List[Any],
    cluster_mask: np.ndarray,
    max_neck_width_px: float = 60.0,
    min_neck_width_px: float = 3.0
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Pair opposing concave pinch points to form candidate cut-lines."""
    n = len(concave_candidates)
    if n < 2:
        return []

    candidates: List[CutCandidate] = []
    for i in range(n):
        for j in range(i + 1, n):
            c_a = concave_candidates[i]
            c_b = concave_candidates[j]
            p1 = c_a["point"] if isinstance(c_a, dict) or hasattr(c_a, "__getitem__") else c_a.point
            p2 = c_b["point"] if isinstance(c_b, dict) or hasattr(c_b, "__getitem__") else c_b.point
            depth_a = c_a["depth"] if isinstance(c_a, dict) or hasattr(c_a, "__getitem__") else c_a.depth
            depth_b = c_b["depth"] if isinstance(c_b, dict) or hasattr(c_b, "__getitem__") else c_b.depth

            dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            if min_neck_width_px <= dist <= max_neck_width_px:
                if is_cutline_valid(p1, p2, cluster_mask):
                    score = (depth_a + depth_b) / (dist + 1.0)
                    candidates.append(CutCandidate(i, j, p1, p2, dist, score))

    candidates.sort(key=lambda c: c.score, reverse=True)

    pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    used_indices = set()
    for cand in candidates:
        if cand.idx_a not in used_indices and cand.idx_b not in used_indices:
            pairs.append((cand.pt_a, cand.pt_b))
            used_indices.add(cand.idx_a)
            used_indices.add(cand.idx_b)

    return pairs


def _relabel_sliced_components(
    sliced_mask: np.ndarray,
    min_piece_area_px: int
) -> Tuple[np.ndarray, int]:
    h, w = sliced_mask.shape[:2]
    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(sliced_mask, connectivity=8)

    valid_labels = [
        lbl for lbl in range(1, num_cc)
        if stats[lbl, cv2.CC_STAT_AREA] >= min_piece_area_px
    ]

    if len(valid_labels) <= 1:
        return (sliced_mask > 0).astype(np.int32), 1

    labeled = np.zeros((h, w), dtype=np.int32)
    for new_idx, old_lbl in enumerate(valid_labels, start=1):
        labeled[labels == old_lbl] = new_idx

    return labeled, len(valid_labels)


def split_cluster_concave_neck(
    cluster_mask: np.ndarray,
    min_defect_depth_px: float = 2.0,
    max_neck_width_px: float = 60.0,
    cut_thickness: int = 2,
    min_piece_area_px: int = 40
) -> Tuple[np.ndarray, int, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Separate touching grains in a binary cluster mask using concave neck cut-lines."""
    h, w = cluster_mask.shape[:2]
    binary_uint8 = (cluster_mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((h, w), dtype=np.int32), 0, []

    largest_contour = max(contours, key=cv2.contourArea)
    concave_pts = detect_concave_points(largest_contour, min_defect_depth_px=min_defect_depth_px)
    pairs = pair_concave_points(concave_pts, binary_uint8, max_neck_width_px=max_neck_width_px)

    if not pairs:
        return (binary_uint8 > 0).astype(np.int32), 1, []

    cut_mask = binary_uint8.copy()
    for p1, p2 in pairs:
        cv2.line(cut_mask, p1, p2, 0, cut_thickness)

    labeled, num_pieces = _relabel_sliced_components(cut_mask, min_piece_area_px)
    return labeled, num_pieces, pairs

