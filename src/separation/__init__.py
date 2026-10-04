from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as skimage_watershed
import pandas as pd


class ComponentCategory(str, Enum):
    SINGLE_GRAIN = "single_grain"
    TOUCHING_PAIR = "touching_pair"
    CLUSTER_COMPLEX = "cluster_complex"
    FRAGMENT_OR_NOISE = "fragment_or_noise"


class GrainStatus(str, Enum):
    ISOLATED_SINGLE = "isolated_single"
    SEPARATED_FROM_CLUSTER = "separated_from_cluster"
    UNDER_SEGMENTED_CLUSTER = "under_segmented_cluster"
    OVER_SEGMENTED_FRAGMENT = "over_segmented_fragment"
    BOUNDARY_TRUNCATED = "boundary_truncated"


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
        return {"point": self.point, "depth": self.depth, "start": self.start, "end": self.end, "contour_idx": self.contour_idx}


@dataclass(frozen=True)
class CutCandidate:
    idx_a: int
    idx_b: int
    pt_a: Tuple[int, int]
    pt_b: Tuple[int, int]
    distance: float
    score: float


@dataclass
class ClusterComparison:
    cluster_id: int
    bbox: Tuple[int, int, int, int]
    original_area: int
    solidity: float
    estimated_grain_count: int
    watershed_count: int
    concave_split_count: int
    final_count: int
    status: str
    cuts_found: int
    labeled_concave: np.ndarray
    applied_cuts: List[Tuple[Tuple[int, int], Tuple[int, int]]]


def compute_distance_transform(binary_mask: np.ndarray, distance_type: int = cv2.DIST_L2, mask_size: int = 5) -> np.ndarray:
    return cv2.distanceTransform((binary_mask > 0).astype(np.uint8), distance_type, mask_size)


def detect_distance_peaks(distance_map: np.ndarray, min_distance: int = 7, threshold_rel: float = 0.35, threshold_abs: Optional[float] = None) -> np.ndarray:
    if distance_map.max() == 0:
        return np.empty((0, 2), dtype=int)
    return peak_local_max(distance_map, min_distance=min_distance, threshold_rel=threshold_rel, threshold_abs=threshold_abs, exclude_border=False)


def extract_connected_components(binary_mask: np.ndarray, connectivity: int = 8) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    return cv2.connectedComponentsWithStats(binary_mask, connectivity=connectivity)


def summarize_connected_components(stats: np.ndarray, min_area: int = 50) -> Dict[str, Any]:
    if len(stats) <= 1:
        return {"total_components": 0, "valid_components": 0, "min_area": 0, "max_area": 0, "median_area": 0.0, "mean_area": 0.0}
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_areas = areas[areas >= min_area]
    return {
        "total_components": int(len(areas)),
        "valid_components": int(len(valid_areas)),
        "min_area": int(areas.min()) if len(areas) else 0,
        "max_area": int(areas.max()) if len(areas) else 0,
        "median_area": float(np.median(valid_areas)) if len(valid_areas) else 0.0,
        "mean_area": float(np.mean(valid_areas)) if len(valid_areas) else 0.0,
    }


def evaluate_component_geometry(
    component_mask: np.ndarray,
    solidity_thresh: float = 0.92,
    single_area_range: Tuple[int, int] = (200, 650),
    pair_area_range: Tuple[int, int] = (650, 1200)
) -> Dict[str, Any]:
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area": 0, "solidity": 0.0, "category": ComponentCategory.FRAGMENT_OR_NOISE}

    c = contours[0]
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-5) if hull_area > 0 else 0.0

    (_, (dim1, dim2), _) = cv2.minAreaRect(c)
    major, minor = max(dim1, dim2), min(dim1, dim2)
    aspect_ratio = major / (minor + 1e-5)

    if area < single_area_range[0]:
        category = ComponentCategory.FRAGMENT_OR_NOISE
    elif area <= single_area_range[1] and solidity >= solidity_thresh:
        category = ComponentCategory.SINGLE_GRAIN
    elif area <= pair_area_range[1] or (solidity < solidity_thresh and area <= pair_area_range[1]):
        category = ComponentCategory.TOUCHING_PAIR
    else:
        category = ComponentCategory.CLUSTER_COMPLEX

    return {"area": area, "perimeter": perimeter, "solidity": solidity, "major_axis": major, "minor_axis": minor, "aspect_ratio": aspect_ratio, "category": category}


def detect_concave_points(contour: np.ndarray, min_defect_depth_px: float = 2.0, min_contour_dist_px: int = 5) -> List[ConcavePoint]:
    if len(contour) < 5:
        return []
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return []
    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return []
    if defects is None:
        return []

    pts = []
    for d in defects:
        s_idx, e_idx, f_idx, raw_depth = d.flatten()
        depth = float(raw_depth / 256.0)
        if depth >= min_defect_depth_px:
            pts.append(ConcavePoint(
                point=tuple(int(v) for v in contour[f_idx].flatten()),
                depth=depth,
                start=tuple(int(v) for v in contour[s_idx].flatten()),
                end=tuple(int(v) for v in contour[e_idx].flatten()),
                contour_idx=int(f_idx)
            ))

    pts.sort(key=lambda p: p.depth, reverse=True)
    suppressed: List[ConcavePoint] = []
    for p in pts:
        if all(math.hypot(p.point[0] - s.point[0], p.point[1] - s.point[1]) >= min_contour_dist_px for s in suppressed):
            suppressed.append(p)
    return suppressed


def is_cutline_valid(pt1: Tuple[int, int], pt2: Tuple[int, int], cluster_mask: np.ndarray, num_samples: int = 25) -> bool:
    h, w = cluster_mask.shape[:2]
    for x, y in zip(np.linspace(pt1[0], pt2[0], num_samples), np.linspace(pt1[1], pt2[1], num_samples)):
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or ix >= w or iy < 0 or iy >= h or cluster_mask[iy, ix] == 0:
            return False
    return True


def pair_concave_points(concave_candidates: List[Any], cluster_mask: np.ndarray, max_neck_width_px: float = 60.0, min_neck_width_px: float = 3.0) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    n = len(concave_candidates)
    if n < 2:
        return []

    cands: List[CutCandidate] = []
    for i in range(n):
        for j in range(i + 1, n):
            ca, cb = concave_candidates[i], concave_candidates[j]
            p1 = ca["point"] if isinstance(ca, dict) or hasattr(ca, "__getitem__") else ca.point
            p2 = cb["point"] if isinstance(cb, dict) or hasattr(cb, "__getitem__") else cb.point
            da = ca["depth"] if isinstance(ca, dict) or hasattr(ca, "__getitem__") else ca.depth
            db = cb["depth"] if isinstance(cb, dict) or hasattr(cb, "__getitem__") else cb.depth

            dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            if min_neck_width_px <= dist <= max_neck_width_px and is_cutline_valid(p1, p2, cluster_mask):
                cands.append(CutCandidate(i, j, p1, p2, dist, (da + db) / (dist + 1.0)))

    cands.sort(key=lambda c: c.score, reverse=True)
    pairs, used = [], set()
    for c in cands:
        if c.idx_a not in used and c.idx_b not in used:
            pairs.append((c.pt_a, c.pt_b))
            used.add(c.idx_a)
            used.add(c.idx_b)
    return pairs


def split_cluster_concave_neck(cluster_mask: np.ndarray, min_defect_depth_px: float = 2.0, max_neck_width_px: float = 60.0, cut_thickness: int = 2, min_piece_area_px: int = 40) -> Tuple[np.ndarray, int, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    h, w = cluster_mask.shape[:2]
    binary_uint8 = (cluster_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((h, w), dtype=np.int32), 0, []

    largest_c = max(contours, key=cv2.contourArea)
    concave_pts = detect_concave_points(largest_c, min_defect_depth_px=min_defect_depth_px)
    pairs = pair_concave_points(concave_pts, binary_uint8, max_neck_width_px=max_neck_width_px)
    if not pairs:
        return (binary_uint8 > 0).astype(np.int32), 1, []

    cut_mask = binary_uint8.copy()
    for p1, p2 in pairs:
        cv2.line(cut_mask, p1, p2, 0, cut_thickness)

    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(cut_mask, connectivity=8)
    valid_labels = [lbl for lbl in range(1, num_cc) if stats[lbl, cv2.CC_STAT_AREA] >= min_piece_area_px]
    if len(valid_labels) <= 1:
        return (binary_uint8 > 0).astype(np.int32), 1, []

    labeled = np.zeros((h, w), dtype=np.int32)
    for new_idx, old_lbl in enumerate(valid_labels, start=1):
        labeled[labels == old_lbl] = new_idx
    return labeled, len(valid_labels), pairs


def _segment_cluster_watershed(roi_mask: np.ndarray, min_peak_distance_px: int, peak_threshold_rel: float) -> Optional[np.ndarray]:
    dist_local = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    peaks = peak_local_max(dist_local, min_distance=min_peak_distance_px, threshold_abs=max(float(dist_local.max()) * peak_threshold_rel, 3.0), exclude_border=False)
    if len(peaks) <= 1:
        return None
    markers = np.zeros_like(roi_mask, dtype=np.int32)
    for seed_idx, (r, c) in enumerate(peaks, start=1):
        markers[r, c] = seed_idx
    return skimage_watershed(-dist_local, ndi.maximum_filter(markers, size=3), mask=roi_mask.astype(bool))


def separate_touching_grains(
    binary_mask: np.ndarray,
    min_peak_distance_px: int = 8,
    peak_threshold_rel: float = 0.38,
    cluster_area_thresh_px: int = 650,
    solidity_thresh: float = 0.92,
    min_valid_area_px: int = 40,
    min_single_area_px: Optional[int] = None
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]], Dict[str, Any]]:
    h, w = binary_mask.shape[:2]
    instance_mask = np.zeros((h, w), dtype=np.int32)
    num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8), connectivity=8)

    current_grain_id = 1
    grain_diags: Dict[int, Dict[str, Any]] = {}
    clusters_proc, clusters_split, clusters_unres, isolated_cnt, cluster_grains, fragments_cnt = 0, 0, 0, 0, 0, 0
    min_single = min_single_area_px if min_single_area_px is not None else int(round(cluster_area_thresh_px * (180.0 / 650.0)))

    for comp_idx in range(1, num_cc):
        comp_area = cc_stats[comp_idx, cv2.CC_STAT_AREA]
        if comp_area < min_valid_area_px:
            continue
        cx, cy, cw, ch = cc_stats[comp_idx, cv2.CC_STAT_LEFT], cc_stats[comp_idx, cv2.CC_STAT_TOP], cc_stats[comp_idx, cv2.CC_STAT_WIDTH], cc_stats[comp_idx, cv2.CC_STAT_HEIGHT]
        roi_slice = (slice(cy, cy + ch), slice(cx, cx + cw))
        roi_mask = (cc_labels[roi_slice] == comp_idx).astype(np.uint8)

        geom = evaluate_component_geometry(roi_mask, solidity_thresh=solidity_thresh, single_area_range=(min_single, cluster_area_thresh_px), pair_area_range=(cluster_area_thresh_px, cluster_area_thresh_px * 2))
        category = geom["category"]

        if category == ComponentCategory.SINGLE_GRAIN:
            instance_mask[roi_slice][roi_mask > 0] = current_grain_id
            grain_diags[current_grain_id] = {"status": GrainStatus.ISOLATED_SINGLE.value, "parent_component_id": comp_idx, "solidity": geom["solidity"], "component_area": comp_area, "was_split": False}
            isolated_cnt += 1
            current_grain_id += 1
        elif category == ComponentCategory.FRAGMENT_OR_NOISE:
            instance_mask[roi_slice][roi_mask > 0] = current_grain_id
            grain_diags[current_grain_id] = {"status": GrainStatus.OVER_SEGMENTED_FRAGMENT.value, "parent_component_id": comp_idx, "solidity": geom["solidity"], "component_area": comp_area, "was_split": False}
            fragments_cnt += 1
            current_grain_id += 1
        else:
            clusters_proc += 1
            local_ws = _segment_cluster_watershed(roi_mask, min_peak_distance_px, peak_threshold_rel)
            if local_ws is not None:
                sub_regions = [(lbl, int(np.sum(local_ws == lbl))) for lbl in np.unique(local_ws) if lbl > 0 and np.sum(local_ws == lbl) >= min_valid_area_px]
                if len(sub_regions) > 1:
                    for lbl, sub_area in sub_regions:
                        instance_mask[roi_slice][local_ws == lbl] = current_grain_id
                        grain_diags[current_grain_id] = {"status": GrainStatus.SEPARATED_FROM_CLUSTER.value, "parent_component_id": comp_idx, "solidity": geom["solidity"], "component_area": comp_area, "sub_area": sub_area, "was_split": True}
                        current_grain_id += 1
                    clusters_split += 1
                    cluster_grains += len(sub_regions)
                    continue

            clusters_unres += 1
            instance_mask[roi_slice][roi_mask > 0] = current_grain_id
            grain_diags[current_grain_id] = {"status": GrainStatus.UNDER_SEGMENTED_CLUSTER.value, "parent_component_id": comp_idx, "solidity": geom["solidity"], "component_area": comp_area, "was_split": False}
            current_grain_id += 1

    summary = {
        "raw_connected_components": num_cc - 1,
        "total_detected_grains": current_grain_id - 1,
        "isolated_single_grains": isolated_cnt,
        "clusters_detected": clusters_proc,
        "clusters_successfully_split": clusters_split,
        "clusters_unresolved": clusters_unres,
        "grains_derived_from_clusters": cluster_grains,
        "fragments_count": fragments_cnt,
    }
    return instance_mask, grain_diags, summary


def benchmark_and_separate_hybrid(
    binary_mask: np.ndarray,
    min_peak_distance_px: int = 8,
    peak_threshold_rel: float = 0.38,
    cluster_area_thresh_px: int = 650,
    solidity_thresh: float = 0.92,
    min_valid_area_px: int = 40,
    median_grain_area_px: float = 380.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, Dict[str, Any], Dict[int, Dict[str, Any]]]:
    h, w = binary_mask.shape[:2]
    mask_ws, _, _ = separate_touching_grains(binary_mask, min_peak_distance_px=min_peak_distance_px, peak_threshold_rel=peak_threshold_rel, cluster_area_thresh_px=cluster_area_thresh_px, solidity_thresh=solidity_thresh, min_valid_area_px=min_valid_area_px)
    num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8), connectivity=8)

    mask_cnc = np.zeros((h, w), dtype=np.int32)
    mask_hyb = mask_ws.copy()
    cnc_id, hyb_id = 1, int(mask_ws.max()) + 1
    records, rois = [], {}
    res_a, res_b, res_c, total_clusters, repaired_cnt, unres_cnt = 0, 0, 0, 0, 0, 0
    min_single = int(round(cluster_area_thresh_px * (180.0 / 650.0)))

    for comp_idx in range(1, num_cc):
        comp_area = cc_stats[comp_idx, cv2.CC_STAT_AREA]
        if comp_area < min_valid_area_px:
            continue
        cx, cy, cw, ch = cc_stats[comp_idx, cv2.CC_STAT_LEFT], cc_stats[comp_idx, cv2.CC_STAT_TOP], cc_stats[comp_idx, cv2.CC_STAT_WIDTH], cc_stats[comp_idx, cv2.CC_STAT_HEIGHT]
        roi_slice = (slice(cy, cy + ch), slice(cx, cx + cw))
        roi_mask = (cc_labels[roi_slice] == comp_idx).astype(np.uint8) * 255

        geom = evaluate_component_geometry(roi_mask, solidity_thresh=solidity_thresh, single_area_range=(min_single, cluster_area_thresh_px), pair_area_range=(cluster_area_thresh_px, cluster_area_thresh_px * 2))
        if geom["category"] in [ComponentCategory.TOUCHING_PAIR, ComponentCategory.CLUSTER_COMPLEX]:
            total_clusters += 1
            ws_uniq = np.unique(mask_ws[roi_slice][roi_mask > 0])
            ws_count = len(ws_uniq[ws_uniq > 0])
            labeled_cnc, cnc_count, cnc_cuts = split_cluster_concave_neck(roi_mask, min_defect_depth_px=2.0, max_neck_width_px=60.0, min_piece_area_px=min_valid_area_px)

            if ws_count >= 2:
                res_a += 1
            if cnc_count >= 2:
                res_b += 1
                for lbl in range(1, cnc_count + 1):
                    mask_cnc[roi_slice][labeled_cnc == lbl] = cnc_id
                    cnc_id += 1
            else:
                mask_cnc[roi_slice][roi_mask > 0] = cnc_id
                cnc_id += 1

            if ws_count < 2 and cnc_count >= 2:
                repaired_cnt += 1
                res_c += 1
                status, final_cnt = "repaired_by_concave", cnc_count
                mask_hyb[roi_slice][roi_mask > 0] = 0
                for lbl in range(1, cnc_count + 1):
                    mask_hyb[roi_slice][labeled_cnc == lbl] = hyb_id
                    hyb_id += 1
            elif ws_count >= 2:
                res_c += 1
                status, final_cnt = "split_by_watershed", ws_count
            else:
                unres_cnt += 1
                status, final_cnt = "unresolved_both", 1

            records.append({
                "cluster_id": comp_idx, "bbox_x": cx, "bbox_y": cy, "bbox_w": cw, "bbox_h": ch,
                "original_area": comp_area, "solidity": round(geom["solidity"], 3),
                "estimated_grain_count": max(2, int(round(comp_area / median_grain_area_px))),
                "watershed_count": ws_count, "concave_split_count": cnc_count,
                "final_count": final_cnt, "confidence_status": status, "cuts_found": len(cnc_cuts)
            })
            rois[comp_idx] = {
                "bbox": (cx, cy, cw, ch), "original_mask": roi_mask, "watershed_roi": mask_ws[roi_slice].copy(),
                "concave_roi": labeled_cnc, "hybrid_roi": mask_hyb[roi_slice].copy(), "cnc_cuts": cnc_cuts, "status": status
            }
        else:
            mask_cnc[roi_slice][roi_mask > 0] = cnc_id
            cnc_id += 1

    bench_df = pd.DataFrame(records)
    summary = {
        "total_clusters_evaluated": total_clusters,
        "method_a_watershed_resolved": res_a,
        "method_a_resolution_rate": round(res_a / max(1, total_clusters) * 100, 1),
        "method_b_concave_resolved": res_b,
        "method_b_resolution_rate": round(res_b / max(1, total_clusters) * 100, 1),
        "method_c_hybrid_resolved": res_c,
        "method_c_resolution_rate": round(res_c / max(1, total_clusters) * 100, 1),
        "clusters_repaired_by_concave": repaired_cnt,
        "remaining_unresolved_clusters": unres_cnt,
    }
    return mask_ws, mask_cnc, mask_hyb, bench_df, summary, rois
