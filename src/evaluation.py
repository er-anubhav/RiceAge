# Instance matching, count error, segmentation metrics, and quality evaluation.

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import os
import json
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import linear_sum_assignment

from src.dataset import AnnotatedGrain, DatasetImageRecord, rasterize_instance_mask


@dataclass
class InstanceMatch:
    pred_id: int
    gt_id: str
    gt_condition: str
    pred_quality_class: str
    iou: float
    is_correct_class: Optional[bool]
    is_class_supported: bool


@dataclass
class ImageEvaluationMetrics:
    image_id: int
    filename: str
    category: str
    gt_count: int
    pred_count: int
    abs_count_error: int
    rel_count_error: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    under_segmented_merges: int
    over_segmented_splits: int
    matches: List[InstanceMatch] = field(default_factory=list)
    unmatched_gt: List[str] = field(default_factory=list)
    unmatched_pred: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "category": self.category,
            "gt_count": self.gt_count,
            "pred_count": self.pred_count,
            "abs_count_error": self.abs_count_error,
            "rel_count_error": round(self.rel_count_error, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mean_matched_iou": round(self.mean_matched_iou, 4),
            "under_segmented_merges": self.under_segmented_merges,
            "over_segmented_splits": self.over_segmented_splits,
        }


@dataclass
class DatasetEvaluationReport:
    total_images_evaluated: int
    overall_gt_count: int
    overall_pred_count: int
    mean_abs_count_error: float
    mean_rel_count_error: float
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_iou: float
    per_image_metrics: List[ImageEvaluationMetrics] = field(default_factory=list)
    category_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    classification_breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "total_images_evaluated": self.total_images_evaluated,
            "overall_gt_count": self.overall_gt_count,
            "overall_pred_count": self.overall_pred_count,
            "mean_abs_count_error": round(self.mean_abs_count_error, 2),
            "mean_rel_count_error": round(self.mean_rel_count_error, 4),
            "mean_precision": round(self.mean_precision, 4),
            "mean_recall": round(self.mean_recall, 4),
            "mean_f1": round(self.mean_f1, 4),
            "mean_iou": round(self.mean_iou, 4),
            "category_breakdown": self.category_breakdown,
            "classification_breakdown": self.classification_breakdown,
        }


def compute_pairwise_overlap_matrices(
    pred_instance_mask: np.ndarray,
    gt_instance_mask: np.ndarray,
    pred_ids: List[int],
    gt_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    # Compute intersection-over-union and containment matrices between predicted and ground-truth instances.
    num_p = len(pred_ids)
    num_g = len(gt_ids)
    if num_p == 0 or num_g == 0:
        return np.zeros((num_p, num_g), dtype=np.float32), np.zeros((num_p, num_g), dtype=np.float32)

    pred_areas = {pid: int(np.sum(pred_instance_mask == pid)) for pid in pred_ids}
    gt_areas = {gid: int(np.sum(gt_instance_mask == gid)) for gid in gt_ids}

    iou_mat = np.zeros((num_p, num_g), dtype=np.float32)
    containment_mat = np.zeros((num_p, num_g), dtype=np.float32)

    pred_slices = {}
    for pid in pred_ids:
        ys, xs = np.where(pred_instance_mask == pid)
        if len(ys) > 0:
            pred_slices[pid] = (ys.min(), ys.max(), xs.min(), xs.max())

    gt_slices = {}
    for gid in gt_ids:
        ys, xs = np.where(gt_instance_mask == gid)
        if len(ys) > 0:
            gt_slices[gid] = (ys.min(), ys.max(), xs.min(), xs.max())

    for i, pid in enumerate(pred_ids):
        p_box = pred_slices.get(pid)
        if not p_box:
            continue
        p_area = pred_areas[pid]
        p_ymin, p_ymax, p_xmin, p_xmax = p_box

        for j, gid in enumerate(gt_ids):
            g_box = gt_slices.get(gid)
            if not g_box:
                continue
            g_ymin, g_ymax, g_xmin, g_xmax = g_box

            if p_xmin > g_xmax or p_xmax < g_xmin or p_ymin > g_ymax or p_ymax < g_ymin:
                continue

            iy_min = max(p_ymin, g_ymin)
            iy_max = min(p_ymax, g_ymax)
            ix_min = max(p_xmin, g_xmin)
            ix_max = min(p_xmax, g_xmax)

            p_roi = pred_instance_mask[iy_min : iy_max + 1, ix_min : ix_max + 1]
            g_roi = gt_instance_mask[iy_min : iy_max + 1, ix_min : ix_max + 1]

            intersection = int(np.sum((p_roi == pid) & (g_roi == gid)))
            if intersection > 0:
                union = p_area + gt_areas[gid] - intersection
                iou_mat[i, j] = float(intersection) / float(max(1, union))
                containment_mat[i, j] = float(intersection) / float(max(1, p_area))

    return iou_mat, containment_mat


def evaluate_image_instances(
    pred_instance_mask: np.ndarray,
    pred_classified_df: pd.DataFrame,
    gt_record: DatasetImageRecord,
    iou_threshold: float = 0.50,
    matching_mode: str = "containment",
    containment_threshold: float = 0.50
) -> ImageEvaluationMetrics:
    # Match predicted instances against ground truth polygons and calculate comprehensive metrics.
    gt_grains = gt_record.grains
    gt_count = len(gt_grains)

    pred_ids = [int(p) for p in np.unique(pred_instance_mask) if p > 0]
    pred_count = len(pred_ids)

    pred_class_map = {}
    if not pred_classified_df.empty and "grain_id" in pred_classified_df.columns:
        for _, row in pred_classified_df.iterrows():
            pred_class_map[int(row["grain_id"])] = str(row.get("quality_class", "UNKNOWN"))

    if gt_count == 0 and pred_count == 0:
        return ImageEvaluationMetrics(
            image_id=gt_record.image_id,
            filename=gt_record.filename,
            category=gt_record.category,
            gt_count=0, pred_count=0, abs_count_error=0, rel_count_error=0.0,
            true_positives=0, false_positives=0, false_negatives=0,
            precision=1.0, recall=1.0, f1=1.0, mean_matched_iou=1.0,
            under_segmented_merges=0, over_segmented_splits=0
        )

    gt_mask = rasterize_instance_mask(gt_grains, pred_instance_mask.shape)
    gt_idx_to_grain = {idx: g for idx, g in enumerate(gt_grains, start=1)}
    gt_indices = list(range(1, gt_count + 1))

    iou_mat, containment_mat = compute_pairwise_overlap_matrices(pred_instance_mask, gt_mask, pred_ids, gt_indices)

    # Cost matrix selection based on matching mode
    if matching_mode == "containment":
        cost_matrix = 1.0 - containment_mat
    else:
        cost_matrix = 1.0 - iou_mat

    p_indices, g_indices = linear_sum_assignment(cost_matrix)

    matches: List[InstanceMatch] = []
    matched_pred = set()
    matched_gt = set()
    matched_ious = []

    for pi, gi in zip(p_indices, g_indices):
        iou_val = float(iou_mat[pi, gi])
        cont_val = float(containment_mat[pi, gi])
        
        is_matched = (cont_val >= containment_threshold) if matching_mode == "containment" else (iou_val >= iou_threshold)

        if is_matched:
            pid = pred_ids[pi]
            gt_grain = gt_idx_to_grain[gt_indices[gi]]
            p_class = pred_class_map.get(pid, "UNKNOWN")
            
            # Classification compatibility check
            gt_cond = gt_grain.condition.lower()
            is_supported = gt_cond in ["whole", "broken"]
            is_correct = None
            if is_supported:
                if gt_cond == "whole" and p_class == "WHOLE_RICE":
                    is_correct = True
                elif gt_cond == "broken" and p_class == "BROKEN_RICE":
                    is_correct = True
                else:
                    is_correct = False

            matches.append(InstanceMatch(
                pred_id=pid,
                gt_id=gt_grain.grain_id,
                gt_condition=gt_grain.condition,
                pred_quality_class=p_class,
                iou=round(iou_val, 4),
                is_correct_class=is_correct,
                is_class_supported=is_supported
            ))
            matched_pred.add(pid)
            matched_gt.add(gt_grain.grain_id)
            matched_ious.append(iou_val)

    tp = len(matches)
    fp = pred_count - tp
    fn = gt_count - tp

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_val = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    # Detect merges (under-segmentation: 1 pred overlapping >=2 GT grains with IoU > 0.15)
    merges = 0
    for i, pid in enumerate(pred_ids):
        if np.sum(iou_mat[i] > 0.15) >= 2:
            merges += 1

    # Detect splits (over-segmentation: 1 GT grain overlapping >=2 preds with IoU > 0.15)
    splits = 0
    for j, gid in enumerate(gt_indices):
        if np.sum(iou_mat[:, j] > 0.15) >= 2:
            splits += 1

    unmatched_gt_ids = [g.grain_id for g in gt_grains if g.grain_id not in matched_gt]
    unmatched_pred_ids = [pid for pid in pred_ids if pid not in matched_pred]

    abs_err = abs(pred_count - gt_count)
    rel_err = float(abs_err / max(1, gt_count))

    return ImageEvaluationMetrics(
        image_id=gt_record.image_id,
        filename=gt_record.filename,
        category=gt_record.category,
        gt_count=gt_count,
        pred_count=pred_count,
        abs_count_error=abs_err,
        rel_count_error=rel_err,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=prec,
        recall=rec,
        f1=f1_val,
        mean_matched_iou=mean_iou,
        under_segmented_merges=merges,
        over_segmented_splits=splits,
        matches=matches,
        unmatched_gt=unmatched_gt_ids,
        unmatched_pred=unmatched_pred_ids
    )


def evaluate_catalog(
    pipeline,
    catalog,
    image_ids: Optional[List[int]] = None,
    iou_threshold: float = 0.50,
    matching_mode: str = "containment",
    containment_threshold: float = 0.50,
    save_visualizations: bool = False,
    vis_output_dir: Optional[str] = None
) -> DatasetEvaluationReport:
    # Run evaluation across images in DatasetCatalog and aggregate metrics.
    records = catalog.images if image_ids is None else [catalog.get_by_id(i) for i in image_ids if catalog.get_by_id(i)]
    image_metrics: List[ImageEvaluationMetrics] = []

    for gt_img in records:
        res = pipeline.run(gt_img.image_path, generate_outputs=False)
        pred_mask = res["instance_mask"]
        pred_df = res["measurements_df"]

        metrics = evaluate_image_instances(
            pred_instance_mask=pred_mask,
            pred_classified_df=pred_df,
            gt_record=gt_img,
            iou_threshold=iou_threshold,
            matching_mode=matching_mode,
            containment_threshold=containment_threshold
        )
        image_metrics.append(metrics)

        if save_visualizations and vis_output_dir:
            os.makedirs(vis_output_dir, exist_ok=True)
            vis_path = os.path.join(vis_output_dir, f"eval_{gt_img.filename.replace(' ', '_')}.jpg")
            orig_bgr = cv2.imread(gt_img.image_path)
            create_evaluation_visualization(orig_bgr, pred_mask, gt_img.grains, metrics, vis_path)

    total_imgs = len(image_metrics)
    if total_imgs == 0:
        return DatasetEvaluationReport(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    all_gt = sum(m.gt_count for m in image_metrics)
    all_pred = sum(m.pred_count for m in image_metrics)
    mean_abs_err = float(np.mean([m.abs_count_error for m in image_metrics]))
    mean_rel_err = float(np.mean([m.rel_count_error for m in image_metrics]))
    mean_prec = float(np.mean([m.precision for m in image_metrics]))
    mean_rec = float(np.mean([m.recall for m in image_metrics]))
    mean_f1 = float(np.mean([m.f1 for m in image_metrics]))
    mean_iou = float(np.mean([m.mean_matched_iou for m in image_metrics if m.mean_matched_iou > 0] or [0.0]))

    # Category breakdown
    cats = sorted(list({m.category for m in image_metrics}))
    cat_breakdown: Dict[str, Dict[str, float]] = {}
    for cat in cats:
        cat_ms = [m for m in image_metrics if m.category == cat]
        cat_breakdown[cat] = {
            "images": len(cat_ms),
            "gt_count": sum(m.gt_count for m in cat_ms),
            "pred_count": sum(m.pred_count for m in cat_ms),
            "abs_error": round(float(np.mean([m.abs_count_error for m in cat_ms])), 2),
            "precision": round(float(np.mean([m.precision for m in cat_ms])), 4),
            "recall": round(float(np.mean([m.recall for m in cat_ms])), 4),
            "f1": round(float(np.mean([m.f1 for m in cat_ms])), 4),
            "mean_iou": round(float(np.mean([m.mean_matched_iou for m in cat_ms if m.mean_matched_iou > 0] or [0.0])), 4),
        }

    # Classification breakdown on matched instances
    class_supported_total = 0
    class_correct_total = 0
    per_condition: Dict[str, Dict[str, int]] = {}

    for m in image_metrics:
        for match in m.matches:
            cond = match.gt_condition
            per_condition.setdefault(cond, {"total": 0, "predicted_whole": 0, "predicted_broken": 0, "other": 0})
            per_condition[cond]["total"] += 1
            if match.pred_quality_class == "WHOLE_RICE":
                per_condition[cond]["predicted_whole"] += 1
            elif match.pred_quality_class == "BROKEN_RICE":
                per_condition[cond]["predicted_broken"] += 1
            else:
                per_condition[cond]["other"] += 1

            if match.is_class_supported:
                class_supported_total += 1
                if match.is_correct_class:
                    class_correct_total += 1

    cls_acc = float(class_correct_total / max(1, class_supported_total)) if class_supported_total > 0 else 0.0

    classification_summary = {
        "supported_classes_evaluated": ["Whole", "Broken"],
        "unsupported_classes_present": ["Chalky", "Damaged", "Foreign Matter", "Discoloured"],
        "supported_evaluated_instances": class_supported_total,
        "supported_correct_predictions": class_correct_total,
        "supported_classification_accuracy": round(cls_acc, 4),
        "condition_prediction_matrix": per_condition,
    }

    return DatasetEvaluationReport(
        total_images_evaluated=total_imgs,
        overall_gt_count=all_gt,
        overall_pred_count=all_pred,
        mean_abs_count_error=mean_abs_err,
        mean_rel_count_error=mean_rel_err,
        mean_precision=mean_prec,
        mean_recall=mean_rec,
        mean_f1=mean_f1,
        mean_iou=mean_iou,
        per_image_metrics=image_metrics,
        category_breakdown=cat_breakdown,
        classification_breakdown=classification_summary
    )


def create_evaluation_visualization(
    original_bgr: np.ndarray,
    pred_instance_mask: np.ndarray,
    gt_grains: List[AnnotatedGrain],
    metrics: ImageEvaluationMetrics,
    output_path: str
) -> None:
    # Render diagnostic overlay distinguishing True Positives, False Positives, and False Negatives.
    vis = original_bgr.copy()
    matched_preds = {m.pred_id: m for m in metrics.matches}
    matched_gt_ids = {m.gt_id for m in metrics.matches}

    # 1. Draw GT polygons: Green if matched (TP), Blue/Cyan if missed (FN)
    for g in gt_grains:
        pts = np.array(g.points, dtype=np.int32).reshape((-1, 1, 2))
        color = (0, 220, 0) if g.grain_id in matched_gt_ids else (255, 120, 0)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        cx = int(np.mean([p[0] for p in g.points]))
        cy = int(np.mean([p[1] for p in g.points]))
        label = f"{g.condition[:3]}"
        cv2.putText(vis, label, (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # 2. Draw Pred instance contours: Green if matched, Red if False Positive
    pred_ids = [int(p) for p in np.unique(pred_instance_mask) if p > 0]
    for pid in pred_ids:
        p_mask = (pred_instance_mask == pid).astype(np.uint8)
        cnts, _ = cv2.findContours(p_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (0, 255, 100) if pid in matched_preds else (0, 0, 255)
        cv2.drawContours(vis, cnts, -1, color, 1)

    # Header scoreboard
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (20, 20), (520, 110), (20, 20, 20), -1)
    cv2.rectangle(vis, (20, 20), (520, 110), (100, 100, 100), 1)
    title = f"{metrics.filename} ({metrics.category}) | GT: {metrics.gt_count} | Pred: {metrics.pred_count}"
    stats_line = f"Prec: {metrics.precision:.2f} | Rec: {metrics.recall:.2f} | F1: {metrics.f1:.2f} | IoU: {metrics.mean_matched_iou:.2f}"
    legend = "Green: Matched TP | Red: Pred FP | Orange: Missed GT"
    cv2.putText(vis, title, (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, stats_line, (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, legend, (30, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
