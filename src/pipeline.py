# End-to-end execution pipeline for Rice Grain Quality Analyzer.

import os
import datetime
import logging
import uuid
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import cv2

from configs.default_config import PipelineConfig
from src.feedback_store import FeedbackStore
from src.preprocessing import resolve_calibration, clear_scanner_bezel, preprocess_image
from src.segmentation import segment_otsu, compare_segmentation_methods, cleanup_mask
from src.separation import separate_touching_grains
from src.measurement import measure_all_grains
from src.classification import classify_grains_rule_based
from src.reporting import (
    create_segmentation_overlay,
    create_instances_visualization,
    create_measurements_visualization,
    save_visualization,
    export_crops_and_masks,
    export_measurements_csv,
    export_summary_json,
    generate_experiment_report,
)
from src.quality_classification import MultiHeadQualityClassifier
from src.quality_grading import (
    compute_sample_quality_metrics,
    evaluate_quality_grade,
)
from src.quality_reporting import export_sample_quality_report


class RiceQualityPipeline:
    # Master pipeline orchestrating end-to-end rice grain analysis.

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        weights_path = getattr(self.config.classification, "quality_weights_path", "configs/quality_classifier_weights.json")
        self.quality_classifier = MultiHeadQualityClassifier(weights_path=weights_path)

    def _scale_spatial_parameters(self, img_h: int, img_w: int) -> Dict[str, int]:
        ref_geom = np.sqrt(2048.0 * 1489.0)
        curr_geom = np.sqrt(float(img_h * img_w))
        scale = curr_geom / ref_geom

        return {
            "ksize": max(3, int(round(self.config.preprocessing.gaussian_kernel * scale))) | 1,
            "cluster_area_thresh": max(60, int(round(self.config.separation.cluster_area_threshold_px * (scale ** 2)))),
            "min_peak_dist": max(3, int(round(self.config.separation.min_peak_distance_px * scale))),
            "min_obj_area": max(8, int(round(self.config.segmentation.min_object_area_px * (scale ** 2)))),
            "min_valid_area": max(8, int(round(self.config.measurement.valid_grain_min_area_px * (scale ** 2)))),
        }

    def _calculate_morphology_statistics(self, df: pd.DataFrame, calib) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        if df.empty:
            return stats

        cols = ["length_px", "width_px", "area_px", "aspect_ratio", "circularity", "solidity"]
        if calib.pixel_scale_mm is not None:
            cols.extend(["length_mm", "width_mm", "area_mm2"])

        for col in cols:
            if col in df.columns:
                s = df[col]
                stats[f"{col}_mean"] = round(float(s.mean()), 2)
                stats[f"{col}_std"] = round(float(s.std()), 2)
                stats[f"{col}_median"] = round(float(s.median()), 2)
                stats[f"{col}_min"] = round(float(s.min()), 2)
                stats[f"{col}_max"] = round(float(s.max()), 2)

        return stats

    def _export_pipeline_artifacts(
        self,
        original_bgr: np.ndarray,
        clean_mask: np.ndarray,
        instance_mask: np.ndarray,
        classified_records: list,
        df_classified: pd.DataFrame,
        summary: dict
    ) -> None:
        cfg = self.config.export

        # visualizer.save_visualization: persist rendered image matrix to disk
        save_visualization(original_bgr, os.path.join(cfg.visualizations_dir, "original.jpg"))

        # visualizer.create_segmentation_overlay: generate semi-transparent green mask overlay on original image
        seg_overlay = create_segmentation_overlay(original_bgr, clean_mask)
        # visualizer.save_visualization: persist rendered image matrix to disk
        save_visualization(seg_overlay, os.path.join(cfg.visualizations_dir, "segmentation.jpg"))

        # visualizer.create_instances_visualization: render pseudo-color instance segmentation map with grain IDs
        inst_vis = create_instances_visualization(original_bgr, instance_mask, classified_records)
        # visualizer.save_visualization: persist rendered image matrix to disk
        save_visualization(inst_vis, os.path.join(cfg.visualizations_dir, "instances.jpg"))

        # visualizer.create_measurements_visualization: overlay oriented bounding boxes and measurement tags on grains
        meas_vis = create_measurements_visualization(original_bgr, classified_records, instance_mask)
        # visualizer.save_visualization: persist rendered image matrix to disk
        save_visualization(meas_vis, os.path.join(cfg.visualizations_dir, "measurements.jpg"))

        if cfg.save_crops or cfg.save_masks:
            # exporter.export_crops_and_masks: export individual grain bounding box image crops and binary masks
            export_crops_and_masks(
                original_bgr=original_bgr,
                instance_mask=instance_mask,
                measurements=classified_records,
                crops_dir=cfg.crops_dir,
                masks_dir=cfg.masks_dir,
                max_count=cfg.max_crops_to_save
            )

        # exporter.export_measurements_csv: save all measured grain features to a structured CSV file
        export_measurements_csv(df_classified, os.path.join(cfg.output_dir, "grain_measurements.csv"))
        # exporter.export_summary_json: save aggregated run metrics and counts to a JSON summary file
        export_summary_json(summary, os.path.join(cfg.output_dir, "summary.json"))
        # report_generator.generate_experiment_report: compile summary and statistics into markdown experiment report
        generate_experiment_report(summary, df_classified, os.path.join(cfg.reports_dir, "M0_BASELINE_REPORT.md"))

    def run(
        self,
        image_path: str,
        raw_reference_path: Optional[str] = None,
        generate_outputs: bool = True
    ) -> Dict[str, Any]:
        # Execute complete pipeline on an input image.
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        timestamp = datetime.datetime.now().isoformat()
        scan_id = uuid.uuid4().hex
        original_bgr = cv2.imread(image_path)
        if original_bgr is None:
            raise ValueError(f"Failed to load image via OpenCV: {image_path}")

        img_h, img_w = original_bgr.shape[:2]
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        # calibration.resolve_calibration: resolve effective DPI and mm-per-pixel scaling from metadata/reference
        calib = resolve_calibration(
            image_path,
            raw_reference_path=raw_reference_path,
            assumed_dpi=self.config.calibration.assumed_dpi
        )

        # border_clearing.clear_scanner_bezel: detect and zero out saturated scanner glass bezel margins
        cleaned_bgr, cleared_margins = clear_scanner_bezel(
            original_bgr,
            saturation_thresh=self.config.preprocessing.bezel_intensity_thresh
        )

        spatial_params = self._scale_spatial_parameters(img_h, img_w)

        # denoise.preprocess_image: convert image to grayscale and apply Gaussian filter smoothing
        _, smoothed = preprocess_image(
            cleaned_bgr,
            gaussian_kernel=spatial_params["ksize"],
            gaussian_sigma=self.config.preprocessing.gaussian_sigma
        )

        # binarize.compare_segmentation_methods: benchmark Otsu against adaptive, triangle, Yen, and Li algorithms
        seg_comparison = compare_segmentation_methods(smoothed)
        # binarize.segment_otsu: binarize smoothed grayscale image using Otsu global optimal thresholding
        binary_raw, otsu_val = segment_otsu(smoothed)

        # morphology.cleanup_mask: perform binary hole-filling, morphological opening, and small-noise removal
        clean_mask = cleanup_mask(
            binary_raw,
            kernel_size=self.config.segmentation.morph_kernel_size,
            min_area_px=spatial_params["min_obj_area"],
            fill_holes=self.config.segmentation.fill_holes
        )

        # separation.separate_touching_grains: separate clustered and touching grains via distance-transform watershed
        instance_mask, grain_diagnostics, separation_stats = separate_touching_grains(
            clean_mask,
            min_peak_distance_px=spatial_params["min_peak_dist"],
            peak_threshold_rel=self.config.separation.peak_threshold_rel,
            cluster_area_thresh_px=spatial_params["cluster_area_thresh"],
            solidity_thresh=self.config.separation.solidity_threshold,
            min_valid_area_px=spatial_params["min_valid_area"],
            enable_concave_fallback=getattr(self.config.separation, "enable_concave_fallback", True),
            concave_max_cluster_area_factor=getattr(self.config.separation, "concave_max_cluster_area_factor", 2.5),
            concave_min_cluster_solidity=getattr(self.config.separation, "concave_min_cluster_solidity", 0.72),
            concave_max_cluster_aspect_ratio=getattr(self.config.separation, "concave_max_cluster_aspect_ratio", 3.8),
        )

        # measurement.measure_all_grains: compute morphology dimensions (length, width, area) and RGB/HSV color metrics
        df_measurements, _ = measure_all_grains(
            instance_mask=instance_mask,
            image_rgb=original_rgb,
            grain_diagnostics=grain_diagnostics,
            cleared_margins=cleared_margins,
            calibration=calib,
            min_area_px=spatial_params["min_valid_area"]
        )

        # classification.rule_based.classify_grains_rule_based: assign quality classes (WHOLE/BROKEN/CLUSTER) using Codex/USDA rules
        df_classified = classify_grains_rule_based(
            df_measurements,
            broken_length_ratio_thresh=self.config.classification.broken_length_ratio_thresh,
            broken_aspect_ratio_thresh=self.config.classification.broken_aspect_ratio_thresh
        )
        classified_records = df_classified.to_dict(orient="records")

        # Persist scan-level grain records for the audit workflow. The current M0
        # classifier supplies rule-based labels only; no fake probabilities or
        # defect-head confidence scores are invented here.
        feedback_persisted = False
        if self.config.export.persist_feedback:
            try:
                feedback_db = self.config.export.feedback_db_path or os.path.join(
                    self.config.export.output_dir, "feedback.db"
                )
                grains_for_review = []
                for row in classified_records:
                    grain_number = int(row["grain_id"])
                    bbox = [
                        int(row["x"]), int(row["y"]),
                        int(row["width"]), int(row["height"])
                    ]
                    features = {
                        key: value for key, value in row.items()
                        if key not in {"grain_id", "quality_class", "classification_reason"}
                    }
                    review_flags = [
                        flag for flag, condition in (
                            ("boundary_truncated", bool(row.get("is_boundary", False))),
                            ("unresolved_cluster", row.get("separation_status") == "under_segmented_cluster"),
                        ) if condition
                    ]
                    grains_for_review.append({
                        "id": f"grain_{grain_number:04d}",
                        "predicted_label": str(row["quality_class"]),
                        "bbox": bbox,
                        "center": [row.get("centroid_x"), row.get("centroid_y")],
                        "probabilities": {},
                        "measurements": {
                            key: row.get(key) for key in (
                                "length_px", "width_px", "length_mm", "width_mm",
                                "area_px", "area_mm2", "aspect_ratio", "circularity",
                                "solidity", "orientation_deg", "mean_r", "mean_g",
                                "mean_b", "mean_intensity", "highlight_ratio"
                            ) if key in row
                        },
                        "features": features,
                        "review_flags": review_flags,
                    })
                    grains_for_review[-1]["measurements"]["center"] = [
                        row.get("centroid_x"), row.get("centroid_y")
                    ]
                    grains_for_review[-1]["measurements"]["review_flags"] = review_flags
                store = FeedbackStore(feedback_db)
                store.cache_scan_grains(
                    scan_id=scan_id,
                    grains=grains_for_review,
                    image_path=os.path.abspath(image_path),
                )
                feedback_persisted = True
            except Exception as exc:
                # Feedback persistence is additive: a locked/unavailable database
                # must never bring down the existing inspection/CLI pipeline.
                logging.getLogger(__name__).warning(
                    "Could not persist scan feedback context for %s: %s", scan_id, exc
                )

        quality_counts = df_classified["quality_class"].value_counts().to_dict() if not df_classified.empty else {}
        status_counts = df_classified["separation_status"].value_counts().to_dict() if not df_classified.empty else {}
        morph_stats = self._calculate_morphology_statistics(df_classified, calib)

        # M4 Quality classification & grading
        sample_id = getattr(self.config.classification, "sample_id", None) or os.path.splitext(os.path.basename(image_path))[0]
        grain_classifications, _, _ = self.quality_classifier.classify_detected_objects(
            original_bgr=original_bgr,
            instance_mask=instance_mask,
            image_name=os.path.basename(image_path)
        )
        quality_metrics = compute_sample_quality_metrics(grain_classifications, sample_id=sample_id)
        require_full = getattr(self.config.classification, "require_full_criteria", False)
        grading_decision = evaluate_quality_grade(quality_metrics, require_full_criteria=require_full)

        summary = {
            "timestamp": timestamp,
            "scan_id": scan_id,
            "feedback": {
                "persisted": feedback_persisted,
                "storage": "sqlite" if feedback_persisted else "unavailable_or_disabled",
            },
            "image_path": image_path,
            "image_dimensions": {"width": img_w, "height": img_h},
            "total_grains_detected": len(df_classified),
            "calibration": {
                "is_verified": calib.is_verified,
                "effective_dpi": calib.effective_dpi,
                "pixel_scale_mm": calib.pixel_scale_mm,
                "pixels_per_mm": calib.pixels_per_mm,
                "source_info": calib.source_info,
            },
            "preprocessing": {"cleared_margins": cleared_margins},
            "segmentation": {
                "selected_method": "otsu",
                "selected_threshold_val": otsu_val,
                "method_comparison": {
                    k: {
                        "threshold_value": v["threshold_value"],
                        "foreground_ratio": v["foreground_ratio"],
                        "raw_connected_components": v["raw_connected_components"],
                        "edge_gradient_magnitude": v["edge_gradient_magnitude"],
                    }
                    for k, v in seg_comparison.items()
                }
            },
            "separation": separation_stats,
            "quality_counts": quality_counts,
            "status_counts": status_counts,
            "morphology_statistics": morph_stats,
            "quality_classification": {
                "is_model_loaded": self.quality_classifier.is_loaded,
                "fallback_warning": self.quality_classifier.fallback_warning,
                "total_classified": len(grain_classifications),
            },
            "quality_grading": {
                "assigned_grade": grading_decision.assigned_grade,
                "status": grading_decision.status,
                "standard_name": grading_decision.standard_name,
                "is_compliant": grading_decision.is_compliant,
            },
        }

        report_files = {}
        if generate_outputs:
            self._export_pipeline_artifacts(
                original_bgr=original_bgr,
                clean_mask=clean_mask,
                instance_mask=instance_mask,
                classified_records=classified_records,
                df_classified=df_classified,
                summary=summary
            )
            reports_dir = self.config.export.reports_dir
            os.makedirs(reports_dir, exist_ok=True)
            report_files = export_sample_quality_report(
                sample_id=sample_id,
                metrics=quality_metrics,
                grading=grading_decision,
                grains=grain_classifications,
                output_dir=reports_dir,
                image_path=image_path
            )

        return {
            "summary": summary,
            "scan_id": scan_id,
            "grains_data": grains_for_review if feedback_persisted else [],
            "measurements_df": df_classified,
            "instance_mask": instance_mask,
            "clean_mask": clean_mask,
            "quality_metrics": quality_metrics,
            "grading_decision": grading_decision,
            "grain_classifications": grain_classifications,
            "report_files": report_files,
        }

