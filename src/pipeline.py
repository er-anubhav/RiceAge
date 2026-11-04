# End-to-end execution pipeline for Rice Grain Quality Analyzer.

import os
import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import cv2

from configs.default_config import PipelineConfig
from src.preprocessing import resolve_calibration, clear_scanner_bezel, preprocess_image
from src.segmentation import segment_otsu, segment_yolo, compare_segmentation_methods, cleanup_mask
from src.separation import separate_touching_grains
from src.measurement import measure_all_grains
from src.classification import classify_grains_rule_based, classify_grains_ai, compute_quality_metrics
from src.reporting import (
    create_segmentation_overlay,
    create_instances_visualization,
    create_measurements_visualization,
    create_defects_visualization,
    save_visualization,
    export_crops_and_masks,
    export_measurements_csv,
    export_summary_json,
    generate_experiment_report,
)


class RiceQualityPipeline:
    # Master pipeline orchestrating end-to-end rice grain analysis.

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

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

        # visualizer.save_visualization: persist original rendered image matrix to disk
        save_visualization(original_bgr, os.path.join(cfg.visualizations_dir, "original.jpg"))

        # visualizer.create_segmentation_overlay: generate semi-transparent mask overlay on original image
        seg_overlay = create_segmentation_overlay(original_bgr, clean_mask)
        save_visualization(seg_overlay, os.path.join(cfg.visualizations_dir, "segmentation.jpg"))

        # visualizer.create_instances_visualization: render pseudo-color instance segmentation map with grain IDs
        inst_vis = create_instances_visualization(original_bgr, instance_mask, classified_records)
        save_visualization(inst_vis, os.path.join(cfg.visualizations_dir, "instances.jpg"))

        # visualizer.create_measurements_visualization: overlay oriented bounding boxes and measurement tags on grains
        meas_vis = create_measurements_visualization(original_bgr, classified_records, instance_mask)
        save_visualization(meas_vis, os.path.join(cfg.visualizations_dir, "measurements.jpg"))

        # visualizer.create_defects_visualization: render color-coded defect overlay (Whole, Broken, Damaged, Chalky, etc.)
        defect_vis = create_defects_visualization(original_bgr, classified_records, instance_mask)
        save_visualization(defect_vis, os.path.join(cfg.visualizations_dir, "defects.jpg"))

        if cfg.save_crops or cfg.save_masks:
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
        report_name = "M1_AI_REPORT.md" if self.config.classification.use_ai_classifier else "M0_BASELINE_REPORT.md"
        generate_experiment_report(summary, df_classified, os.path.join(cfg.reports_dir, report_name))

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

        # Stage 1: Segmentation Engine Selection (YOLO Deep Learning vs Classic Watershed)
        use_yolo = (
            self.config.segmentation.engine == "yolo" and
            os.path.exists(self.config.segmentation.yolo_weights_path)
        )

        if use_yolo:
            instance_mask, clean_mask, grain_diagnostics, separation_stats = segment_yolo(
                cleaned_bgr,
                weights_path=self.config.segmentation.yolo_weights_path,
                conf_threshold=0.25,
                imgsz=1280,
                max_det=2000
            )
            otsu_val = 0.0
            seg_comparison = {
                "yolo11s_seg": {
                    "threshold_value": 0.0,
                    "foreground_ratio": float(np.mean(clean_mask > 0)),
                    "raw_connected_components": len(grain_diagnostics),
                    "edge_gradient_magnitude": 0.0
                }
            }
        else:
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
                min_valid_area_px=spatial_params["min_valid_area"]
            )

        # Stage 2: Measurement & Feature Extraction
        df_measurements, _ = measure_all_grains(
            instance_mask=instance_mask,
            image_rgb=original_rgb,
            grain_diagnostics=grain_diagnostics,
            cleared_margins=cleared_margins,
            calibration=calib,
            min_area_px=spatial_params["min_valid_area"]
        )

        # Stage 3: Classification Engine Selection (AI ResNet-50 vs Rule-Based Morphology)
        use_ai = (
            self.config.classification.use_ai_classifier and
            os.path.exists(self.config.classification.classifier_weights_path)
        )

        if use_ai:
            df_classified = classify_grains_ai(
                df_measurements,
                original_bgr,
                weights_path=self.config.classification.classifier_weights_path,
                broken_length_ratio_thresh=self.config.classification.broken_length_ratio_thresh,
                broken_aspect_ratio_thresh=self.config.classification.broken_aspect_ratio_thresh,
                pixel_scale_mm=calib.pixel_scale_mm
            )
        else:
            df_classified = classify_grains_rule_based(
                df_measurements,
                broken_length_ratio_thresh=self.config.classification.broken_length_ratio_thresh,
                broken_aspect_ratio_thresh=self.config.classification.broken_aspect_ratio_thresh,
                pixel_scale_mm=calib.pixel_scale_mm
            )

        classified_records = df_classified.to_dict(orient="records")

        quality_counts = df_classified["quality_class"].value_counts().to_dict() if not df_classified.empty else {}
        status_counts = df_classified["separation_status"].value_counts().to_dict() if not df_classified.empty else {}
        morph_stats = self._calculate_morphology_statistics(df_classified, calib)

        # ISO 7301 / Codex CXS 198-1995 & APSCSCL Specifications (Count and Mass Basis)
        metrics = compute_quality_metrics(df_classified, pixel_scale_mm=calib.pixel_scale_mm)
        commercial_grade = metrics["commercial_grade"]
        count_basis = metrics["count_basis"]
        mass_basis = metrics["mass_basis"]

        defect_breakdown = {}
        if "defects" in df_classified.columns:
            for dlist in df_classified["defects"]:
                for d in dlist:
                    defect_breakdown[d] = defect_breakdown.get(d, 0) + 1

        summary = {
            "timestamp": timestamp,
            "image_path": image_path,
            "image_dimensions": {"width": img_w, "height": img_h},
            "total_grains_detected": len(df_classified),
            "commercial_grade": commercial_grade,
            "count_basis": count_basis,
            "mass_basis": mass_basis,
            "total_sample_mass_g": metrics["total_mass_g"],
            "engine_mode": {
                "segmentation": f"yolo11s-seg ({'onnx' if self.config.segmentation.yolo_weights_path.endswith('.onnx') else 'pytorch'})" if use_yolo else f"classic-{self.config.segmentation.method}",
                "classification": f"resnet50-defect-ai ({'onnx' if self.config.classification.classifier_weights_path.endswith('.onnx') else 'pytorch'})" if use_ai else "classic-rule-based",
                "runtime": "onnxruntime" if (self.config.segmentation.yolo_weights_path.endswith(".onnx") or self.config.classification.classifier_weights_path.endswith(".onnx")) else "pytorch"
            },
            "calibration": {
                "is_verified": calib.is_verified,
                "effective_dpi": calib.effective_dpi,
                "pixel_scale_mm": calib.pixel_scale_mm,
                "pixels_per_mm": calib.pixels_per_mm,
                "source_info": calib.source_info,
            },
            "preprocessing": {"cleared_margins": cleared_margins},
            "segmentation": {
                "selected_method": "yolo11s_seg" if use_yolo else "otsu",
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
            "defect_breakdown": defect_breakdown,
            "status_counts": status_counts,
            "morphology_statistics": morph_stats,
        }

        if generate_outputs:
            self._export_pipeline_artifacts(
                original_bgr=original_bgr,
                clean_mask=clean_mask,
                instance_mask=instance_mask,
                classified_records=classified_records,
                df_classified=df_classified,
                summary=summary
            )

        return {
            "summary": summary,
            "measurements_df": df_classified,
            "instance_mask": instance_mask,
            "clean_mask": clean_mask,
        }
