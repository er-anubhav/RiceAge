"""End-to-end execution pipeline for Rice Grain Quality Analyzer."""

import os
import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import cv2

from configs.default_config import PipelineConfig
from src.preprocessing.calibration import resolve_calibration, inspect_metadata
from src.preprocessing.border_clearing import clear_scanner_bezel
from src.preprocessing.denoise import preprocess_image
from src.segmentation.binarize import segment_otsu, compare_segmentation_methods
from src.segmentation.morphology import cleanup_mask
from src.separation.watershed import separate_touching_grains
from src.measurement import measure_all_grains
from src.classification.rule_based import classify_grains_rule_based
from src.reporting.visualizer import (
    create_segmentation_overlay,
    create_instances_visualization,
    create_measurements_visualization,
    save_visualization,
)
from src.reporting.exporter import (
    export_crops_and_masks,
    export_measurements_csv,
    export_summary_json,
)
from src.reporting.report_generator import generate_experiment_report


class RiceQualityPipeline:
    """Master pipeline orchestrating end-to-end rice grain analysis."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def run(
        self,
        image_path: str,
        raw_reference_path: Optional[str] = None,
        generate_outputs: bool = True
    ) -> Dict[str, Any]:
        """Execute complete pipeline on an input image.
        
        Args:
            image_path: Path to the sample image (e.g. data/samples/sample_1489x2048.jpg).
            raw_reference_path: Optional path to the unscaled raw scan for verified calibration.
            generate_outputs: Whether to export crops, masks, visualizations, and CSV.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        timestamp = datetime.datetime.now().isoformat()
        original_bgr = cv2.imread(image_path)
        if original_bgr is None:
            raise ValueError(f"Failed to load image via OpenCV: {image_path}")

        img_h, img_w = original_bgr.shape[:2]
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        # 1. Calibration
        calib = resolve_calibration(
            image_path,
            raw_reference_path=raw_reference_path,
            assumed_dpi=self.config.calibration.assumed_dpi
        )

        # 2. Scanner Bezel and Border Artifact Clearance
        cleaned_bgr, cleared_margins = clear_scanner_bezel(
            original_bgr,
            saturation_thresh=self.config.preprocessing.bezel_intensity_thresh
        )

        # 3. Denoising & Filtering
        gray, smoothed = preprocess_image(
            cleaned_bgr,
            gaussian_kernel=self.config.preprocessing.gaussian_kernel,
            gaussian_sigma=self.config.preprocessing.gaussian_sigma
        )

        # 4. Comparative & Primary Binarization
        seg_comparison = compare_segmentation_methods(smoothed)
        binary_raw, otsu_val = segment_otsu(smoothed)

        # 5. Morphological Cleanup
        clean_mask = cleanup_mask(
            binary_raw,
            kernel_size=self.config.segmentation.morph_kernel_size,
            min_area_px=self.config.segmentation.min_object_area_px,
            fill_holes=self.config.segmentation.fill_holes
        )

        # 6. Separation via Selective Marker-Controlled Watershed
        instance_mask, grain_diagnostics, separation_stats = separate_touching_grains(
            clean_mask,
            min_peak_distance_px=self.config.separation.min_peak_distance_px,
            peak_threshold_rel=self.config.separation.peak_threshold_rel,
            cluster_area_thresh_px=self.config.separation.cluster_area_threshold_px,
            solidity_thresh=self.config.separation.solidity_threshold,
            min_valid_area_px=self.config.measurement.valid_grain_min_area_px
        )

        # 7. Morphology and Color Feature Extraction
        df_measurements, records = measure_all_grains(
            instance_mask=instance_mask,
            image_rgb=original_rgb,
            grain_diagnostics=grain_diagnostics,
            cleared_margins=cleared_margins,
            calibration=calib,
            min_area_px=self.config.measurement.valid_grain_min_area_px
        )

        # 8. Standard Rule-Based Quality Categorization
        df_classified = classify_grains_rule_based(
            df_measurements,
            broken_length_ratio_thresh=self.config.classification.broken_length_ratio_thresh,
            broken_aspect_ratio_thresh=self.config.classification.broken_aspect_ratio_thresh
        )
        classified_records = df_classified.to_dict(orient="records")

        # 9. Aggregate Statistics
        quality_counts = df_classified["quality_class"].value_counts().to_dict() if not df_classified.empty else {}
        status_counts = df_classified["separation_status"].value_counts().to_dict() if not df_classified.empty else {}

        # Statistical distributions
        stats = {}
        for col in ["length_px", "width_px", "area_px", "aspect_ratio", "circularity", "solidity"]:
            if col in df_classified.columns and not df_classified.empty:
                s = df_classified[col]
                stats[f"{col}_mean"] = round(float(s.mean()), 2)
                stats[f"{col}_std"] = round(float(s.std()), 2)
                stats[f"{col}_median"] = round(float(s.median()), 2)
                stats[f"{col}_min"] = round(float(s.min()), 2)
                stats[f"{col}_max"] = round(float(s.max()), 2)

        if calib.pixel_scale_mm is not None:
            for col in ["length_mm", "width_mm", "area_mm2"]:
                if col in df_classified.columns and not df_classified.empty:
                    s = df_classified[col]
                    stats[f"{col}_mean"] = round(float(s.mean()), 2)
                    stats[f"{col}_std"] = round(float(s.std()), 2)
                    stats[f"{col}_median"] = round(float(s.median()), 2)
                    stats[f"{col}_min"] = round(float(s.min()), 2)
                    stats[f"{col}_max"] = round(float(s.max()), 2)

        summary = {
            "timestamp": timestamp,
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
            "preprocessing": {
                "cleared_margins": cleared_margins,
            },
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
            "morphology_statistics": stats,
        }

        # 10. Generate Output Artifacts
        if generate_outputs:
            cfg_exp = self.config.export

            # 1. outputs/visualizations/original.jpg
            save_visualization(original_bgr, os.path.join(cfg_exp.visualizations_dir, "original.jpg"))

            # 2. outputs/visualizations/segmentation.jpg
            seg_overlay = create_segmentation_overlay(original_bgr, clean_mask)
            save_visualization(seg_overlay, os.path.join(cfg_exp.visualizations_dir, "segmentation.jpg"))

            # 3. outputs/visualizations/instances.jpg
            inst_vis = create_instances_visualization(original_bgr, instance_mask, classified_records)
            save_visualization(inst_vis, os.path.join(cfg_exp.visualizations_dir, "instances.jpg"))

            # 4. outputs/visualizations/measurements.jpg
            meas_vis = create_measurements_visualization(original_bgr, classified_records, instance_mask)
            save_visualization(meas_vis, os.path.join(cfg_exp.visualizations_dir, "measurements.jpg"))

            # 5. crops & masks
            if cfg_exp.save_crops or cfg_exp.save_masks:
                export_crops_and_masks(
                    original_bgr=original_bgr,
                    instance_mask=instance_mask,
                    measurements=classified_records,
                    crops_dir=cfg_exp.crops_dir,
                    masks_dir=cfg_exp.masks_dir,
                    max_count=cfg_exp.max_crops_to_save
                )

            # 6. CSV
            csv_path = os.path.join(cfg_exp.output_dir, "grain_measurements.csv")
            export_measurements_csv(df_classified, csv_path)

            # 7. JSON Summary
            json_path = os.path.join(cfg_exp.output_dir, "summary.json")
            export_summary_json(summary, json_path)

            # 8. Report
            report_path = os.path.join(cfg_exp.reports_dir, "M0_BASELINE_REPORT.md")
            generate_experiment_report(summary, df_classified, report_path)

        return {
            "summary": summary,
            "measurements_df": df_classified,
            "instance_mask": instance_mask,
            "clean_mask": clean_mask,
        }
