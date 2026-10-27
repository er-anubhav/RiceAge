# Configuration parameters for the Rice Grain Quality Analyzer pipeline.

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class PreprocessingConfig:
    # Preprocessing and border clearing settings.
    gaussian_kernel: int = 5
    gaussian_sigma: float = 0.8
    # Saturated bezel threshold for scanner edges
    bezel_intensity_thresh: int = 245
    # Max fraction of border allowed to search for bezel
    max_bezel_search_ratio: float = 0.05
    # Distance from image border in pixels to flag a grain as boundary
    boundary_margin_px: int = 4


@dataclass
class CalibrationConfig:
    # Calibration settings for converting pixels to millimeters.
    # Assumed DPI if metadata is missing or if downscaled
    assumed_dpi: Optional[float] = None
    # Nominal 1200 DPI scanner physical constant (1 inch = 25.4 mm)
    # 1 px @ 1200 DPI = 25.4 / 1200 = 0.0211667 mm
    raw_dpi: float = 1200.0


@dataclass
class SegmentationConfig:
    # Segmentation and thresholding parameters.
    method: str = "otsu"  # 'otsu', 'adaptive', 'triangle'
    # Morphological cleaning
    morph_kernel_size: int = 3
    min_object_area_px: int = 40  # filter out dust/noise < 40 px
    fill_holes: bool = True


@dataclass
class SeparationConfig:
    # Touching grain separation (watershed & distance transform) settings.
    enable_watershed: bool = True
    # Area threshold above which a connected component is scrutinized as a cluster
    cluster_area_threshold_px: int = 650
    # Solidity (area / convex_hull_area) threshold below which component is considered touching
    solidity_threshold: float = 0.92
    # Distance transform peak finding parameters
    min_peak_distance_px: int = 8
    peak_threshold_rel: float = 0.35
    # Maximum allowed clusters to split
    max_cluster_area_px: int = 15000


@dataclass
class MeasurementConfig:
    # Feature extraction and metrics calculation settings.
    calculate_color: bool = True
    calculate_highlight_ratio: bool = True
    specular_highlight_thresh: int = 240
    # Minimum area to consider a valid grain for reporting
    valid_grain_min_area_px: int = 50


@dataclass
class ClassificationConfig:
    # Baseline rule-based quality categorization.
    # Broken grain length ratio relative to whole grain median length (Codex/IRRI standard: < 75%)
    broken_length_ratio_thresh: float = 0.75
    # Aspect ratio threshold (length / width)
    broken_aspect_ratio_thresh: float = 1.6


@dataclass
class ExportConfig:
    # Output generation settings.
    save_crops: bool = True
    save_masks: bool = True
    max_crops_to_save: int = 1000  # save up to 1000 crops
    output_dir: str = "outputs"
    visualizations_dir: str = "outputs/visualizations"
    crops_dir: str = "outputs/crops"
    masks_dir: str = "outputs/masks"
    reports_dir: str = "outputs/reports"
    # Persist per-grain server-side feedback context for the review workflow.
    # This is additive and can be disabled to preserve a no-database run.
    persist_feedback: bool = True
    feedback_db_path: Optional[str] = None


@dataclass
class PipelineConfig:
    # Master pipeline configuration.
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
