import os
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
    engine: str = "yolo"  # 'yolo' (Deep Learning) or 'classic' (Otsu/Adaptive/Triangle)
    method: str = "otsu"  # 'otsu', 'adaptive', 'triangle' (for classic engine)
    runtime: str = "auto"  # 'auto', 'onnx', 'pytorch'
    yolo_weights_path: str = "runs/rice_seg_yolo11s_1280/weights/best.pt"
    yolo_onnx_path: str = "runs/rice_seg_yolo11s_1280/weights/best.onnx"
    conf_threshold: float = 0.15  # YOLO segmentation confidence threshold (captures faint/touching grains down to 0.15)
    max_det: int = 3000  # maximum detection instances in YOLO NMS (prevents silent truncation)
    # Morphological cleaning (for classic engine)
    morph_kernel_size: int = 3
    min_object_area_px: int = 40  # filter out dust/noise < 40 px
    fill_holes: bool = True


@dataclass
class SeparationConfig:
    # Separation Profile: 'baseline' (safe rollback) or 'candidate_2' (optimized balanced)
    profile: str = "baseline"
    # Touching grain separation (watershed & distance transform) settings.
    enable_watershed: bool = True
    # Area threshold above which a connected component is scrutinized as a cluster
    cluster_area_threshold_px: int = 650
    # Solidity (area / convex_hull_area) threshold below which component is considered touching
    solidity_threshold: float = 0.92
    # Distance transform peak finding parameters
    min_peak_distance_px: int = 8
    peak_threshold_rel: float = 0.35
    # Enable selective cluster recovery on under-segmented clusters
    enable_selective_cluster_recovery: bool = False
    # Maximum allowed clusters to split
    max_cluster_area_px: int = 15000

    def apply_profile(self, profile_name: str) -> None:
        p = (profile_name or "").strip().lower()
        if p in ("candidate_2", "candidate2", "cand2"):
            self.profile = "candidate_2"
            self.min_peak_distance_px = 6
            self.peak_threshold_rel = 0.32
            self.solidity_threshold = 0.92
            self.enable_selective_cluster_recovery = True
        elif p in ("baseline", "base", "default"):
            self.profile = "baseline"
            self.min_peak_distance_px = 8
            self.peak_threshold_rel = 0.35
            self.solidity_threshold = 0.92
            self.enable_selective_cluster_recovery = False
        else:
            raise ValueError(f"Unknown separation profile '{profile_name}'. Must be 'baseline' or 'candidate_2'.")

    def __post_init__(self):
        env_p = os.environ.get("RICEAGE_SEPARATION_PROFILE") or os.environ.get("RICEAGE_PROFILE")
        if env_p:
            self.apply_profile(env_p)
        elif self.profile:
            self.apply_profile(self.profile)

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
    # AI Classifier settings
    use_ai_classifier: bool = True
    runtime: str = "auto"  # 'auto', 'onnx', 'pytorch'
    classifier_weights_path: str = "runs/stage2_resnet50/best_resnet50.pth"
    classifier_onnx_path: str = "runs/stage2_resnet50/grain_resnet50.onnx"
    classifier_model_type: str = "resnet50"


@dataclass
class DualViewConfig:
    # Operational Mode Selection: 'single_view' (default) or 'bottom_primary_verification' (Mode C)
    mode: str = "single_view"
    # Feature flag to enable experimental dual-view registration & fusion
    enabled: bool = False  # Feature flag default: False (safe single-view baseline preserved)
    # Architecture Mode: 'bottom_primary' (single view) or 'bottom_primary_verification' (Mode C)
    architecture_mode: str = "bottom_primary"

    def __post_init__(self):
        env_mode = os.environ.get("RICEAGE_DUAL_VIEW_MODE")
        if env_mode:
            self.apply_mode(env_mode)

    def apply_mode(self, mode_name: str) -> None:
        clean = mode_name.strip().lower()
        if clean in ("single_view", "single", "bottom_primary_only", "baseline"):
            self.mode = "single_view"
            self.enabled = False
            self.architecture_mode = "bottom_primary"
        elif clean in ("bottom_primary_verification", "mode_c", "verification"):
            self.mode = "bottom_primary_verification"
            self.enabled = True
            self.architecture_mode = "bottom_primary_verification"
        elif clean in ("symmetric", "experimental_canonical_fusion", "canonical_fusion", "mode_b"):
            raise ValueError(
                "Symmetric dual-view fusion (Mode B) is strictly prohibited from production "
                "to prevent +78% grain count inflation and statutory non-compliance. "
                "Supported modes: 'single_view' (default) or 'bottom_primary_verification'."
            )
        else:
            raise ValueError(
                f"Unknown dual-view operational mode: '{mode_name}'. "
                "Supported modes: 'single_view' (default) or 'bottom_primary_verification'."
            )
    primary_scanner: str = "bottom"
    secondary_scanner: str = "top"
    # Secondary Top Image Quality Gate Thresholds
    top_min_laplacian_variance: float = 8.0   # Blur threshold on top image foreground
    top_min_contrast: float = 12.0             # RMS contrast minimum floor
    top_min_brightness: float = 15.0           # Underexposure minimum floor
    top_max_brightness: float = 240.0          # Overexposure saturation ceiling
    # Registration & Matching Parameters (raw sensor pixel coordinates at 600 DPI)
    max_matching_distance_px: float = 35.0  # Max distance in unified raw sensor coords
    ambiguity_delta_px: float = 8.0          # Nearest-neighbor ambiguity margin
    cost_spatial_weight: float = 0.50        # Weight for reprojection distance
    cost_area_weight: float = 0.30           # Weight for area difference
    cost_aspect_ratio_weight: float = 0.20   # Weight for aspect ratio difference
    max_hungarian_cost: float = 0.85         # Cost ceiling above which match is rejected
    cluster_area_threshold_px: float = 9000.0 # Area threshold above which grain is segregated as unresolved cluster
    cluster_solidity_threshold: float = 0.80 # Solidity threshold below which grain is considered cluster
    # Static direct homography reference (Phase 3.2 raw coordinate reflection matrix)
    homography_matrix: Tuple[Tuple[float, float, float], ...] = (
        (-0.998463, -0.000463, 4917.872),
        ( 0.001606,  0.999657,   21.086),
        ( 0.000000,  0.000000,    1.000)
    )
    # Registration health checks
    min_determinant: float = -1.05
    max_determinant: float = -0.95
    min_scale: float = 0.95
    max_scale: float = 1.05
    max_reprojection_rmse_px: float = 8.0
    # Safe Single-View Fallback Configuration (when dual_view is disabled)
    single_view_fallback_scanner: str = "bottom"  # "bottom" or "top"
    # Defect Fusion Governance & Traceability
    defect_fusion_rule: str = "surface_defect_precedence"
    defect_fusion_version: str = "1.0.0-experimental"
    defect_precedence_is_standard_approved: bool = False
    defect_severity_order: Tuple[str, ...] = (
        "Foreign Matter",
        "Damaged Grains",
        "Discoloured Grains",
        "Small Brokens (<25%)",
        "Big Brokens (25-75%)",
        "Chalky Grains",
        "Good / Head Rice"
    )


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
    feedback_db_path: Optional[str] = None
    persist_feedback: bool = False


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
    dual_view: DualViewConfig = field(default_factory=DualViewConfig)
