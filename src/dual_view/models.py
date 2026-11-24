"""
Dual-View Canonical Data Models for RiceAge Multi-Scanner System.
Establishes a single canonical PhysicalGrain record per physical grain,
linking ViewObservation records from bottom and top optical platens.
In Bottom-Primary Architecture, bottom scanner observations are authoritative,
while top scanner observations provide secondary verification only.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple


class CorrespondenceStatus(str, Enum):
    """
    Standard correspondence and registration statuses for dual-view physical grains.
    """
    MATCHED = "MATCHED"
    UNMATCHED_BOTTOM = "UNMATCHED_BOTTOM"
    UNMATCHED_TOP = "UNMATCHED_TOP"
    AMBIGUOUS = "AMBIGUOUS"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"
    SEGMENTATION_UNRESOLVED = "SEGMENTATION_UNRESOLVED"


class TopImageQualityStatus(str, Enum):
    """
    Quality gate assessment statuses for secondary top platen image.
    When top image is degraded (blurry, underexposed, overexposed), top verification
    must yield NOT_VERIFIABLE without misclassifying optical quality failure as a
    grain classification discrepancy.
    """
    SUITABLE = "SUITABLE"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"
    BLURRY = "BLURRY"
    UNDEREXPOSED = "UNDEREXPOSED"
    OVEREXPOSED = "OVEREXPOSED"
    LOW_CONTRAST = "LOW_CONTRAST"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class ViewObservation:
    """
    Represents an optical observation of a grain from a single scanner platen.
    Preserves scanner-specific coordinates, masks, confidences, and detection IDs.
    """
    observation_id: str
    scanner_id: str  # "bottom" or "top"
    detection_id: str  # Original scanner detection identifier (e.g., "b_12", "t_34")
    centroid_x: float
    centroid_y: float
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    area_px: float
    length_mm: float = 0.0
    width_mm: float = 0.0
    aspect_ratio: float = 1.0
    defect_class: str = "Good / Head Rice"
    defect_confidence: float = 1.0
    is_broken: bool = False
    mask_or_crop_reference: Optional[str] = None
    detection_confidence: float = 1.0
    raw_features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhysicalGrain:
    """
    Canonical representation of a single physical rice grain.
    Contains 0 or 1 bottom observation, and 0 or 1 top observation.
    Operates as the single source of truth for sample-level aggregation.
    """
    physical_grain_id: str
    correspondence_status: CorrespondenceStatus
    correspondence_confidence: float = 0.0
    bottom_observation_id: Optional[str] = None
    top_observation_id: Optional[str] = None
    bottom_obs: Optional[ViewObservation] = None
    top_obs: Optional[ViewObservation] = None
    quality_attributes: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_dual_view(self) -> bool:
        """True if successfully matched and observed from both surfaces."""
        return self.correspondence_status == CorrespondenceStatus.MATCHED

    @property
    def has_bottom(self) -> bool:
        return self.bottom_obs is not None

    @property
    def has_top(self) -> bool:
        return self.top_obs is not None

    @property
    def bottom_predicted_class(self) -> Optional[str]:
        return self.quality_attributes.get("bottom_predicted_class")

    @property
    def bottom_confidence(self) -> Optional[float]:
        return self.quality_attributes.get("bottom_confidence")

    @property
    def top_predicted_class(self) -> Optional[str]:
        return self.quality_attributes.get("top_predicted_class")

    @property
    def top_confidence(self) -> Optional[float]:
        return self.quality_attributes.get("top_confidence")

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "physical_grain_id": self.physical_grain_id,
            "correspondence_status": self.correspondence_status.value if isinstance(self.correspondence_status, CorrespondenceStatus) else str(self.correspondence_status),
            "correspondence_confidence": round(self.correspondence_confidence, 4),
            "bottom_observation_id": self.bottom_observation_id,
            "top_observation_id": self.top_observation_id,
            "quality_attributes": self.quality_attributes,
            "provenance": self.provenance,
            "bottom_observation": self.bottom_obs.to_dict() if self.bottom_obs else None,
            "top_observation": self.top_obs.to_dict() if self.top_obs else None,
        }
        return res


@dataclass
class DualViewFusionResult:
    """
    Consolidated container for the dual-view prototype output.
    Enforces strict distinction between single physical grains, multi-grain clusters,
    and unvalidated optical mass proxies.
    In Bottom-Primary Architecture, bottom scanner observations are authoritative,
    while top scanner observations provide secondary verification only.
    """
    physical_grains: List[PhysicalGrain]
    total_physical_grains: int  # Total individual physical grain candidates
    total_single_physical_grains: int  # Strictly single grain candidates (excluding cluster blobs)
    total_inventory_records: int  # Total canonical records in dataset (single grains + clusters)
    matched_count: int  # Verified 1-to-1 dual-view pairs
    unmatched_bottom_count: int
    unmatched_top_count: int
    ambiguous_count: int
    registration_failed_count: int
    unresolved_cluster_count: int  # Multi-grain clusters segregated from single grains
    estimated_mass_g: float
    estimated_optical_mass_unvalidated_g: float
    is_experimental: bool = False
    architecture_mode: str = "BOTTOM_SCANNER_PRIMARY"
    primary_scanner: str = "bottom"
    secondary_scanner: str = "top"
    discrepancy_count: int = 0
    count_agreement_pct: float = 100.0
    count_tolerance_status: str = "VERIFIED_EXCELLENT"
    bottom_grain_count: int = 0
    top_grain_count: int = 0
    top_image_quality_status: str = "SUITABLE"
    top_image_quality_metrics: Dict[str, Any] = field(default_factory=dict)
    mass_measurement_status: str = "NOT_PHYSICALLY_MEASURED"
    is_physically_measured: bool = False
    statutory_acceptance_usable: bool = False
    statutory_mass_disclaimer: str = (
        "Optical mass is an uncalibrated geometric proxy. Prohibited from statutory "
        "procurement, tender acceptance, or legal trade decisions without physical balance verification."
    )
    registration_metadata: Dict[str, Any] = field(default_factory=dict)
    aggregation_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_physical_grains": self.total_physical_grains,
            "total_single_physical_grains": self.total_single_physical_grains,
            "total_inventory_records": self.total_inventory_records,
            "matched_count": self.matched_count,
            "unmatched_bottom_count": self.unmatched_bottom_count,
            "unmatched_top_count": self.unmatched_top_count,
            "ambiguous_count": self.ambiguous_count,
            "registration_failed_count": self.registration_failed_count,
            "unresolved_cluster_count": self.unresolved_cluster_count,
            "discrepancy_count": self.discrepancy_count,
            "count_agreement_pct": self.count_agreement_pct,
            "count_tolerance_status": self.count_tolerance_status,
            "bottom_grain_count": self.bottom_grain_count,
            "top_grain_count": self.top_grain_count,
            "top_image_quality_status": self.top_image_quality_status,
            "top_image_quality_metrics": self.top_image_quality_metrics,
            "architecture_mode": self.architecture_mode,
            "primary_scanner": self.primary_scanner,
            "secondary_scanner": self.secondary_scanner,
            "estimated_mass_g": round(self.estimated_mass_g, 4),
            "estimated_optical_mass_unvalidated_g": round(self.estimated_optical_mass_unvalidated_g, 4),
            "mass_measurement_status": self.mass_measurement_status,
            "is_physically_measured": self.is_physically_measured,
            "statutory_acceptance_usable": self.statutory_acceptance_usable,
            "statutory_mass_disclaimer": self.statutory_mass_disclaimer,
            "is_experimental": self.is_experimental,
            "registration_metadata": self.registration_metadata,
            "aggregation_summary": self.aggregation_summary,
            "physical_grains": [g.to_dict() for g in self.physical_grains],
        }
