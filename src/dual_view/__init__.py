"""
RiceAge Dual-View Experimental Integration Prototype Module.
Exports data models, registration, correspondence, and fusion engines.
"""

from src.dual_view.models import (
    CorrespondenceStatus,
    TopImageQualityStatus,
    ViewObservation,
    PhysicalGrain,
    DualViewFusionResult
)
from src.dual_view.registration import DualViewRegistration
from src.dual_view.correspondence import DualViewCorrespondenceEngine
from src.dual_view.fusion import (
    DualViewFusionEngine,
    dataframe_to_view_observations,
    assess_image_quality
)

__all__ = [
    "CorrespondenceStatus",
    "TopImageQualityStatus",
    "ViewObservation",
    "PhysicalGrain",
    "DualViewFusionResult",
    "DualViewRegistration",
    "DualViewCorrespondenceEngine",
    "DualViewFusionEngine",
    "dataframe_to_view_observations",
    "assess_image_quality"
]
