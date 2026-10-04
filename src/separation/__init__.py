"""Separation package for touching rice grain analysis."""

from .connected_components import extract_connected_components, summarize_connected_components
from .distance_transform import compute_distance_transform, detect_distance_peaks
from .cluster_classifier import evaluate_component_geometry, ComponentCategory, GrainStatus
from .watershed import separate_touching_grains

__all__ = [
    "extract_connected_components",
    "summarize_connected_components",
    "compute_distance_transform",
    "detect_distance_peaks",
    "evaluate_component_geometry",
    "ComponentCategory",
    "GrainStatus",
    "separate_touching_grains",
]
