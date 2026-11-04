# Hardware module initialization
from src.hardware.dual_platen import (
    DualPlatenManager,
    acquire_hardware_scan,
    create_distortion_free_composite,
    DEFAULT_SCAN_DPI,
    SCANNER_BOTTOM_ID,
    SCANNER_TOP_ID,
    SETTLE_DELAY
)

__all__ = [
    "DualPlatenManager",
    "acquire_hardware_scan",
    "create_distortion_free_composite",
    "DEFAULT_SCAN_DPI",
    "SCANNER_BOTTOM_ID",
    "SCANNER_TOP_ID",
    "SETTLE_DELAY"
]
