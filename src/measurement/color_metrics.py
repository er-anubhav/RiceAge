"""Per-grain color, brightness, and specular highlight feature extraction."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2


@dataclass
class GrainColor:
    """Photometric and color properties of an individual grain."""
    grain_id: int
    mean_r: float
    mean_g: float
    mean_b: float
    mean_intensity: float
    std_intensity: float
    highlight_ratio: float  # fraction of pixels > highlight threshold (lustre indicator)
    uniformity: float       # surface uniformity (1 - cv)


def extract_grain_color(
    grain_id: int,
    rgb_image: np.ndarray,
    grain_mask: np.ndarray,
    specular_thresh: int = 240
) -> Optional[GrainColor]:
    """Calculate color, brightness, and surface reflection metrics for a single grain.
    
    Args:
        grain_id: Unique grain identifier.
        rgb_image: (H, W, 3) RGB or BGR image (assumes RGB ordering).
        grain_mask: (H, W) uint8 binary mask.
        specular_thresh: Pixel intensity threshold to detect specular highlights.
    """
    grain_pixels = grain_mask > 0
    if not np.any(grain_pixels):
        return None

    r_vals = rgb_image[:, :, 0][grain_pixels]
    g_vals = rgb_image[:, :, 1][grain_pixels]
    b_vals = rgb_image[:, :, 2][grain_pixels]

    mean_r = float(np.mean(r_vals))
    mean_g = float(np.mean(g_vals))
    mean_b = float(np.mean(b_vals))

    gray_vals = 0.299 * r_vals + 0.587 * g_vals + 0.114 * b_vals
    mean_int = float(np.mean(gray_vals))
    std_int = float(np.std(gray_vals))

    highlight_count = np.sum(gray_vals >= specular_thresh)
    highlight_ratio = float(highlight_count / len(gray_vals))

    cov = std_int / (mean_int + 1e-5)
    uniformity = float(max(0.0, 1.0 - cov))

    return GrainColor(
        grain_id=grain_id,
        mean_r=round(mean_r, 1),
        mean_g=round(mean_g, 1),
        mean_b=round(mean_b, 1),
        mean_intensity=round(mean_int, 1),
        std_intensity=round(std_int, 1),
        highlight_ratio=round(highlight_ratio, 4),
        uniformity=round(uniformity, 3)
    )

