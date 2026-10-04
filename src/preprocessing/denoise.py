"""Image denoising and illumination preparation module."""

from typing import Tuple
import numpy as np
import cv2


def preprocess_image(
    image: np.ndarray,
    gaussian_kernel: int = 5,
    gaussian_sigma: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert image to grayscale and apply mild Gaussian filtering.
    
    Args:
        image: Input RGB/BGR or grayscale image.
        gaussian_kernel: Kernel size for Gaussian filter (must be odd).
        gaussian_sigma: Standard deviation for Gaussian kernel.
        
    Returns:
        gray: Original grayscale image (uint8).
        smoothed: Filtered grayscale image ready for segmentation (uint8).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gaussian_kernel > 1:
        # Ensure odd kernel
        if gaussian_kernel % 2 == 0:
            gaussian_kernel += 1
        smoothed = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), gaussian_sigma)
    else:
        smoothed = gray.copy()

    return gray, smoothed
