"""
comparison.py
-------------
Computes difference maps between two aligned grayscale images
using SSIM and absolute pixel difference.
"""

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    def ssim(img1: np.ndarray, img2: np.ndarray, full: bool = False, **kwargs):
        """Pure OpenCV/NumPy fallback for structural similarity index (SSIM)."""
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        kernel_size = 11
        sigma = 1.5

        mu1 = cv2.GaussianBlur(img1, (kernel_size, kernel_size), sigma)
        mu2 = cv2.GaussianBlur(img2, (kernel_size, kernel_size), sigma)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (kernel_size, kernel_size), sigma) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (kernel_size, kernel_size), sigma) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (kernel_size, kernel_size), sigma) - mu1_mu2

        sim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        score = float(sim_map.mean())
        if full:
            return score, sim_map
        return score


def compute_ssim_diff(
    img1: np.ndarray,
    img2: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Compute SSIM between two grayscale images and return the difference map.

    Returns:
        score: Overall SSIM score (1.0 = identical).
        diff_map: Difference map as uint8 (0 = identical, 255 = max difference).
    """
    score, sim_map = ssim(img1, img2, full=True)

    # Convert similarity map to difference map
    # sim_map values are in [-1, 1], where 1 = identical
    diff_map = (1.0 - sim_map) / 2.0  # Normalize to [0, 1]
    diff_map = (diff_map * 255).astype(np.uint8)

    return score, diff_map


def compute_pixel_diff(
    img1: np.ndarray,
    img2: np.ndarray,
) -> np.ndarray:
    """
    Compute absolute pixel difference between two grayscale images.

    Returns:
        diff_map: Absolute difference as uint8.
    """
    diff = cv2.absdiff(img1, img2)
    return diff


def threshold_diff(
    diff_map: np.ndarray,
    threshold: int = 30,
) -> np.ndarray:
    """
    Apply binary threshold to a difference map.

    Returns:
        Binary mask where 255 = changed pixel, 0 = unchanged.
    """
    _, binary = cv2.threshold(diff_map, threshold, 255, cv2.THRESH_BINARY)
    return binary


def create_heatmap(diff_map: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale difference map to a colored heatmap for visualization.
    """
    heatmap = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
    return heatmap


def compare_images(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = "ssim",
    threshold: int = 30,
) -> dict:
    """
    Full comparison pipeline.

    Args:
        img1: Preprocessed grayscale reference image.
        img2: Preprocessed grayscale target image (already aligned).
        method: 'ssim' or 'pixel'.
        threshold: Threshold for binary mask.

    Returns:
        Dictionary with keys:
            'diff_map': Grayscale difference map.
            'binary_mask': Thresholded binary mask.
            'heatmap': Colored heatmap visualization.
            'ssim_score': SSIM score (only if method='ssim').
    """
    result = {}

    if method == "ssim":
        score, diff_map = compute_ssim_diff(img1, img2)
        result["ssim_score"] = score
    else:
        diff_map = compute_pixel_diff(img1, img2)
        result["ssim_score"] = None

    result["diff_map"] = diff_map
    result["binary_mask"] = threshold_diff(diff_map, threshold)
    result["heatmap"] = create_heatmap(diff_map)

    return result
