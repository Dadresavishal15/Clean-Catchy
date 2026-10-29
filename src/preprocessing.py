"""
preprocessing.py
----------------
Handles image loading, grayscale conversion, resizing, and noise reduction.
"""

import cv2
import numpy as np


def load_image(image_bytes: bytes) -> np.ndarray:
    """Load an image from raw bytes into a BGR numpy array."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid PNG or JPG file.")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_to_match(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize both images to the same dimensions (the larger of the two
    dimensions), preserving aspect ratio by padding with white.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    target_h = max(h1, h2)
    target_w = max(w1, w2)

    img1_resized = _resize_and_pad(img1, target_w, target_h)
    img2_resized = _resize_and_pad(img2, target_w, target_h)

    return img1_resized, img2_resized


def _resize_and_pad(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image to fit within target dimensions and pad with white."""
    h, w = img.shape[:2]

    # Calculate scale to fit within target
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Determine padding color (white)
    if len(img.shape) == 2:
        padded = np.full((target_h, target_w), 255, dtype=np.uint8)
    else:
        padded = np.full((target_h, target_w, img.shape[2]), 255, dtype=np.uint8)

    # Center the image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded


def apply_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply mild Gaussian blur to reduce noise."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def normalize_lighting(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to normalize uneven lighting.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def preprocess_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    blur_ksize: int = 5,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for a pair of images.
    Returns two grayscale, size-matched, optionally blurred and normalized images.
    """
    # Convert to grayscale
    gray1 = to_grayscale(img1)
    gray2 = to_grayscale(img2)

    # Resize to match
    gray1, gray2 = resize_to_match(gray1, gray2)

    # Normalize lighting
    if normalize:
        gray1 = normalize_lighting(gray1)
        gray2 = normalize_lighting(gray2)

    # Apply blur
    if blur_ksize > 0:
        gray1 = apply_blur(gray1, blur_ksize)
        gray2 = apply_blur(gray2, blur_ksize)

    return gray1, gray2
