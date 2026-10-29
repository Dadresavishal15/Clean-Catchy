"""
alignment.py
------------
Aligns two images using feature detection (ORB) and RANSAC-based
homography estimation. Also supports phase-correlation for simple
translation-only alignment.
"""

import cv2
import numpy as np


def align_orb(
    img_ref: np.ndarray,
    img_target: np.ndarray,
    max_features: int = 5000,
    good_match_ratio: float = 0.75,
    min_matches: int = 10,
) -> tuple[np.ndarray, bool, str]:
    """
    Align img_target to img_ref using ORB feature matching + RANSAC homography.

    Returns:
        aligned_image: The warped target image aligned to the reference.
        success: Whether alignment was successful.
        message: Status message about the alignment process.
    """
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=max_features)

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img_ref, None)
    kp2, des2 = orb.detectAndCompute(img_target, None)

    if des1 is None or des2 is None:
        return img_target, False, "Could not detect features in one or both images."

    # Match features using BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des2, des1, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < good_match_ratio * n.distance:
                good_matches.append(m)

    if len(good_matches) < min_matches:
        return (
            img_target,
            False,
            f"Only {len(good_matches)} good matches found (need {min_matches}). "
            "Images may be too different or alignment is not needed.",
        )

    # Extract matched point coordinates
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        return img_target, False, "Could not compute homography matrix."

    # Warp target image
    h, w = img_ref.shape[:2]
    aligned = cv2.warpPerspective(img_target, H, (w, h), borderValue=255)

    inliers = int(mask.sum()) if mask is not None else 0
    msg = f"Aligned using {inliers} inlier matches out of {len(good_matches)} good matches."

    return aligned, True, msg


def align_phase_correlation(
    img_ref: np.ndarray,
    img_target: np.ndarray,
) -> tuple[np.ndarray, bool, str]:
    """
    Align img_target to img_ref using phase correlation.
    Only corrects translation (shift), not rotation or scale.

    Returns:
        aligned_image, success, message
    """
    # Phase correlation requires float32
    ref_f = np.float32(img_ref)
    tgt_f = np.float32(img_target)

    shift, response = cv2.phaseCorrelate(ref_f, tgt_f)

    dx, dy = shift
    if abs(dx) < 1 and abs(dy) < 1:
        return img_target, True, "No significant shift detected; images are already aligned."

    # Build translation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = img_ref.shape[:2]
    aligned = cv2.warpAffine(img_target, M, (w, h), borderValue=255)

    msg = f"Shifted by ({dx:.1f}, {dy:.1f}) pixels. Phase correlation response: {response:.4f}."
    return aligned, True, msg


def align_images(
    img_ref: np.ndarray,
    img_target: np.ndarray,
    method: str = "orb",
) -> tuple[np.ndarray, bool, str]:
    """
    Align two images using the specified method.

    Args:
        img_ref: Reference (original) grayscale image.
        img_target: Target (modified) grayscale image to align.
        method: 'orb' for ORB+RANSAC, 'phase' for phase correlation, 'none' to skip.

    Returns:
        aligned_image, success, message
    """
    if method == "orb":
        return align_orb(img_ref, img_target)
    elif method == "phase":
        return align_phase_correlation(img_ref, img_target)
    elif method == "none":
        return img_target, True, "Alignment skipped."
    else:
        raise ValueError(f"Unknown alignment method: {method}")
