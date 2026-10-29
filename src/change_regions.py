"""
change_regions.py
-----------------
Extracts, filters, and highlights changed regions from a binary difference mask.
"""

import cv2
import numpy as np


def clean_mask(
    binary_mask: np.ndarray,
    kernel_size: int = 5,
    morph_op: str = "close",
) -> np.ndarray:
    """
    Apply morphological operations to clean up noise in the binary mask.

    Args:
        binary_mask: Binary mask (0/255).
        kernel_size: Size of the structuring element.
        morph_op: 'close' (fill gaps), 'open' (remove specks), or 'both'.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if morph_op == "close":
        return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    elif morph_op == "open":
        return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    elif morph_op == "both":
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed
    else:
        return binary_mask


def find_change_regions(
    binary_mask: np.ndarray,
    min_area: int = 100,
) -> list[dict]:
    """
    Find contours in the binary mask and return bounding boxes for regions
    above the minimum area threshold.

    Returns:
        List of dicts with keys: 'bbox' (x, y, w, h), 'area', 'contour'.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append({
                "bbox": (x, y, w, h),
                "area": area,
                "contour": contour,
            })

    # Sort by area descending
    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions


def draw_highlights(
    image: np.ndarray,
    regions: list[dict],
    box_color: tuple = (0, 0, 255),
    box_thickness: int = 2,
    overlay_alpha: float = 0.3,
) -> np.ndarray:
    """
    Draw bounding boxes and semi-transparent overlays on the image
    to highlight changed regions.

    Args:
        image: The original image (BGR or grayscale, will be converted to BGR).
        regions: List of region dicts from find_change_regions.
        box_color: BGR color for the bounding boxes.
        box_thickness: Thickness of bounding box lines.
        overlay_alpha: Transparency of the overlay fill.

    Returns:
        Annotated image with highlights.
    """
    if len(image.shape) == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    overlay = result.copy()

    for region in regions:
        x, y, w, h = region["bbox"]
        # Draw filled rectangle on overlay
        cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, -1)
        # Draw border on result
        cv2.rectangle(result, (x, y), (x + w, y + h), box_color, box_thickness)

    # Blend overlay
    result = cv2.addWeighted(overlay, overlay_alpha, result, 1 - overlay_alpha, 0)

    # Redraw borders on top for clarity
    for region in regions:
        x, y, w, h = region["bbox"]
        cv2.rectangle(result, (x, y), (x + w, y + h), box_color, box_thickness)

    return result


def get_summary(regions: list[dict], image_shape: tuple) -> dict:
    """
    Generate a summary of detected change regions.

    Returns:
        Dictionary with summary statistics.
    """
    total_area = sum(r["area"] for r in regions)
    image_area = image_shape[0] * image_shape[1]
    percent_changed = (total_area / image_area * 100) if image_area > 0 else 0

    return {
        "num_regions": len(regions),
        "total_changed_area": total_area,
        "image_area": image_area,
        "percent_changed": round(percent_changed, 2),
        "regions": [
            {"bbox": r["bbox"], "area": r["area"]}
            for r in regions
        ],
    }
