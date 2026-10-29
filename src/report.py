"""
report.py
---------
Generates summary reports and exports annotated comparison images.
"""

import cv2
import numpy as np


def create_side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    label1: str = "Original",
    label2: str = "Modified",
) -> np.ndarray:
    """Create a side-by-side comparison image with labels."""
    # Ensure both are BGR
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Match heights
    max_h = max(h1, h2)
    if h1 < max_h:
        pad = np.full((max_h - h1, w1, 3), 255, dtype=np.uint8)
        img1 = np.vstack([img1, pad])
    if h2 < max_h:
        pad = np.full((max_h - h2, w2, 3), 255, dtype=np.uint8)
        img2 = np.vstack([img2, pad])

    # Add labels
    label_h = 40
    label_bar1 = np.full((label_h, w1, 3), 40, dtype=np.uint8)
    label_bar2 = np.full((label_h, w2, 3), 40, dtype=np.uint8)

    cv2.putText(label_bar1, label1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(label_bar2, label2, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    col1 = np.vstack([label_bar1, img1])
    col2 = np.vstack([label_bar2, img2])

    # Separator
    sep = np.full((col1.shape[0], 4, 3), 200, dtype=np.uint8)

    return np.hstack([col1, sep, col2])


def create_comparison_strip(
    original: np.ndarray,
    aligned: np.ndarray,
    heatmap: np.ndarray,
    highlighted: np.ndarray,
) -> np.ndarray:
    """
    Create a 4-panel comparison strip:
    Original | Aligned | Heatmap | Highlighted Changes
    """
    panels = [original, aligned, heatmap, highlighted]
    labels = ["Original", "Aligned", "Difference Map", "Changes Detected"]

    # Ensure all BGR
    bgr_panels = []
    for p in panels:
        if len(p.shape) == 2:
            bgr_panels.append(cv2.cvtColor(p, cv2.COLOR_GRAY2BGR))
        else:
            bgr_panels.append(p.copy())

    # Resize all to same height
    target_h = max(p.shape[0] for p in bgr_panels)
    resized = []
    for p in bgr_panels:
        if p.shape[0] != target_h:
            scale = target_h / p.shape[0]
            new_w = int(p.shape[1] * scale)
            p = cv2.resize(p, (new_w, target_h))
        resized.append(p)

    # Add labels
    labeled = []
    for img, label in zip(resized, labels):
        h, w = img.shape[:2]
        label_bar = np.full((35, w, 3), 40, dtype=np.uint8)
        cv2.putText(label_bar, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        labeled.append(np.vstack([label_bar, img]))

    # Separators
    result = labeled[0]
    for panel in labeled[1:]:
        # Match heights if needed
        if result.shape[0] != panel.shape[0]:
            max_h = max(result.shape[0], panel.shape[0])
            if result.shape[0] < max_h:
                pad = np.full((max_h - result.shape[0], result.shape[1], 3), 255, dtype=np.uint8)
                result = np.vstack([result, pad])
            if panel.shape[0] < max_h:
                pad = np.full((max_h - panel.shape[0], panel.shape[1], 3), 255, dtype=np.uint8)
                panel = np.vstack([panel, pad])
        sep = np.full((result.shape[0], 3, 3), 180, dtype=np.uint8)
        result = np.hstack([result, sep, panel])

    return result


def encode_image_to_bytes(img: np.ndarray, fmt: str = ".png") -> bytes:
    """Encode an image array to bytes for download."""
    success, encoded = cv2.imencode(fmt, img)
    if not success:
        raise RuntimeError(f"Failed to encode image as {fmt}")
    return encoded.tobytes()


def format_summary_text(summary: dict) -> str:
    """Format the change summary into a readable string."""
    lines = [
        "=" * 50,
        "  CHANGE DETECTION SUMMARY",
        "=" * 50,
        f"  Regions detected:    {summary['num_regions']}",
        f"  Total changed area:  {summary['total_changed_area']:,} px",
        f"  Image total area:    {summary['image_area']:,} px",
        f"  Percentage changed:  {summary['percent_changed']}%",
        "-" * 50,
    ]

    if summary["regions"]:
        lines.append("  Region details:")
        for i, region in enumerate(summary["regions"], 1):
            x, y, w, h = region["bbox"]
            lines.append(
                f"    #{i}: Position ({x},{y}), "
                f"Size {w}x{h}, Area {region['area']:,} px"
            )
    else:
        lines.append("  No significant changes detected.")

    lines.append("=" * 50)
    return "\n".join(lines)
