"""
Streamlit Web Application: Handwritten Notes Change Detector
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

from src.preprocessing import load_image, preprocess_pair
from src.alignment import align_images
from src.comparison import compare_images
from src.change_regions import clean_mask, find_change_regions, draw_highlights, get_summary
from src.report import (
    create_comparison_strip,
    create_side_by_side,
    encode_image_to_bytes,
    format_summary_text,
)

st.set_page_config(
    page_title="Handwritten Notes Change Detector",
    page_icon="📝",
    layout="wide",
)

st.title("📝 Handwritten Notes Change Detector")
st.markdown(
    """
    Compare two scans or photos of handwritten notes/pages to detect additions, erasures, and edits.
    Uses classic Computer Vision techniques: **ORB Feature Alignment**, **SSIM Difference Mapping**, and **Contour Region Extraction**.
    """
)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("1. Alignment Options")
align_method = st.sidebar.selectbox(
    "Alignment Method",
    options=["orb", "phase", "none"],
    format_func=lambda x: {
        "orb": "ORB Feature Matching + RANSAC",
        "phase": "Phase Correlation (Translation only)",
        "none": "Skip Alignment (Pre-aligned scans)",
    }[x],
    index=0,
)

st.sidebar.subheader("2. Preprocessing & Diff Options")
comp_method = st.sidebar.selectbox(
    "Difference Method",
    options=["ssim", "pixel"],
    format_func=lambda x: "SSIM (Structural Similarity)" if x == "ssim" else "Absolute Pixel Difference",
    index=0,
)

normalize_lighting_flag = st.sidebar.checkbox("Normalize Lighting (CLAHE)", value=True)
blur_size = st.sidebar.slider("Gaussian Blur Kernel Size", min_value=0, max_value=11, value=5, step=2)
diff_threshold = st.sidebar.slider("Difference Threshold", min_value=10, max_value=100, value=35, step=5)

st.sidebar.subheader("3. Region Filtering")
min_area = st.sidebar.slider("Minimum Change Area (px²)", min_value=20, max_value=1000, value=120, step=10)
morph_op = st.sidebar.selectbox("Noise Cleanup", options=["close", "open", "both", "none"], index=0)

# Main area: File Uploaders
col_up1, col_up2 = st.columns(2)

with col_up1:
    file1 = st.file_uploader("Upload Original Page (Before)", type=["png", "jpg", "jpeg"])
with col_up2:
    file2 = st.file_uploader("Upload Modified Page (After)", type=["png", "jpg", "jpeg"])

# Helper function to generate sample test images if user wants demo
demo_clicked = st.button("Generate & Use Demo Note Samples")

if demo_clicked:
    # Create synthetic original note
    sample1 = np.full((700, 500, 3), 255, dtype=np.uint8)
    cv2.putText(sample1, "Computer Vision Notes", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    cv2.putText(sample1, "1. Edge Detection (Sobel, Canny)", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.putText(sample1, "2. Feature Matching (ORB, SIFT)", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.putText(sample1, "3. Homography & RANSAC", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.putText(sample1, "Exam date: 24th October", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 20, 20), 2)

    # Create modified note (slight shift/rotation + additions + erasures)
    sample2 = sample1.copy()
    # Add an item
    cv2.putText(sample2, "4. Image Segmentation (SSIM)", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 0, 0), 2)
    # Erase exam date and replace
    cv2.rectangle(sample2, (45, 295), (400, 335), (255, 255, 255), -1)
    cv2.putText(sample2, "Exam date: 15th November (POSTPONED)", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 2)

    # Apply minor rotation / translation to simulate camera shift
    center = (250, 350)
    rot_mat = cv2.getRotationMatrix2D(center, angle=1.2, scale=1.01)
    rot_mat[0, 2] += 5
    rot_mat[1, 2] += -4
    sample2 = cv2.warpAffine(sample2, rot_mat, (500, 700), borderValue=(255, 255, 255))

    st.session_state["demo_img1"] = sample1
    st.session_state["demo_img2"] = sample2

img1_bgr = None
img2_bgr = None

if file1 is not None and file2 is not None:
    img1_bgr = load_image(file1.read())
    img2_bgr = load_image(file2.read())
elif "demo_img1" in st.session_state and "demo_img2" in st.session_state:
    img1_bgr = st.session_state["demo_img1"]
    img2_bgr = st.session_state["demo_img2"]
    st.info("Showing results for generated synthetic notes. You can also upload your own files above!")

if img1_bgr is not None and img2_bgr is not None:
    st.success("Images loaded successfully. Running detection pipeline...")

    # Step 1: Preprocessing
    with st.spinner("Preprocessing images..."):
        gray1, gray2 = preprocess_pair(
            img1_bgr,
            img2_bgr,
            blur_ksize=blur_size,
            normalize=normalize_lighting_flag,
        )

    # Step 2: Alignment
    with st.spinner("Aligning modified page to original..."):
        aligned_gray2, align_success, align_msg = align_images(
            gray1, gray2, method=align_method
        )
        if not align_success:
            st.warning(f"Alignment notice: {align_msg}")
        else:
            st.info(f"Alignment status: {align_msg}")

    # Step 3: Comparison & Difference Map
    with st.spinner("Computing difference map..."):
        comp_res = compare_images(
            gray1,
            aligned_gray2,
            method=comp_method,
            threshold=diff_threshold,
        )
        diff_map = comp_res["diff_map"]
        binary_mask = comp_res["binary_mask"]
        heatmap = comp_res["heatmap"]
        ssim_score = comp_res["ssim_score"]

    # Step 4: Noise reduction & region extraction
    cleaned_mask = clean_mask(binary_mask, kernel_size=5, morph_op=morph_op)
    regions = find_change_regions(cleaned_mask, min_area=min_area)

    # Step 5: Highlight on target image
    highlighted_img = draw_highlights(aligned_gray2, regions)
    summary = get_summary(regions, aligned_gray2.shape)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Detected Changes", f"{summary['num_regions']} regions")
    m2.metric("Changed Area", f"{summary['percent_changed']}%")
    if ssim_score is not None:
        m3.metric("SSIM Similarity", f"{ssim_score:.4f}")
    else:
        m3.metric("Metric", "Absolute Diff")
    m4.metric("Status", "Clean" if summary['num_regions'] == 0 else "Changes Found")

    # Display views in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Results & Highlights", "🗺️ Difference Map & Heatmap", "🔍 Preprocessed & Aligned", "📊 Detailed Report"])

    with tab1:
        st.subheader("Highlighted Visual Changes")
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB), caption="Original Document", use_container_width=True)
        c2.image(cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB), caption="Detected Changes (Red Bounding Boxes)", use_container_width=True)

        # Export buttons
        strip_img = create_comparison_strip(gray1, aligned_gray2, heatmap, highlighted_img)
        strip_bytes = encode_image_to_bytes(strip_img, ".png")
        annotated_bytes = encode_image_to_bytes(highlighted_img, ".png")

        d_col1, d_col2 = st.columns(2)
        d_col1.download_button(
            label="⬇️ Download Highlighted Image",
            data=annotated_bytes,
            file_name="detected_changes.png",
            mime="image/png",
        )
        d_col2.download_button(
            label="⬇️ Download Full 4-Panel Strip",
            data=strip_bytes,
            file_name="comparison_strip.png",
            mime="image/png",
        )

    with tab2:
        st.subheader("Difference Map and Heatmap")
        h1, h2 = st.columns(2)
        h1.image(diff_map, caption="Grayscale Difference", use_container_width=True, clamp=True)
        h2.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), caption="Difference Heatmap (Jet Colormap)", use_container_width=True)

    with tab3:
        st.subheader("Alignment Verification")
        a1, a2 = st.columns(2)
        a1.image(gray1, caption="Original Grayscale Reference", use_container_width=True, clamp=True)
        a2.image(aligned_gray2, caption="Warped & Aligned Modified Page", use_container_width=True, clamp=True)

    with tab4:
        st.subheader("Summary Report")
        st.code(format_summary_text(summary), language="text")

else:
    st.info("👆 Please upload two images or click **'Generate & Use Demo Note Samples'** to get started.")
