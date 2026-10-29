# Handwritten Notes Change Detector

A robust Computer Vision system that compares two photos or scans of handwritten notes/documents and visually highlights added, removed, or modified regions.

Built following the **Soft Computing & Computer Vision Project Roadmap (Project 2: Handwritten Notes Change Detector)**.

---

## 📌 Features

- **Chunk 1 — File Upload & Preview:** Supports drag-and-drop upload for PNG/JPG images or 1-click synthetic demo sample generation.
- **Chunk 2 — Preprocessing:** Aspect-ratio preserving resizing, white padding, grayscale conversion, Gaussian blur noise reduction, and CLAHE adaptive lighting normalization.
- **Chunk 3 — Robust Alignment:** ORB (Oriented FAST and Rotated BRIEF) feature matching with Lowe's ratio test and RANSAC homography estimation, plus Phase-Correlation translation support.
- **Chunk 4 — Difference Mapping:** Structural Similarity Index (SSIM) difference analysis and absolute pixel differencing with interactive threshold tuning.
- **Chunk 5 — Region Extraction & Highlighting:** Morphological noise cleanup (open/close), contour detection, area filtering, bounding boxes, and semi-transparent highlight overlays.
- **Chunk 6 — Interactive Streamlit UI & Export:** Tabbed interface with metrics, difference heatmaps, alignment checks, summary reports, and 1-click image/strip downloads.
- **Chunk 8 — Evaluation & Tests:** Automated unit tests covering preprocessing, SSIM comparison, and contour region detection.

---

## 🛠️ Tech Stack

- **Python 3.11+**
- **OpenCV (`opencv-python`)**: Image processing, ORB feature detection, homography, and morphology.
- **scikit-image (`scikit-image`)**: Structural Similarity Index (SSIM).
- **NumPy**: Matrix operations and transformations.
- **Streamlit**: Interactive web dashboard.
- **Pillow**: Image handling.
- **pytest**: Test suite.

---

## 📂 Project Structure

```text
CV/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── src/
│   ├── __init__.py             # Module initialization
│   ├── preprocessing.py        # Loading, grayscale, resize, CLAHE, blur
│   ├── alignment.py            # ORB + RANSAC homography, phase correlation
│   ├── comparison.py           # SSIM & pixel difference maps, heatmaps
│   ├── change_regions.py       # Morphology, contours, bounding boxes, stats
│   └── report.py               # Side-by-side, 4-panel strips, text summaries
└── tests/
    ├── test_preprocessing.py   # Preprocessing tests
    └── test_comparison.py      # SSIM & contour tests
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run the Test Suite
```bash
pytest tests/
```

---

## 🧪 Demonstration Steps (For College/Viva Demo)

1. **Launch App:** Run `streamlit run app.py`.
2. **Instant Demo:** Click **"Generate & Use Demo Note Samples"** to see automatic alignment of shifted/rotated notes with detected handwritten updates.
3. **Upload Real Samples:** Upload an original note and an edited photo.
4. **Tune Sliders:** Adjust difference threshold and minimum area in the left sidebar to filter out subtle pen pressure differences.
5. **Download Export:** Download the annotated image or full 4-panel comparison strip from the Results tab.
