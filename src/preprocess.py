"""
preprocess.py
-------------
Handles all image preprocessing and enhancement using OpenCV.

Steps:
  1. Convert RGB → Grayscale
  2. Resize to a fixed dimension (IMG_SIZE x IMG_SIZE)
  3. Normalize pixel values to [0, 1]
  4. Flatten each image into a 1-D feature vector (for ML models)

Also provides enhance_image() for the visual demo:
  - Brightness & contrast adjustment  (cv2.convertScaleAbs)
  - Gaussian blur                      (cv2.GaussianBlur)
"""

import cv2
import numpy as np

# ── Fixed image size used throughout the project ──────────────────────────────
IMG_SIZE = 32          # 32 × 32 pixels  (matches CIFAR-10 native size)
DISPLAY_SIZE = 256     # upscaled size used only for the enhancement demo


# ── Main preprocessing pipeline ───────────────────────────────────────────────
def preprocess(images: np.ndarray) -> np.ndarray:
    """
    Convert a batch of raw RGB images into normalised, flattened feature vectors.

    Parameters
    ----------
    images : np.ndarray
        Array of shape (N, H, W, 3) with dtype uint8 and values in [0, 255].

    Returns
    -------
    np.ndarray
        Array of shape (N, IMG_SIZE * IMG_SIZE) with dtype float32 in [0, 1].
    """
    processed = []

    for img in images:
        # Step 1 – Grayscale conversion (reduces features, speeds up training)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Step 2 – Resize to fixed dimension so every sample has the same shape
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_AREA)

        # Step 3 – Normalize to [0, 1]  (helps SVM / k-NN converge faster)
        normalized = resized.astype(np.float32) / 255.0

        # Step 4 – Flatten 2-D image → 1-D feature vector
        processed.append(normalized.flatten())

    return np.array(processed, dtype=np.float32)


# ── Enhancement demo ───────────────────────────────────────────────────────────
def enhance_image(img: np.ndarray):
    """
    Apply three common image-enhancement operations to a single RGB image.

    Operations
    ----------
    1. Grayscale                  – baseline / original
    2. Brightness + Contrast      – cv2.convertScaleAbs(alpha=1.5, beta=40)
    3. Gaussian Blur              – 7 × 7 kernel  (low-pass / noise-reduction)

    Parameters
    ----------
    img : np.ndarray
        Single RGB image, shape (H, W, 3).

    Returns
    -------
    gray, enhanced, blurred : np.ndarray  (each shape DISPLAY_SIZE × DISPLAY_SIZE)
    """
    # --- Convert to grayscale and upscale for better visual clarity -----------
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (DISPLAY_SIZE, DISPLAY_SIZE),
                      interpolation=cv2.INTER_NEAREST)

    # --- Brightness & contrast adjustment ------------------------------------
    #   alpha > 1  → higher contrast
    #   beta  > 0  → brighter image
    enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=40)

    # --- Gaussian blur (smoothing / noise reduction) -------------------------
    blurred = cv2.GaussianBlur(gray, (7, 7), sigmaX=0)

    return gray, enhanced, blurred