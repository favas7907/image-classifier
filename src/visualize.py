"""
visualize.py
------------
Two visualisation utilities:

  show_predictions()
      Grid of 16 test images (4 × 4).
      Title colour: green = correct prediction, red = wrong.
      Saved to outputs/predictions.png

  show_enhancement_comparison()
      Side-by-side panel: Original | Enhanced | Blurred.
      Saved to outputs/enhancement_comparison.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # works on headless / GUI-less environments
import matplotlib.pyplot as plt
import cv2

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

OUTPUT_DIR = 'outputs'
DISPLAY_SIZE = 256    # px used when upscaling images for the enhancement demo


# ── Predictions grid ──────────────────────────────────────────────────────────
def show_predictions(model, X_raw: np.ndarray, X_processed: np.ndarray,
                     y_true: np.ndarray, n: int = 16,
                     model_name: str = 'SVM') -> None:
    """
    Save a 4 × 4 grid of test images annotated with predicted vs actual labels.

    Parameters
    ----------
    model        : fitted sklearn estimator
    X_raw        : np.ndarray  raw RGB images  shape (M, 32, 32, 3)
    X_processed  : np.ndarray  preprocessed feature vectors  shape (M, 1024)
    y_true       : np.ndarray  true integer labels  shape (M,)
    n            : int  number of images to display (must be a perfect square)
    model_name   : str  used in the title and filename
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    y_pred = model.predict(X_processed)

    # Pick n random indices
    rng = np.random.default_rng(seed=0)
    indices = rng.choice(len(X_raw), size=n, replace=False)

    cols = int(np.sqrt(n))
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.8))
    fig.suptitle(
        f'{model_name} Predictions  —  '
        'green = correct  |  red = wrong',
        fontsize=13,
    )

    for plot_idx, img_idx in enumerate(indices):
        ax = axes[plot_idx // cols][plot_idx % cols]

        img = X_raw[img_idx]
        actual_label    = CIFAR10_CLASSES[y_true[img_idx]]
        predicted_label = CIFAR10_CLASSES[y_pred[img_idx]]
        is_correct = (actual_label == predicted_label)

        ax.imshow(img)
        ax.set_title(
            f'Pred : {predicted_label}\nTrue : {actual_label}',
            color='green' if is_correct else 'red',
            fontsize=8.5,
        )
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUTPUT_DIR, f'{model_name}_predictions.png')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Predictions grid saved → {out_path}")


# ── Enhancement comparison ────────────────────────────────────────────────────
def show_enhancement_comparison(gray: np.ndarray, enhanced: np.ndarray,
                                 blurred: np.ndarray,
                                 title: str = 'Image Enhancement') -> None:
    """
    Save a three-panel figure comparing the enhancement stages.

    Parameters
    ----------
    gray     : grayscale original        (uint8, H × W)
    enhanced : brightness/contrast tuned (uint8, H × W)
    blurred  : Gaussian-blurred          (uint8, H × W)
    title    : figure super-title
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    panel_labels = [
        'Original\n(Grayscale)',
        'Enhanced\n(Brightness +40, Contrast ×1.5)',
        'Blurred\n(Gaussian 7×7)',
    ]
    panels = [gray, enhanced, blurred]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(title, fontsize=14, y=1.02)

    for ax, label, panel in zip(axes, panel_labels, panels):
        ax.imshow(panel, cmap='gray', vmin=0, vmax=255)
        ax.set_title(label, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'enhancement_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Enhancement comparison saved → {out_path}")

    # Also write a raw OpenCV side-by-side for quick inspection
    raw_side_by_side = np.hstack([gray, enhanced, blurred])
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'enhancement_raw.png'), raw_side_by_side)