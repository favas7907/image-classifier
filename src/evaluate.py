"""
evaluate.py
-----------
Evaluates a trained classifier and saves results to outputs/.

Metrics produced
----------------
- Accuracy (overall % correct)
- Per-class precision, recall, F1-score  (classification_report)
- Confusion matrix heatmap (saved as PNG)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — works on all systems
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# CIFAR-10 human-readable class names (index = label integer)
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

OUTPUT_DIR = 'outputs'


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str):
    """
    Evaluate *model* on the test split and save a confusion matrix PNG.

    Parameters
    ----------
    model      : fitted sklearn estimator with a .predict() method
    X_test     : np.ndarray  shape (N, n_features)
    y_test     : np.ndarray  shape (N,)
    model_name : str  used in titles and filenames  (e.g. "SVM", "KNN")

    Returns
    -------
    accuracy : float  (0 – 1)
    y_pred   : np.ndarray  predicted labels
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Predictions ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_test, y_pred)

    # ── Console output ────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  {model_name}  Results")
    print(f"{'='*50}")
    print(f"  Accuracy : {accuracy * 100:.2f}%\n")
    print(classification_report(
        y_test, y_pred,
        target_names=CIFAR10_CLASSES,
        digits=3,
    ))

    # ── Confusion matrix heatmap ──────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    _save_confusion_matrix(cm, model_name, accuracy)

    return accuracy, y_pred


# ── Private helpers ────────────────────────────────────────────────────────────
def _save_confusion_matrix(cm: np.ndarray, model_name: str,
                            accuracy: float) -> None:
    """Draw and save a colour-coded confusion matrix as a PNG file."""
    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw heatmap manually (no seaborn dependency)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate each cell with its integer count
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha='center', va='center', fontsize=8,
                color='white' if cm[i, j] > thresh else 'black',
            )

    ticks = np.arange(len(CIFAR10_CLASSES))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=9)

    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(
        f'{model_name} — Confusion Matrix  '
        f'(Accuracy: {accuracy * 100:.2f}%)',
        fontsize=13, pad=14,
    )

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f'{model_name}_confusion_matrix.png')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {out_path}")