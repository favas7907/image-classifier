"""
train.py
--------
Trains two classical ML classifiers on preprocessed image features:

  1. Support Vector Machine (SVM)
     - Kernel : RBF  (handles non-linear boundaries well)
     - C      : 10   (regularisation; higher = less regularised)
     - gamma  : 'scale'  (1 / (n_features * X.var()))

  2. k-Nearest Neighbours (k-NN)
     - k      : 5    (majority vote among 5 nearest neighbours)
     - metric : minkowski (Euclidean by default, p=2)
     - n_jobs : -1   (use all CPU cores)

Both models receive the same flat feature vectors produced by preprocess.py.
"""

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def train_models(X_train: np.ndarray, y_train: np.ndarray):
    """
    Fit SVM and k-NN classifiers on the training data.

    Parameters
    ----------
    X_train : np.ndarray  shape (N, n_features)
    y_train : np.ndarray  shape (N,)  integer class labels

    Returns
    -------
    svm : fitted SVC instance
    knn : fitted KNeighborsClassifier instance
    """

    # ── SVM ───────────────────────────────────────────────────────────────────
    print("  [SVM] Training SVM with RBF kernel …  (may take 2-5 min on CPU)")
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        random_state=42,
        verbose=False,
    )
    svm.fit(X_train, y_train)
    print("  [SVM] Training complete.")

    # ── k-NN ──────────────────────────────────────────────────────────────────
    print("  [k-NN] Training k-NN (k=5) …")
    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric='minkowski',
        n_jobs=-1,          # parallelise distance computation
    )
    knn.fit(X_train, y_train)
    print("  [k-NN] Training complete.")

    return svm, knn