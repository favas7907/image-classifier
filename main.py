"""
main.py
-------
Entry point for the CIFAR-10 Image Classification project.

Pipeline
--------
1.  Download CIFAR-10 via torchvision (no TensorFlow needed)
2.  Merge train + test splits, then re-split 80 / 20
3.  Preprocess with OpenCV  (grayscale → resize → normalize → flatten)
4.  Train SVM and k-NN classifiers
5.  Evaluate both models  (accuracy + confusion matrix)
6.  Visualise predictions grid
7.  Show image enhancement demo

Quick start
-----------
    python main.py                  # uses default MAX_SAMPLES = 10 000
    python main.py --samples 5000   # faster, lower accuracy
    python main.py --samples -1     # full 60 000 images  (very slow for SVM)
"""

import argparse
import numpy as np
import cv2

from torchvision import datasets
from sklearn.model_selection import train_test_split

from src.preprocess import preprocess, enhance_image
from src.train      import train_models
from src.evaluate   import evaluate_model
from src.visualize  import show_predictions, show_enhancement_comparison


# ── CLI argument ───────────────────────────────────────────────────────────────
def _parse_args():
    parser = argparse.ArgumentParser(
        description='CIFAR-10 image classifier: SVM + k-NN via OpenCV features'
    )
    parser.add_argument(
        '--samples', type=int, default=10_000,
        help=(
            'Max number of images to use (default 10 000). '
            'Use -1 for the full 60 000-image dataset (slow for SVM).'
        ),
    )
    return parser.parse_args()


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main():
    args = _parse_args()
    MAX_SAMPLES = args.samples   # cap dataset size for manageable runtime

    # ==========================================================================
    # 1. LOAD CIFAR-10  (downloaded once, cached in ./data)
    # ==========================================================================
    print("\n[1/6] Downloading / loading CIFAR-10 dataset …")
    train_data = datasets.CIFAR10(root='./data', train=True,  download=True)
    test_data  = datasets.CIFAR10(root='./data', train=False, download=True)

    # Convert PIL images → numpy arrays
    X_train_raw = np.array([np.array(img) for img, _ in train_data])
    y_train_raw = np.array([lbl              for _, lbl in train_data])
    X_test_raw  = np.array([np.array(img) for img, _ in test_data])
    y_test_raw  = np.array([lbl              for _, lbl in test_data])

    # Merge into one pool, then re-split reproducibly
    X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)
    y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)

    # Optionally subsample (stratified so all 10 classes are present)
    if MAX_SAMPLES > 0 and MAX_SAMPLES < len(X_all):
        _, X_all, _, y_all = train_test_split(
            X_all, y_all,
            test_size=MAX_SAMPLES,
            stratify=y_all,
            random_state=42,
        )
        print(f"  Using {MAX_SAMPLES:,} images (stratified subsample).")
    else:
        print(f"  Using full dataset: {len(X_all):,} images.")

    print(f"  Dataset shape : {X_all.shape}  |  Labels shape : {y_all.shape}")

    # ==========================================================================
    # 2. PREPROCESSING  (OpenCV pipeline)
    # ==========================================================================
    print("\n[2/6] Preprocessing images (grayscale → resize → normalise) …")
    X = preprocess(X_all)   # shape → (N, 1024)

    # 80 / 20 train-test split, stratified by class
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_all, np.arange(len(X_all)),
        test_size=0.2,
        random_state=42,
        stratify=y_all,
    )
    X_test_raw_subset = X_all[idx_test]   # keep raw images for visualisation

    print(f"  Train : {X_train.shape[0]:,} samples")
    print(f"  Test  : {X_test.shape[0]:,} samples")

    # ==========================================================================
    # 3. TRAIN MODELS
    # ==========================================================================
    print("\n[3/6] Training classifiers …")
    svm, knn = train_models(X_train, y_train)

    # ==========================================================================
    # 4. EVALUATE MODELS
    # ==========================================================================
    print("\n[4/6] Evaluating models …")
    svm_acc, _ = evaluate_model(svm, X_test, y_test, "SVM")
    knn_acc, _ = evaluate_model(knn, X_test, y_test, "KNN")

    # Summary
    print(f"\n  ┌─────────────────────────────┐")
    print(f"  │  SVM accuracy : {svm_acc*100:5.2f}%       │")
    print(f"  │  k-NN accuracy: {knn_acc*100:5.2f}%       │")
    print(f"  └─────────────────────────────┘")

    # ==========================================================================
    # 5. VISUALISE PREDICTIONS
    # ==========================================================================
    print("\n[5/6] Generating predictions grid …")
    show_predictions(svm, X_test_raw_subset, X_test, y_test,
                     n=16, model_name='SVM')
    show_predictions(knn, X_test_raw_subset, X_test, y_test,
                     n=16, model_name='KNN')

    # ==========================================================================
    # 6. IMAGE ENHANCEMENT DEMO
    # ==========================================================================
    print("\n[6/6] Generating image enhancement demo …")
    sample_img = X_all[0]                       # one raw RGB image
    gray, enhanced, blurred = enhance_image(sample_img)
    show_enhancement_comparison(gray, enhanced, blurred,
                                title='CIFAR-10 Image Enhancement Demo')

    # Optional: try to display with OpenCV (skipped gracefully if headless)
    try:
        combined = np.hstack([gray, enhanced, blurred])
        cv2.imshow("Original  |  Enhanced  |  Blurred  (press any key)", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("  (cv2.imshow not available in this environment — file saved instead)")

    print("\n✅ All done!  Check the outputs/ folder for results.\n")


if __name__ == '__main__':
    main()