# 🖼️ CIFAR-10 Image Classifier — SVM & k-NN with OpenCV

A beginner-friendly image classification project that uses **OpenCV** for
preprocessing and two classical ML models — **SVM** and **k-NN** — to classify
images from the CIFAR-10 dataset.  No deep learning, no GPU required.

---

## 📁 Project Structure

```
image-classifier/
│
├── main.py                  # Entry point — runs the full pipeline
│
├── src/
│   ├── __init__.py          # Makes src/ a Python package
│   ├── preprocess.py        # OpenCV: grayscale, resize, normalize, enhance
│   ├── train.py             # Train SVM and k-NN classifiers
│   ├── evaluate.py          # Accuracy, classification report, confusion matrix
│   └── visualize.py         # Predictions grid & enhancement comparison plots
│
├── data/                    # CIFAR-10 downloaded here automatically
├── outputs/                 # All saved PNG results land here
│
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

**CIFAR-10** — 60 000 colour images across 10 classes:

| Label | Class       | Label | Class      |
|-------|-------------|-------|------------|
| 0     | airplane    | 5     | dog        |
| 1     | automobile  | 6     | frog       |
| 2     | bird        | 7     | horse      |
| 3     | cat         | 8     | ship       |
| 4     | deer        | 9     | truck      |

Each image is **32 × 32 pixels, RGB**.  
The dataset is downloaded automatically on first run via `torchvision.datasets.CIFAR10`.

> ⚠️ By default the script uses **10 000 images** (stratified subsample) to keep
> training time under ~5 minutes on a laptop.  Pass `--samples -1` to use the
> full 60 000.

---

## ⚙️ Preprocessing Steps

All preprocessing is done with **OpenCV** inside `src/preprocess.py`:

| Step | Operation | Reason |
|------|-----------|--------|
| 1 | `cv2.cvtColor(img, COLOR_RGB2GRAY)` | Reduces feature space from 3072 to 1024 |
| 2 | `cv2.resize(img, (32, 32))` | Standardises all images to the same size |
| 3 | `img / 255.0` | Normalises pixel values to **[0, 1]** for stable ML training |
| 4 | `.flatten()` | Converts 2-D array → 1-D feature vector for SVM / k-NN |

### Image Enhancement (Demo Only)

`enhance_image()` applies three techniques to a sample image for visual comparison:

| Technique | OpenCV call | Effect |
|-----------|-------------|--------|
| Brightness + Contrast | `cv2.convertScaleAbs(alpha=1.5, beta=40)` | Increases contrast (×1.5), brightens (+40) |
| Gaussian Blur | `cv2.GaussianBlur(img, (7,7), 0)` | Smooths noise |

---

## 🤖 Model Choice

### Support Vector Machine (SVM)
- **Kernel:** RBF (Radial Basis Function) — handles non-linear class boundaries
- **C = 10** — moderate regularisation
- **gamma = 'scale'** — automatic scaling per feature variance
- Typically achieves **~40–45% accuracy** on 10 000 samples of CIFAR-10 grayscale

### k-Nearest Neighbours (k-NN)
- **k = 5** — majority vote among 5 closest training images
- **Metric:** Euclidean distance
- Fast to train, slower to predict on large datasets
- Typically achieves **~35–38% accuracy** on 10 000 samples of CIFAR-10 grayscale

> **Why not higher accuracy?**  
> CIFAR-10 is a genuinely difficult dataset.  Classical ML on raw grayscale pixels
> tops out around 40–50%.  Deep CNNs achieve 90%+.  This project is intentionally
> classical and educational.

---

## 📊 Evaluation Results

_(example run with `--samples 10000`)_

```
SVM  Accuracy : ~42%
k-NN Accuracy : ~36%
```

Output files saved to `outputs/`:

| File | Description |
|------|-------------|
| `SVM_confusion_matrix.png` | 10 × 10 heatmap for SVM |
| `KNN_confusion_matrix.png` | 10 × 10 heatmap for k-NN |
| `SVM_predictions.png` | 4 × 4 grid: predicted vs actual labels |
| `KNN_predictions.png` | 4 × 4 grid: predicted vs actual labels |
| `enhancement_comparison.png` | Side-by-side enhancement panels |
| `enhancement_raw.png` | Raw OpenCV horizontal stack |

---

## 🚀 How to Run

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/image-classifier.git
cd image-classifier
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> On a headless server (no display), replace `opencv-python` with
> `opencv-python-headless` in `requirements.txt`.

### 4. Run the project

```bash
# Default — uses 10 000 images, fast (~3-5 min on laptop)
python main.py

# Use fewer images for a quick smoke test (~1 min)
python main.py --samples 3000

# Use the full 60 000 images (SVM may take 30+ min)
python main.py --samples -1
```

### 5. View results

Open the `outputs/` folder — all PNGs are saved there automatically.

---

## 📋 Requirements Summary

```
numpy, scikit-learn, scipy
opencv-python
torch, torchvision       (for CIFAR-10 download only — no GPU used)
matplotlib
```

---

## 💡 Tips for Improvement

| Idea | How |
|------|-----|
| Add HOG features | `skimage.feature.hog()` before training |
| Use colour histograms | Keep all 3 channels, bin pixel values |
| Try a Random Forest | Drop-in replacement in `train.py` |
| Increase sample size | `--samples 30000` for better accuracy |
| PCA dimensionality reduction | `sklearn.decomposition.PCA` before SVM |

---

## 📄 License

MIT — free to use, modify, and distribute.