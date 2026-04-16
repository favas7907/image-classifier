# CIFAR-10 Image Classifier using SVM and k-NN

A simple image classification project built with classical machine learning techniques on the CIFAR-10 dataset.  
This notebook uses OpenCV for preprocessing and compares two traditional classifiers, Support Vector Machine (SVM) and k-Nearest Neighbors (k-NN), without using any deep learning or GPU-based training.

---

## Project Overview

The goal of this project is to classify CIFAR-10 images into 10 categories using machine learning methods that are easy to understand and run on a normal CPU environment.

Instead of using a neural network, the notebook follows a classic pipeline:

1. Install and import required libraries
2. Download the CIFAR-10 dataset
3. Preprocess images using OpenCV
4. Show a small image enhancement demo
5. Train SVM and k-NN models
6. Evaluate the models with accuracy and confusion matrix
7. Visualize predictions on test images

---

## Dataset

This project uses the **CIFAR-10** dataset, which contains 60,000 images across 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Each image is 32×32 pixels with RGB color channels.

---

## Features

- Classical ML approach with no deep learning
- OpenCV-based preprocessing
- Grayscale conversion and resizing
- Image enhancement demo
- SVM and k-NN model comparison
- Accuracy evaluation
- Confusion matrix visualization
- Prediction previews with correct/incorrect labels

---

## Tech Stack

- Python
- NumPy
- OpenCV
- Matplotlib
- Torchvision
- scikit-learn

---

## How It Works

### 1. Data Loading
The CIFAR-10 dataset is downloaded using `torchvision.datasets.CIFAR10`.

### 2. Preprocessing
Images are converted to grayscale, resized, normalized, and flattened into feature vectors.

### 3. Training
Two classical classifiers are trained:

- **SVM** with an RBF kernel
- **k-NN** with `k=5`

### 4. Evaluation
The notebook prints:

- accuracy score
- classification report
- confusion matrix

### 5. Prediction Visualization
A small grid of test images is shown with predicted and true labels for easy comparison.

---

## Results

With `MAX_SAMPLES = 10,000`, the notebook usually gives results in this range:

- **SVM**: around 40–45%
- **k-NN**: around 35–38%

These scores are normal for classical ML on raw CIFAR-10 images.  
CIFAR-10 is a difficult dataset, and deep learning models usually perform much better.

---

## Why the Accuracy Is Limited

This project uses flattened grayscale images, which means a lot of visual information is lost.

Classical models like SVM and k-NN do not understand image structure the way CNNs do, so their performance is naturally lower on this dataset.

---

## Possible Improvements

A few upgrades that could improve the notebook:

- use HOG features instead of raw pixels
- add PCA for dimensionality reduction
- tune SVM hyperparameters with GridSearchCV
- compare with Random Forest or Logistic Regression
- increase the sample size to 30,000 or full 60,000
- try larger image sizes like 64×64

---

## Folder Structure

```bash
cifar10_classifier.ipynb
README.md