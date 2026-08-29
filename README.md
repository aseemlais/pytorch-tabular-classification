# 🌾 Rice Type Classification using PyTorch

A deep learning project for binary classification of rice samples using a fully connected neural network built with **PyTorch**. The project demonstrates an end-to-end workflow for tabular data, from preprocessing and model training to evaluation and inference.

## 📌 Overview

The dataset contains morphological measurements of rice grains. The model uses 10 numerical features to classify each sample into one of two classes.

### Features

- Area
- MajorAxisLength
- MinorAxisLength
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Perimeter
- Roundness
- AspectRation

**Target:** `Class`

## 🛠️ Tech Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Conda
- Git/GitHub

## 🔄 Workflow

```text
Raw Dataset
    ↓
Data Cleaning
    ↓
Train / Validation / Test Split
    ↓
StandardScaler
    ↓
PyTorch Tensors
    ↓
Dataset & DataLoader
    ↓
Neural Network
    ↓
Training & Validation
    ↓
Test Evaluation
    ↓
Model & Scaler Saving
    ↓
Inference
```

## 🧠 Model Architecture

A simple fully connected neural network is used:

```text
Input: 10 features
        ↓
Linear(10 → 10)
        ↓
Linear(10 → 1)
        ↓
Sigmoid
        ↓
Binary Prediction
```

The model contains **121 trainable parameters**.

### Training

- Loss Function: Binary Cross Entropy (`BCELoss`)
- Optimizer: Adam
- Learning Rate: `0.001`
- Batch Size: `32`
- Epochs: `10`

## 📊 Data Split

| Dataset | Samples |
|---|---:|
| Training | 14,548 |
| Validation | 1,818 |
| Test | 1,819 |
| **Total** | **18,185** |

The `StandardScaler` is fitted only on the training data and then applied to validation and test data to avoid data leakage.

## 📈 Results

| Metric | Result |
|---|---:|
| Validation Accuracy | ~99.39% |
| Test Loss | 0.0303 |
| Test Accuracy | ~99.12% |

The model was also evaluated using:

- Confusion Matrix
- Precision
- Recall
- F1-Score

The test confusion matrix showed a small number of false positives and false negatives, indicating strong classification performance.

## 💾 Saved Model

The trained model and preprocessing scaler are saved in:

```text
models/
├── rice_classifier.pth
└── scaler.pkl
```

- `rice_classifier.pth` → trained PyTorch model parameters
- `scaler.pkl` → fitted `StandardScaler`

## 📁 Project Structure

```text
pytorch-tabular-classification/
│
├── data/
│   └── raw/                  # Dataset (ignored by Git)
│
├── models/
│   ├── rice_classifier.pth
│   └── scaler.pkl
│
├── notebooks/
│
├── src/
│   ├── data_loader.py
│   ├── dataset.py
│   ├── model.py
│   ├── training.py
│   ├── evaluate.py
│   └── visualization.py
│
├── main.py
├── predict.py
├── environment.yml
├── .gitignore
└── README.md
```

## 🚀 Installation & Usage

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
conda activate pytorch-tabular
```

### 3. Train and evaluate the model

```bash
python main.py
```

This performs preprocessing, training, validation, test evaluation, visualization, and model/scaler saving.

### 4. Run inference

```bash
python predict.py
```

## 🔐 Notes

- The raw dataset is excluded from Git using `.gitignore`.
- Kaggle API credentials should never be committed to the repository.
- The current implementation runs on CPU. GPU/CUDA support can be added when a CUDA-enabled PyTorch installation and compatible NVIDIA GPU are available.

## 🚀 Future Improvements

- Improve the neural network architecture.
- Add early stopping and learning-rate scheduling.
- Build a user-friendly prediction interface.
- Compare different PyTorch architectures.
- Deploy the model using FastAPI or Streamlit.

## 📚 Learning Outcomes

This project provided hands-on practice with:

- PyTorch tensors
- `Dataset` and `DataLoader`
- Neural network construction with `nn.Module`
- Forward propagation
- Loss functions
- Backpropagation
- Adam optimization
- Training vs. evaluation modes
- `torch.no_grad()`
- Model serialization with `state_dict()`
- Model evaluation metrics
- End-to-end inference pipeline