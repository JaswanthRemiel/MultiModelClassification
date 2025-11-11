# MultiModelClassification

![Banner](https://github.com/user-attachments/assets/960508ca-9df5-4d94-9e6b-80773ea47c4b)

A multimodal classification project that combines text and image inputs to predict a target class. This repository includes ready-to-use pre-trained models, a simple application interface, and Jupyter/py notebooks for experimentation and evaluation.

- Language composition: Jupyter Notebook (99.4%), Python (0.6%)
- Models included:
  - Text: BiLSTM + Attention (`text_classification_model.h5`)
  - Vision: ResNet50 (`image_classification_model.pt`)
  - Tokenizer: Pre-fitted tokenizer for the text pipeline (`tokenizer.joblib`)

Tip: This README includes Jekyll front matter and is optimized for the GitHub Pages Slate theme. When you enable GitHub Pages with the Slate theme, this page will render as your site’s homepage.

---

## Table of contents

- [Features](#features)
- [Repository structure](#repository-structure)
- [Quickstart](#quickstart)
  - [1) Clone and environment](#1-clone-and-environment)
  - [2) Install dependencies](#2-install-dependencies)
  - [3) Run the app](#3-run-the-app)
  - [4) Run the notebooks](#4-run-the-notebooks)
- [Data expectations](#data-expectations)
- [Models and architecture](#models-and-architecture)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Citations and acknowledgments](#citations-and-acknowledgments)
- [Publish with GitHub Pages (Slate theme)](#publish-with-github-pages-slate-theme)

---

## Features

- Multimodal classification by fusing:
  - Visual signal from a ResNet50 encoder
  - Textual signal from a BiLSTM + Attention encoder
- Pre-trained weights included for fast, reproducible inference
- Simple application interface (`app.py`) to try the model end-to-end
- Notebook-first workflow (EDA, inference, and evaluation)
- Clean separation of assets (models, tokenizer) and notebooks

---

## Repository structure

```text
MultiModelClassification/
├── NotebooksPY/                   # Jupyter/Python notebooks for experimentation
├── app.py                         # Main application interface (run locally)
├── image_classification_model.pt  # Pretrained ResNet50 image classification model
├── text_classification_model.h5   # Pretrained BiLSTM+Attention text classification model
├── tokenizer.joblib               # Pre-fitted tokenizer for text preprocessing
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation (renders on GitHub Pages)
```

Notes:
- The included model artifacts enable immediate local inference without training.
- Large data should not be committed; point notebooks to your local data directory.

---

## Quickstart

### 1) Clone and environment

```bash
git clone https://github.com/JaswanthRemiel/MultiModelClassification.git
cd MultiModelClassification

# Create and activate a virtual environment (choose one)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
# or
conda create -n mmc python=3.10 -y && conda activate mmc
```

### 2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you need GPU acceleration for PyTorch, ensure the wheel matches your CUDA version (see PyTorch Get Started for the correct index-url). CPU-only is fine for smaller demos.

### 3) Run the app

The repository includes a main application interface in `app.py`. Depending on how the app is implemented, use one of the following:

- If it’s a standard Python script (CLI or simple server):
  ```bash
  python app.py
  ```
  Then open the printed local URL (commonly http://127.0.0.1:5000 or http://127.0.0.1:8000).

- If it’s a Streamlit app:
  ```bash
  pip install streamlit  # if not already installed
  streamlit run app.py
  ```
  Then visit the local Streamlit URL (commonly http://localhost:8501).

Tip: If the app prints usage help, run `python app.py --help`.

### 4) Run the notebooks

Launch Jupyter and open the notebooks inside `NotebooksPY/`:

```bash
jupyter lab
# or
jupyter notebook
```

Inside each notebook, adjust:
- Paths to your data and images
- Model checkpoint paths (defaults point to repo root)
- Batch size, device (cpu/cuda), and other hyperparameters as needed

---

## Data expectations

The typical multimodal row contains paired text and an image path. A common format is a CSV:

```csv
id,text,image_path,label
0001,"Short description for the image","data/images/0001.jpg","class_a"
0002,"Another description","data/images/0002.jpg","class_b"
```

Recommended layout:

```text
data/
  images/
    0001.jpg
    0002.jpg
  train.csv
  val.csv
  test.csv
```

- image_path can be absolute or relative to the notebook/app working directory.
- For multilabel targets, you can store a delimiter-separated string or adapt the notebook to multi-hot labels.

---

## Models and architecture

- Text encoder: BiLSTM + Attention
  - Pretrained weights: `text_classification_model.h5`
  - Tokenizer: `tokenizer.joblib` (ensure you use this tokenizer for consistent preprocessing)

- Image encoder: ResNet50
  - Pretrained weights: `image_classification_model.pt`

- Fusion/classifier:
  - Fusion and classification occur in the application/notebooks. Typical approaches include:
    - Concatenation of pooled text and image embeddings + MLP head
    - Late fusion (e.g., weighted voting) for quick baselines

Tips:
- Keep the tokenizer and text model versions in sync.
- If you fine-tune or retrain, save new checkpoints under a `models/` folder and update paths in the app/notebooks.

---

## Configuration

Common configuration points (set in notebooks or at the top of `app.py`):
- Paths: data root, models, tokenizer
- Device: `cpu` or `cuda`
- Inference settings: batch size, max tokens, image resize/crop dimensions
- Label mapping: class names and ids

---

## Evaluation

Typical metrics for classification:
- Accuracy, Precision, Recall, F1 (macro/micro/weighted)
- ROC-AUC (binary, or one-vs-rest for multiclass)
- Confusion matrix and per-class breakdown

Most notebooks include cells to compute and visualize these. Save your runs’ metrics to CSV/JSON for comparison across configurations.

---

## Reproducibility

Set seeds and control non-determinism when benchmarking:

```python
import os, random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

Keep notes of:
- Exact model/tag names and versions
- Dataset splits and preprocessing
- Environment details (Python, CUDA, CUDA driver)

---

## Troubleshooting

- CUDA / GPU issues:
  - Ensure the installed PyTorch build matches your CUDA driver
  - Try CPU-only wheels if you don’t need GPU or are testing quickly
- Out-of-memory:
  - Reduce batch size, image resolution, or use mixed precision (AMP)
- Tokenization mismatches:
  - Always use the provided `tokenizer.joblib` with `text_classification_model.h5`
- File not found:
  - Verify relative paths from the current working directory (Jupyter vs. app differs)

---

## Roadmap

- Add configurable fusion strategies (concat + MLP, gated fusion, cross-attention)
- Provide a Colab notebook for zero-setup demos
- Add model cards and training notes for included checkpoints
- Optional CLI for batch inference on folders/CSVs

---

## Contributing

Contributions are welcome! You can:
- Open issues for bugs or feature requests
- Submit PRs that improve the notebooks, app UX, or documentation
- Share datasets/configs that highlight new use cases

Please include a clear description, reproduction steps (if applicable), and relevant logs/screenshots.

---

## Citations and acknowledgments

If you use this project, please consider citing the libraries and models you rely on:
- PyTorch
- torchvision
- scikit-learn
- Keras/TensorFlow (for the BiLSTM + Attention implementation if applicable)
- Any pre-trained backbones and datasets used in your experiments

---
