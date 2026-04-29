# Model training

Training pipeline for the ASL sign classifier. Input: WLASL videos. Output: an `.onnx` file the backend can serve.

## Stack

- Python 3.11
- PyTorch (primary) / TensorFlow + Keras (alternate path for TF.js export)
- MediaPipe Hands (Python) — landmark extraction during preprocessing
- OpenCV — video frame reading
- NumPy / Pandas — data handling
- Jupyter — exploratory notebooks

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** On Apple Silicon, install PyTorch with the appropriate `--index-url` for MPS support. On Linux/CUDA, install the CUDA-matching wheel from pytorch.org.

## Pipeline

```
WLASL videos
    │
    ▼
src/preprocessing/extract_landmarks.py
    │  (MediaPipe Hands → 63-dim per frame)
    ▼
data/processed/*.npy
    │
    ▼
src/preprocessing/build_dataset.py
    │  (filter top-N words, train/val/test split)
    ▼
src/train.py
    │  (PyTorch LSTM training)
    ▼
checkpoints/*.pt
    │
    ▼
src/export_onnx.py
    │  (PyTorch → ONNX)
    ▼
exports/asl_model.onnx
    │
    ▼
copy to backend/models/ for serving
```

## Commands

```sh
# 1. Extract landmarks from raw videos
python -m src.preprocessing.extract_landmarks

# 2. Build train/val/test splits
python -m src.preprocessing.build_dataset

# 3. Train the LSTM
python -m src.train

# 4. Evaluate on the test split
python -m src.evaluate

# 5. Export the trained checkpoint to ONNX
python -m src.export_onnx
```

## Data

The WLASL dataset is **not** committed to this repo (videos are large, and the dataset has its own license). Download it separately and place videos in `data/raw/`.
