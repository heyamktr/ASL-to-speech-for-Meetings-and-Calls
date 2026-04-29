"""ONNX Runtime model loading and prediction."""

import numpy as np


class SignClassifier:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        # TODO: create onnxruntime.InferenceSession, warm up with dummy input

    def predict(self, landmarks: np.ndarray) -> str:
        """Run inference on a (30, 63) sliding window. Returns the predicted word."""
        # TODO: implement
        raise NotImplementedError
