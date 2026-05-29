"""ONNX Runtime model loading and prediction."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

RAW_FEATURE_DIM = 144
MODEL_INPUT_DIM = 292


def _normalize_hand(coords: np.ndarray, detected: np.ndarray) -> np.ndarray:
    if not detected.any():
        return coords

    wrist = coords[detected, 0:1, :]
    coords[detected] -= wrist

    hand_size = np.linalg.norm(coords[detected, 9, :], axis=1)
    valid = hand_size > 0
    det_idx = np.where(detected)[0]
    coords[det_idx[valid]] /= hand_size[valid, np.newaxis, np.newaxis]
    return coords


def normalize_window(window: np.ndarray) -> np.ndarray:
    """Apply the same raw 144-dim normalization used during dataset building."""
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != RAW_FEATURE_DIM:
        raise ValueError(f"landmark window must have shape (frames, {RAW_FEATURE_DIM})")

    frames = arr.shape[0]
    right_hand = arr[:, 0:63].reshape(frames, 21, 3).copy()
    left_hand = arr[:, 63:126].reshape(frames, 21, 3).copy()
    pose = arr[:, 126:144].reshape(frames, 6, 3).copy()

    right_detected = arr[:, 0:63].any(axis=1)
    left_detected = arr[:, 63:126].any(axis=1)
    pose_detected = arr[:, 126:144].any(axis=1)

    right_hand = _normalize_hand(right_hand, right_detected)
    left_hand = _normalize_hand(left_hand, left_detected)

    if pose_detected.any():
        left_shoulder = pose[pose_detected, 0, :]
        right_shoulder = pose[pose_detected, 1, :]
        midpoint = (left_shoulder + right_shoulder) / 2
        pose[pose_detected] -= midpoint[:, np.newaxis, :]

        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
        valid = shoulder_width > 0
        det_idx = np.where(pose_detected)[0]
        pose[det_idx[valid]] /= shoulder_width[valid, np.newaxis, np.newaxis]

    return np.concatenate(
        [
            right_hand.reshape(frames, 63),
            left_hand.reshape(frames, 63),
            pose.reshape(frames, 18),
        ],
        axis=1,
    ).astype(np.float32)


def build_model_features(window: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert a raw (frames, 144) window into ONNX inputs.

    Output features have shape (1, seq_len, 292):
      144 normalized coordinates + 2 hand-presence bits + 146 velocity dims.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    normalized = normalize_window(window)
    if normalized.shape[0] > seq_len:
        normalized = normalized[-seq_len:]

    length = normalized.shape[0]
    padded = np.zeros((seq_len, RAW_FEATURE_DIM), dtype=np.float32)
    padded[:length] = normalized

    right_present = padded[:, 0:63].any(axis=1, keepdims=True).astype(np.float32)
    left_present = padded[:, 63:126].any(axis=1, keepdims=True).astype(np.float32)
    positions = np.concatenate(
        [
            padded[:, 0:63],
            right_present,
            padded[:, 63:126],
            left_present,
            padded[:, 126:144],
        ],
        axis=1,
    ).astype(np.float32)

    velocity = np.diff(positions, axis=0, prepend=positions[:1])
    features = np.concatenate([positions, velocity], axis=1).astype(np.float32)
    if features.shape[1] != MODEL_INPUT_DIM:
        raise ValueError(f"model features must have {MODEL_INPUT_DIM} values per frame")

    lengths = np.array([max(1, length)], dtype=np.int64)
    return features[np.newaxis, :, :], lengths


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)

#I wired this class into the FastAPI app in main.py, where it's initialized on startup 
#and used for predictions in the websocket endpoint nè chị Uyên
class SignClassifier:
    def __init__(
        self,
        model_path: str,
        label_map_path: str,
        seq_len: int = 100,
        confidence_threshold: float = 0.4,
    ) -> None:
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        self.seq_len = seq_len
        self.confidence_threshold = confidence_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}") 

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"], 
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        self.labels = self._load_labels()
        self._warm_up()

    def _load_labels(self) -> list[str]:
        if not self.label_map_path.exists():
            output_shape = self.session.get_outputs()[0].shape
            num_classes = output_shape[-1] if isinstance(output_shape[-1], int) else 0
            return [str(idx) for idx in range(num_classes)]

        with open(self.label_map_path) as f:
            raw = json.load(f)

        max_id = max(int(key) for key in raw)
        return [raw.get(str(idx), str(idx)) for idx in range(max_id + 1)]

    def _warm_up(self) -> None:
        dummy = np.zeros((self.seq_len, RAW_FEATURE_DIM), dtype=np.float32)
        features, lengths = build_model_features(dummy, self.seq_len)
        self._run(features, lengths)

    def _run(self, features: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        inputs = {
            self.input_names[0]: features,
            self.input_names[1]: lengths,
        }
        return self.session.run([self.output_name], inputs)[0]

    def predict(self, landmarks: np.ndarray) -> tuple[str, float]:
        """Run inference on a raw (frames, 144) sliding window."""
        features, lengths = build_model_features(landmarks, self.seq_len)
        logits = self._run(features, lengths)
        probabilities = _softmax(logits)[0]

        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        prediction = self.labels[class_id] if class_id < len(self.labels) else str(class_id)

        if confidence < self.confidence_threshold:
            return "uncertain", confidence
        return prediction, confidence
