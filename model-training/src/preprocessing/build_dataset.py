"""Build train/val/test splits from extracted landmarks.

Reads individual per-video .npy files from data/processed/, trims dead frames,
drops low-quality clips, normalizes, and writes data/splits/*.npz files ready
for the PyTorch DataLoader in train.py.

Output layout
-------------
data/splits/
    train.npz   — arrays X (N, STORED_LEN, 144), y (N,), lengths (N,)
    val.npz
    test.npz
    label_map.json   — {label_int: gloss_str}
    meta.json        — stored_len, num_classes, per-split counts

`lengths[i]` is the number of real (non-padding) frames in X[i]. The training
DataLoader uses this to apply random temporal crops and mask attention on padding.
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
SPLITS_DIR = _REPO_ROOT / "data" / "splits"
RAW_DATA_DIR = _REPO_ROOT / "raw_data"

# Store sequences at full length (max observed: 195 frames).
# Temporal cropping to the model's working window happens in train.py.
STORED_LEN = 200

MIN_FRAMES = 10      # drop clips shorter than this after trimming (bad detections)
MAX_ZERO_PCT = 0.40  # drop clips where >40% of trimmed frames have zero hand detection

# Must match FEATURE_DIM in extract_landmarks.py
FEATURE_DIM = 144   # 63 right hand + 63 left hand + 18 pose


def _normalize_hand(coords: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Wrist-centre and hand-size normalise one hand block.

    coords  : (T, 21, 3)
    detected: (T,) bool
    """
    if not detected.any():
        return coords
    wrist = coords[detected, 0:1, :]
    coords[detected] -= wrist
    hand_size = np.linalg.norm(coords[detected, 9, :], axis=1)
    valid = hand_size > 0
    det_idx = np.where(detected)[0]
    coords[det_idx[valid]] /= hand_size[valid, np.newaxis, np.newaxis]
    return coords


def normalize_clip(arr: np.ndarray) -> np.ndarray:
    """Normalise a (T, 144) landmark array.

    Feature layout (must match extract_landmarks.py):
      [0:63]    right hand  (21 landmarks × xyz)
      [63:126]  left hand   (21 landmarks × xyz)
      [126:144] pose        (6 landmarks  × xyz)

    Each hand is wrist-centred and hand-size scaled independently.
    Pose is centred on shoulder midpoint and scaled by shoulder width.
    """
    T = arr.shape[0]

    rh = arr[:, 0:63].reshape(T, 21, 3).copy()
    rh_detected = arr[:, 0:63].any(axis=1)
    rh = _normalize_hand(rh, rh_detected)

    lh = arr[:, 63:126].reshape(T, 21, 3).copy()
    lh_detected = arr[:, 63:126].any(axis=1)
    lh = _normalize_hand(lh, lh_detected)

    pose = arr[:, 126:144].reshape(T, 6, 3).copy()
    pose_detected = arr[:, 126:144].any(axis=1)

    if pose_detected.any():
        l_shoulder = pose[pose_detected, 0, :]
        r_shoulder = pose[pose_detected, 1, :]
        midpoint = (l_shoulder + r_shoulder) / 2
        pose[pose_detected] -= midpoint[:, np.newaxis, :]
        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder, axis=1)
        valid = shoulder_width > 0
        det_idx = np.where(pose_detected)[0]
        pose[det_idx[valid]] /= shoulder_width[valid, np.newaxis, np.newaxis]

    return np.concatenate([
        rh.reshape(T, 63),
        lh.reshape(T, 63),
        pose.reshape(T, 18),
    ], axis=1).astype(np.float32)


def trim_zeros(arr: np.ndarray) -> np.ndarray:
    """Remove leading and trailing frames where no hand was detected (all zeros)."""
    detected = arr.any(axis=1)
    if not detected.any():
        return arr
    first = int(detected.argmax())
    last = int(len(detected) - detected[::-1].argmax())
    return arr[first:last]


def pad_to_stored(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Zero-pad arr to STORED_LEN frames and return the actual frame count."""
    T = arr.shape[0]
    length = min(T, STORED_LEN)
    out = np.zeros((STORED_LEN, FEATURE_DIM), dtype=np.float32)
    out[:length] = arr[:length]
    return out, length


def build_splits(num_words: int = 100, top_n_classes: int | None = None) -> None:
    """Build splits from nslt_{num_words}.json."""
    split_file = RAW_DATA_DIR / f"nslt_{num_words}.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file) as f:
        split_data = json.load(f)

    wlasl_file = RAW_DATA_DIR / "WLASL_v0.3.json"
    label_to_gloss: dict[int, str] = {}
    if wlasl_file.exists():
        with open(wlasl_file) as f:
            wlasl_data = json.load(f)
        for entry in wlasl_data:
            for inst in entry["instances"]:
                vid = inst["video_id"]
                if vid in split_data:
                    label_to_gloss[split_data[vid]["action"][0]] = entry["gloss"]

    keep_labels: set[int] | None = None
    label_remap: dict[int, int] = {}
    if top_n_classes is not None:
        from collections import Counter
        train_counts = Counter(
            meta["action"][0]
            for meta in split_data.values()
            if meta["subset"] == "train"
        )
        top_labels = [lbl for lbl, _ in train_counts.most_common(top_n_classes)]
        keep_labels = set(top_labels)
        label_remap = {old: new for new, old in enumerate(sorted(top_labels))}
        print(f"Filtering to top {top_n_classes} classes: "
              f"{[label_to_gloss.get(l, str(l)) for l in sorted(top_labels)]}")

    actual_num_classes = top_n_classes if top_n_classes is not None else num_words

    buckets: dict[str, list] = {"train": [], "val": [], "test": []}
    dropped = 0
    dropped_zero = 0

    for vid_id, meta in tqdm(split_data.items(), desc="Building splits"):
        label = meta["action"][0]

        if keep_labels is not None and label not in keep_labels:
            continue

        npy_path = PROCESSED_DIR / f"{vid_id}.npy"
        if not npy_path.exists():
            continue

        arr = np.load(npy_path)     # (T, 144)
        arr = trim_zeros(arr)

        if not arr.any() or arr.shape[0] < MIN_FRAMES:
            dropped += 1
            continue

        # Drop clips where MediaPipe consistently failed to detect hands
        zero_pct = (~arr.any(axis=1)).mean()
        if zero_pct > MAX_ZERO_PCT:
            dropped_zero += 1
            continue

        arr = normalize_clip(arr)
        arr_padded, length = pad_to_stored(arr)

        remapped_label = label_remap.get(label, label)
        subset = meta["subset"]
        buckets[subset].append((arr_padded, remapped_label, length))

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    counts = {}
    for subset, samples in buckets.items():
        if not samples:
            print(f"[WARN] {subset} split is empty")
            continue
        X = np.stack([s[0] for s in samples]).astype(np.float32)  # (N, STORED_LEN, 144)
        y = np.array([s[1] for s in samples], dtype=np.int64)
        lengths = np.array([s[2] for s in samples], dtype=np.int64)
        np.savez_compressed(SPLITS_DIR / f"{subset}.npz", X=X, y=y, lengths=lengths)
        counts[subset] = len(samples)
        print(f"  {subset:5s}: {len(samples):4d} samples  X={X.shape}")

    with open(SPLITS_DIR / "label_map.json", "w") as f:
        json.dump({str(k): v for k, v in sorted(label_to_gloss.items())}, f, indent=2)

    meta_out = {
        "stored_len": STORED_LEN,
        "num_classes": actual_num_classes,
        "min_frames": MIN_FRAMES,
        "max_zero_pct": MAX_ZERO_PCT,
        "counts": counts,
        "dropped": dropped,
        "dropped_zero_frames": dropped_zero,
    }
    with open(SPLITS_DIR / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nDropped {dropped} clips (blank/too short), "
          f"{dropped_zero} for >{MAX_ZERO_PCT:.0%} interior zero frames")
    print(f"Splits saved to: {SPLITS_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=None,
                        help="Keep only the N most common classes")
    args = parser.parse_args()
    build_splits(num_words=100, top_n_classes=args.top_n)
