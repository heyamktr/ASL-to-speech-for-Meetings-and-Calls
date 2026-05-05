"""Build train/val/test splits from extracted landmarks.

Reads individual per-video .npy files from data/processed/, trims dead frames,
pads/truncates to a fixed sequence length, and writes data/splits/*.npz files
ready for the PyTorch DataLoader in train.py.

Output layout
-------------
data/splits/
    train.npz   — arrays X (N, SEQ_LEN, 144) and y (N,)
    val.npz
    test.npz
    label_map.json   — {label_int: gloss_str}
    meta.json        — SEQ_LEN, num_classes, per-split counts
"""

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
SPLITS_DIR = _REPO_ROOT / "data" / "splits"
RAW_DATA_DIR = _REPO_ROOT / "raw_data"

SEQ_LEN = 100       # frames per sample — covers ~93rd percentile after trimming
MIN_FRAMES = 10     # drop clips shorter than this after trimming (bad detections)

# Must match FEATURE_DIM in extract_landmarks.py
FEATURE_DIM = 144   # 63 right hand + 63 left hand + 18 pose


def _normalize_hand(coords: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Wrist-centre and hand-size normalise one hand block.

    coords  : (T, 21, 3)
    detected: (T,) bool — True where this hand was detected
    Returns : (T, 21, 3) normalised in-place
    """
    if not detected.any():
        return coords

    # 1. Subtract wrist (landmark 0) — puts wrist at origin
    wrist = coords[detected, 0:1, :]
    coords[detected] -= wrist

    # 2. Scale by wrist→middle-finger-base (landmark 9) distance
    hand_size = np.linalg.norm(coords[detected, 9, :], axis=1)
    valid   = hand_size > 0
    det_idx = np.where(detected)[0]
    coords[det_idx[valid]] /= hand_size[valid, np.newaxis, np.newaxis]

    return coords


def normalize_clip(arr: np.ndarray) -> np.ndarray:
    """Normalise a (T, 144) landmark array.

    Feature layout (must match extract_landmarks.py):
      [0:63]    right hand  (21 landmarks × xyz)
      [63:126]  left hand   (21 landmarks × xyz)
      [126:144] pose        (6 landmarks  × xyz)
                            left_shoulder, right_shoulder,
                            left_elbow,    right_elbow,
                            left_wrist,    right_wrist

    Each hand is wrist-centred and hand-size scaled independently.
    Pose is centred on the shoulder midpoint and scaled by shoulder width,
    making it invariant to the signer's distance from the camera.
    """
    T = arr.shape[0]

    # ── Right hand ──────────────────────────────────────────────────────────
    rh = arr[:, 0:63].reshape(T, 21, 3).copy()
    rh_detected = arr[:, 0:63].any(axis=1)
    rh = _normalize_hand(rh, rh_detected)

    # ── Left hand ───────────────────────────────────────────────────────────
    lh = arr[:, 63:126].reshape(T, 21, 3).copy()
    lh_detected = arr[:, 63:126].any(axis=1)
    lh = _normalize_hand(lh, lh_detected)

    # ── Pose ────────────────────────────────────────────────────────────────
    # Layout: [left_shoulder(0), right_shoulder(1), left_elbow(2),
    #          right_elbow(3), left_wrist(4), right_wrist(5)]
    pose = arr[:, 126:144].reshape(T, 6, 3).copy()
    pose_detected = arr[:, 126:144].any(axis=1)

    if pose_detected.any():
        l_shoulder = pose[pose_detected, 0, :]   # (D, 3)
        r_shoulder = pose[pose_detected, 1, :]   # (D, 3)

        # Centre on shoulder midpoint
        midpoint = (l_shoulder + r_shoulder) / 2           # (D, 3)
        pose[pose_detected] -= midpoint[:, np.newaxis, :]

        # Scale by shoulder width
        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder, axis=1)  # (D,)
        valid   = shoulder_width > 0
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
        return arr                          # fully blank — caller will drop it
    first = int(detected.argmax())
    last = int(len(detected) - detected[::-1].argmax())
    return arr[first:last]


def pad_or_truncate(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Fix sequence to exactly seq_len frames.

    Short sequences are zero-padded at the end (post-padding matches
    PyTorch pack_padded_sequence convention). Long ones are centre-cropped
    so the most active part of the sign is kept.
    """
    T = arr.shape[0]
    if T >= seq_len:
        # Centre crop: keep the middle seq_len frames
        start = (T - seq_len) // 2
        return arr[start : start + seq_len]
    # Zero-pad at the end
    pad = np.zeros((seq_len - T, arr.shape[1]), dtype=arr.dtype)
    return np.vstack([arr, pad])


def build_splits(num_words: int = 100, top_n_classes: int | None = None) -> None:
    """Build splits from nslt_{num_words}.json.

    top_n_classes: if set, keep only the N classes with the most training
    samples and remap their labels to 0..N-1. Useful for validating the
    pipeline on a smaller, easier problem before scaling to all 100 classes.
    """
    split_file = RAW_DATA_DIR / f"nslt_{num_words}.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file) as f:
        split_data = json.load(f)

    # Build label_int -> gloss from WLASL_v0.3.json
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

    # Optional: keep only the top-N most common classes (by training samples)
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

    for vid_id, meta in tqdm(split_data.items(), desc="Building splits"):
        label = meta["action"][0]

        if keep_labels is not None and label not in keep_labels:
            continue

        npy_path = PROCESSED_DIR / f"{vid_id}.npy"
        if not npy_path.exists():
            continue

        arr = np.load(npy_path)             # (T, 144)
        arr = trim_zeros(arr)

        if arr.shape[0] < MIN_FRAMES or not arr.any():
            dropped += 1
            continue

        arr = normalize_clip(arr)           # wrist-centred, hand-size scaled
        arr = pad_or_truncate(arr, SEQ_LEN) # (SEQ_LEN, 144)
        remapped_label = label_remap.get(label, label)
        subset = meta["subset"]             # 'train' | 'val' | 'test'

        buckets[subset].append((arr, remapped_label))

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    counts = {}
    for subset, samples in buckets.items():
        if not samples:
            print(f"[WARN] {subset} split is empty")
            continue
        X = np.stack([s[0] for s in samples], axis=0).astype(np.float32)
        y = np.array([s[1] for s in samples], dtype=np.int64)
        np.savez_compressed(SPLITS_DIR / f"{subset}.npz", X=X, y=y)
        counts[subset] = len(samples)
        print(f"  {subset:5s}: {len(samples):4d} samples  X={X.shape}")

    with open(SPLITS_DIR / "label_map.json", "w") as f:
        json.dump({str(k): v for k, v in sorted(label_to_gloss.items())}, f, indent=2)

    meta_out = {"seq_len": SEQ_LEN, "num_classes": actual_num_classes,
                "min_frames": MIN_FRAMES, "counts": counts, "dropped": dropped}
    with open(SPLITS_DIR / "meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nDropped {dropped} clips (shorter than {MIN_FRAMES} frames after trimming)")
    print(f"Splits saved to: {SPLITS_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=None,
                        help="Keep only the N most common classes (e.g. 20)")
    args = parser.parse_args()
    build_splits(num_words=100, top_n_classes=args.top_n)
