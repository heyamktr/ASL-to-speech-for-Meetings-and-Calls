"""Extract 144-dim hand + pose landmarks from WLASL videos using MediaPipe.

Feature layout per frame (144 numbers total):
  [0:63]    right hand — 21 landmarks × (x, y, z)
  [63:126]  left hand  — 21 landmarks × (x, y, z)  (zeros if not detected)
  [126:144] pose       —  6 landmarks × (x, y, z)
            order: left_shoulder, right_shoulder,
                   left_elbow,    right_elbow,
                   left_wrist,    right_wrist
"""

import json
import ssl
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR  = _REPO_ROOT / "raw_data"
VIDEOS_DIR    = RAW_DATA_DIR / "videos"
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
HAND_MODEL_PATH = RAW_DATA_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = RAW_DATA_DIR / "pose_landmarker.task"

# ── Feature constants ──────────────────────────────────────────────────────────
NUM_HAND_LM  = 21
COORDS       = 3
# MediaPipe Pose landmark indices we care about:
# 11=left_shoulder  12=right_shoulder
# 13=left_elbow     14=right_elbow
# 15=left_wrist     16=right_wrist
POSE_INDICES = [11, 12, 13, 14, 15, 16]
FEATURE_DIM  = NUM_HAND_LM * COORDS * 2 + len(POSE_INDICES) * COORDS  # 144

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    print(f"Downloading {path.name} ...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx) as resp, open(path, "wb") as f:
        f.write(resp.read())
    print(f"  saved to {path}")


# ── Core extraction ────────────────────────────────────────────────────────────
def extract_video_landmarks(
    video_path: Path,
    hand_detector,
    pose_detector,
    frame_start: int = 0,
    frame_end: int = -1,
) -> np.ndarray:
    """Read a video clip and return (T, 144) landmark array.

    For every frame:
      - HandLandmarker finds up to 2 hands, classified as Left or Right.
        Right hand → indices [0:63], Left hand → indices [63:126].
        Missing hand → zeros.
      - PoseLandmarker finds the body; we take 6 key landmarks (shoulders,
        elbows, wrists) → indices [126:144]. Missing → zeros.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    if frame_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    frames = []
    current = frame_start

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_end >= 0 and current > frame_end:
            break
        current += 1

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ── Hands ──────────────────────────────────────────────────────────
        hand_result  = hand_detector.detect(mp_image)
        right_coords = np.zeros(NUM_HAND_LM * COORDS, dtype=np.float32)
        left_coords  = np.zeros(NUM_HAND_LM * COORDS, dtype=np.float32)

        if hand_result.hand_landmarks:
            for i, hand_lms in enumerate(hand_result.hand_landmarks):
                # category_name is 'Right' or 'Left' from MediaPipe's perspective
                label = hand_result.handedness[i][0].category_name
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lms],
                    dtype=np.float32,
                ).flatten()
                if label == "Right":
                    right_coords = coords
                else:
                    left_coords = coords

        # ── Pose ───────────────────────────────────────────────────────────
        pose_result = pose_detector.detect(mp_image)
        pose_coords = np.zeros(len(POSE_INDICES) * COORDS, dtype=np.float32)

        if pose_result.pose_landmarks:
            lms = pose_result.pose_landmarks[0]
            pose_coords = np.array(
                [[lms[idx].x, lms[idx].y, lms[idx].z] for idx in POSE_INDICES],
                dtype=np.float32,
            ).flatten()

        frames.append(np.concatenate([right_coords, left_coords, pose_coords]))

    cap.release()

    if frames:
        return np.stack(frames, axis=0)
    return np.zeros((1, FEATURE_DIM), dtype=np.float32)


# ── Main ───────────────────────────────────────────────────────────────────────
def main(num_words: int = 100, force: bool = False) -> None:
    """Extract landmarks for all videos in nslt_{num_words}.json.

    force=True  — re-extract even if the .npy file exists. Use this when
                  the feature format changed (e.g. going from 63 to 144 dims).
    force=False — skip files that already have the correct FEATURE_DIM; any
                  file with the wrong shape is automatically re-extracted.
    """
    print(f"Feature dim  : {FEATURE_DIM}  (right hand 63 + left hand 63 + pose 18)")
    print(f"Videos dir   : {VIDEOS_DIR}  (exists: {VIDEOS_DIR.exists()})")
    print(f"Output dir   : {PROCESSED_DIR}")
    print()

    _download_if_missing(POSE_MODEL_PATH, POSE_MODEL_URL)

    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"hand_landmarker.task not found at {HAND_MODEL_PATH}\n"
            "Download from: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker"
        )

    split_file = RAW_DATA_DIR / f"nslt_{num_words}.json"
    with open(split_file) as f:
        split_data = json.load(f)

    # Load per-clip frame bounds (compilation videos)
    wlasl_file = RAW_DATA_DIR / "WLASL_v0.3.json"
    frame_bounds: dict[str, tuple[int, int]] = {}
    if wlasl_file.exists():
        with open(wlasl_file) as f:
            wlasl_data = json.load(f)
        for entry in wlasl_data:
            for inst in entry["instances"]:
                vid = inst["video_id"]
                fs  = inst.get("frame_start", -1)
                fe  = inst.get("frame_end",   -1)
                if fs >= 0 and fe > fs:
                    frame_bounds[vid] = (fs, fe)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,                        # both hands now
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    skipped = 0
    with (
        mp_vision.HandLandmarker.create_from_options(hand_options) as hand_det,
        mp_vision.PoseLandmarker.create_from_options(pose_options) as pose_det,
    ):
        for video_id, meta in tqdm(split_data.items(), desc="Extracting landmarks"):
            video_path  = VIDEOS_DIR    / f"{video_id}.mp4"
            output_path = PROCESSED_DIR / f"{video_id}.npy"

            if not force and output_path.exists():
                # Re-extract only if the feature dimension is outdated
                existing = np.load(output_path)
                if existing.shape[1] == FEATURE_DIM:
                    continue   # already correct format

            if not video_path.exists():
                skipped += 1
                continue

            try:
                fs, fe = frame_bounds.get(video_id, (0, -1))
                arr = extract_video_landmarks(video_path, hand_det, pose_det, fs, fe)
                np.save(output_path, arr)
            except Exception as exc:
                print(f"\n[WARN] Failed on {video_id}: {exc}")
                skipped += 1

    print(f"\nDone. Skipped {skipped} videos (missing or failed).")
    print(f"Processed files saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if .npy already exists")
    args = parser.parse_args()
    main(num_words=100, force=args.force)
