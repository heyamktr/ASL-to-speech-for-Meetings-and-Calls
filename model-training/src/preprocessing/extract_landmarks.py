"""Extract 63-dim hand landmarks from WLASL videos using MediaPipe Hands."""

from pathlib import Path

import numpy as np


def extract_video_landmarks(video_path: Path) -> np.ndarray:
    """Read a video file and return an array of shape (num_frames, 63).

    21 hand landmarks × (x, y, z) per frame = 63 numbers.
    """
    # TODO: open video with OpenCV, run MediaPipe Hands per frame, stack results
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: iterate over data/raw, write .npy arrays to data/processed
    pass
