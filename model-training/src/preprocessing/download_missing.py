"""Download videos from WLASL_v0.3.json that are missing from raw_data/videos/."""

import json
import logging
import os
import random
import subprocess
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEOS_DIR = _REPO_ROOT / "raw_data" / "videos"
WLASL_JSON = _REPO_ROOT.parent / "WLASL" / "start_kit" / "WLASL_v0.3.json"
SPLIT_FILE = _REPO_ROOT / "raw_data" / "nslt_100.json"


def _build_id_url_map(wlasl_json: Path) -> dict[str, str]:
    with open(wlasl_json) as f:
        data = json.load(f)
    return {inst["video_id"]: inst["url"] for entry in data for inst in entry["instances"]}


def _missing_ids(split_file: Path, videos_dir: Path) -> list[str]:
    with open(split_file) as f:
        split = json.load(f)
    existing = {p.stem for p in videos_dir.glob("*.mp4")}
    return [vid_id for vid_id in split if vid_id not in existing]


def _download_direct(url: str, dest: Path) -> None:
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=30) as resp:
        dest.write_bytes(resp.read())
    time.sleep(random.uniform(0.5, 1.5))


def _download_youtube(url: str, dest_dir: Path, video_id: str) -> bool:
    cmd = [
        "yt-dlp",
        url,
        "-o", str(dest_dir / f"{video_id}.%(ext)s"),
        "--merge-output-format", "mp4",
        "--quiet",
        "--no-warnings",
    ]
    result = subprocess.run(cmd, timeout=120)
    return result.returncode == 0


def main() -> None:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    id_to_url = _build_id_url_map(WLASL_JSON)
    missing = _missing_ids(SPLIT_FILE, VIDEOS_DIR)

    log.info(f"Videos to download: {len(missing)}")

    yt_failed, direct_failed, ok = [], [], 0

    for i, vid_id in enumerate(missing, 1):
        dest = VIDEOS_DIR / f"{vid_id}.mp4"
        if dest.exists():
            continue

        url = id_to_url.get(vid_id, "")
        if not url:
            log.warning(f"[{i}/{len(missing)}] No URL for {vid_id}, skipping")
            continue

        is_youtube = "youtube" in url or "youtu.be" in url
        log.info(f"[{i}/{len(missing)}] {'YT' if is_youtube else 'direct'} {vid_id}")

        try:
            if is_youtube:
                success = _download_youtube(url, VIDEOS_DIR, vid_id)
                if not success:
                    log.warning(f"  yt-dlp failed for {vid_id}")
                    yt_failed.append(vid_id)
                    continue
            else:
                _download_direct(url, dest)

            ok += 1
        except Exception as exc:
            log.warning(f"  Error on {vid_id}: {exc}")
            (yt_failed if is_youtube else direct_failed).append(vid_id)
            if dest.exists():
                dest.unlink()

    log.info(f"\nDone. Downloaded: {ok}  |  YT failed: {len(yt_failed)}  |  Direct failed: {len(direct_failed)}")
    if yt_failed or direct_failed:
        log.info("Failed IDs: " + ", ".join(yt_failed + direct_failed))


if __name__ == "__main__":
    main()
