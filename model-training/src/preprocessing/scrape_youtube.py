"""Scrape YouTube for additional ASL training videos.

For each of the 100 target words, searches YouTube for short tutorial clips
and downloads up to MAX_PER_WORD new videos. New entries are appended to
nslt_100.json so they flow straight into the existing pipeline:
    python -m src.preprocessing.extract_landmarks
    python -m src.preprocessing.build_dataset

Usage:
    python -m src.preprocessing.scrape_youtube               # all underrepresented words
    python -m src.preprocessing.scrape_youtube --all         # all 100 words
    python -m src.preprocessing.scrape_youtube --word BOOK   # single word
    python -m src.preprocessing.scrape_youtube --min-count 15  # only words below threshold
"""

import argparse
import json
import logging
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
VIDEOS_DIR   = _REPO_ROOT / "raw_data" / "videos"
SPLIT_FILE   = _REPO_ROOT / "raw_data" / "nslt_100.json"
WLASL_JSON   = _REPO_ROOT.parent / "WLASL" / "start_kit" / "WLASL_v0.3.json"

# Tune these to control download volume and quality.
MAX_PER_WORD   = 4    # max NEW videos to add per word
MAX_DURATION   = 90   # seconds — skip long compilation videos
SEARCH_POOL    = 10   # YouTube search results to consider per query
DEFAULT_MIN    = 15   # scrape words with fewer than this many train samples

# yt-dlp and node paths
_VE = _REPO_ROOT / ".venv" / "bin" / "yt-dlp"
YT_DLP    = str(_VE) if _VE.exists() else "yt-dlp"
NODE_PATH = "/usr/local/bin/node"

# High-quality ASL channels to prefer (yt-dlp will still rank by relevance,
# but these search terms bias toward tutorial content).
SEARCH_QUERIES = [
    "{word} ASL American Sign Language how to sign",
    "sign language {word} tutorial",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _build_class_map(split_data: dict, wlasl_data: list) -> tuple[dict, dict]:
    """Return (class_id→GLOSS, GLOSS→class_id)."""
    c2g: dict[int, str] = {}
    for entry in wlasl_data:
        for inst in entry["instances"]:
            vid = inst["video_id"]
            if vid in split_data:
                cid = split_data[vid]["action"][0]
                c2g[cid] = entry["gloss"].upper()
    return c2g, {v: k for k, v in c2g.items()}


def _train_counts(split_data: dict, class_to_gloss: dict) -> Counter:
    counts: Counter = Counter()
    for meta in split_data.values():
        if meta["subset"] == "train":
            gloss = class_to_gloss.get(meta["action"][0])
            if gloss:
                counts[gloss] += 1
    return counts


def _already_in_split(split_data: dict) -> set[str]:
    return set(split_data.keys())


def _download_videos(word: str, want: int) -> list[str]:
    """Search YouTube and download up to `want` new mp4s.

    Returns list of new video IDs (stems of the downloaded files).
    """
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    before = {p.stem for p in VIDEOS_DIR.glob("*.mp4")}

    for query_template in SEARCH_QUERIES:
        query = f"ytsearch{SEARCH_POOL}:{query_template.format(word=word)}"
        cmd = [
            YT_DLP,
            "--js-runtimes", f"node:{NODE_PATH}",
            "--remote-components", "ejs:github",
            "--cookies-from-browser", "chrome",
            query,
            "-o", str(VIDEOS_DIR / "%(id)s.%(ext)s"),
            "--merge-output-format", "mp4",
            "--format", "mp4/best[height<=480]/best",
            "--match-filters", f"duration<={MAX_DURATION}",
            "--max-downloads", str(want),
            "--ignore-errors",
            "--quiet",
            "--no-warnings",
        ]
        try:
            subprocess.run(cmd, timeout=300, check=False)
        except subprocess.TimeoutExpired:
            log.warning(f"  [{word}] yt-dlp timed out on query: {query_template}")

        after = {p.stem for p in VIDEOS_DIR.glob("*.mp4")}
        new = list(after - before)
        if new:
            return new[:want]

        time.sleep(random.uniform(1.5, 3.0))

    return []


# ── Main ───────────────────────────────────────────────────────────────────────

def scrape(
    target_words: list[str] | None = None,
    min_count: int = DEFAULT_MIN,
) -> None:
    split_data: dict = _load_json(SPLIT_FILE)
    wlasl_data: list = _load_json(WLASL_JSON)

    class_to_gloss, gloss_to_class = _build_class_map(split_data, wlasl_data)
    counts = _train_counts(split_data, class_to_gloss)

    if target_words:
        words_to_scrape = [w.upper() for w in target_words]
    else:
        # Sort by count ascending so most-starved words go first.
        all_words = sorted(class_to_gloss.values())
        words_to_scrape = [w for w in sorted(all_words, key=lambda w: counts[w])
                           if counts[w] < min_count]
        if not words_to_scrape:
            log.info(f"All words already have >= {min_count} train samples. "
                     "Use --all or lower --min-count.")
            return

    log.info(f"Words to scrape ({len(words_to_scrape)}): "
             f"{', '.join(words_to_scrape)}")
    log.info(f"Max new videos per word: {MAX_PER_WORD}")

    already_in = _already_in_split(split_data)
    total_added = 0

    for word in words_to_scrape:
        class_id = gloss_to_class.get(word)
        if class_id is None:
            log.warning(f"  [{word}] unknown word, skipping")
            continue

        current = counts[word]
        want    = MAX_PER_WORD
        log.info(f"[{word}]  current train={current}  downloading up to {want}")

        new_ids = _download_videos(word, want)

        added = 0
        for vid_id in new_ids:
            if vid_id in already_in:
                continue
            split_data[vid_id] = {
                "subset": "train",
                "action": [class_id, 0, 0],
            }
            already_in.add(vid_id)
            added += 1
            total_added += 1
            log.info(f"  added {vid_id} → {word} (class {class_id})")

        if added == 0:
            log.warning(f"  [{word}] no new videos found")

        time.sleep(random.uniform(2.0, 4.0))

    _save_json(SPLIT_FILE, split_data)
    log.info(f"\nDone. Added {total_added} new video entries to nslt_100.json.")
    log.info("Next steps:")
    log.info("  python -m src.preprocessing.extract_landmarks")
    log.info("  python -m src.preprocessing.build_dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--word", nargs="+",
                       help="Scrape specific word(s) only")
    group.add_argument("--all", action="store_true",
                       help="Scrape all 100 words regardless of current count")
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN,
                        help=f"Only scrape words with fewer than N train samples "
                             f"(default: {DEFAULT_MIN})")
    args = parser.parse_args()

    if args.all:
        scrape(target_words=None, min_count=99999)
    elif args.word:
        scrape(target_words=args.word, min_count=0)
    else:
        scrape(target_words=None, min_count=args.min_count)
