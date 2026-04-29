"""Build train/val/test splits from extracted landmarks."""

from pathlib import Path


def build_splits(processed_dir: Path, num_words: int = 100) -> None:
    """Filter to top-N most common words and write train/val/test index files."""
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    pass
