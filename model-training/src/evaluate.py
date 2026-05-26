"""Evaluate a trained checkpoint on the held-out test split.

Run from the model-training/ directory:
    python -m src.evaluate
    python -m src.evaluate --split val   # evaluate on val instead of test
    python -m src.evaluate --tta 1       # disable TTA (single center crop)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.lstm import SignGRU
from src.train import LandmarkDataset, temporal_crop, add_velocity, TTA_CROPS, SEQ_LEN

_REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = _REPO_ROOT / "data" / "splits"
CKPT_DIR   = _REPO_ROOT / "checkpoints"


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label_map: dict,
    num_crops: int = TTA_CROPS,
    top_k: tuple[int, ...] = (1, 3, 5),
) -> None:
    model.eval()

    all_logits = []
    all_labels = []

    for x_stored, y_batch, len_batch in loader:
        x_stored  = x_stored.to(device)
        len_batch = len_batch.to(device)

        crop_logits = []
        for crop_i in range(num_crops):
            x_crop, crop_lens = temporal_crop(
                x_stored, len_batch, SEQ_LEN,
                mode="uniform", crop_idx=crop_i, num_crops=num_crops,
            )
            x_crop = add_velocity(x_crop)
            crop_logits.append(model(x_crop, lengths=crop_lens))

        logits = torch.stack(crop_logits).mean(0).cpu()
        all_logits.append(logits)
        all_labels.append(y_batch)

    logits = torch.cat(all_logits)   # (N, C)
    labels = torch.cat(all_labels)   # (N,)
    N = len(labels)

    print(f"\nEvaluated {N} samples with {num_crops}-crop TTA\n")

    for k in top_k:
        top_k_preds = logits.topk(k, dim=1).indices  # (N, k)
        correct = (top_k_preds == labels[:, None]).any(dim=1).sum().item()
        print(f"  Top-{k} accuracy: {correct}/{N} = {correct/N:.2%}")

    # Per-class breakdown (top-1)
    pred1 = logits.argmax(dim=1)
    print("\nPer-class top-1 accuracy:")
    class_correct = {}
    class_total   = {}
    for p, t in zip(pred1.tolist(), labels.tolist()):
        class_total[t]   = class_total.get(t, 0) + 1
        class_correct[t] = class_correct.get(t, 0) + int(p == t)

    rows = []
    for cls_id in sorted(class_total):
        gloss = label_map.get(str(cls_id), str(cls_id))
        c, tot = class_correct.get(cls_id, 0), class_total[cls_id]
        rows.append((gloss, c, tot))

    # Print worst classes first
    rows.sort(key=lambda r: r[1] / r[2])
    for gloss, c, tot in rows:
        bar = "#" * c + "." * (tot - c)
        print(f"  {gloss:20s}  {c}/{tot}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--tta",   type=int, default=TTA_CROPS,
                        help="Number of temporal crops to average (1 = no TTA)")
    parser.add_argument("--ckpt",  default=str(CKPT_DIR / "best.pt"))
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    print(f"Checkpoint: {ckpt_path}")
    print(f"Trained for {ckpt['epoch']} epochs  |  val_acc={ckpt['val_acc']:.3f}")
    print(f"Config: {cfg}")

    model = SignGRU(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        num_classes=cfg["num_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    ds     = LandmarkDataset(args.split)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    label_map_path = SPLITS_DIR / "label_map.json"
    label_map = {}
    if label_map_path.exists():
        with open(label_map_path) as f:
            label_map = json.load(f)

    evaluate_split(model, loader, device, label_map, num_crops=args.tta)


if __name__ == "__main__":
    main()
