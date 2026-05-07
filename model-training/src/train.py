"""Training loop for the GRU sign classifier.

Run from the model-training/ directory:
    python -m src.train              # resume from best.pt if it exists
    python -m src.train --no-resume  # start fresh

Architecture changes from v1
-----------------------------
* Both hands included (146-dim input vs 82-dim right-hand-only).
* Bidirectional GRU + temporal attention replaces last-hidden-state LSTM.
* Sequence lengths propagated so attention ignores padding frames.
* Random temporal crop each epoch (training diversity).
* 4-crop test-time averaging at validation (free accuracy boost).
* hidden_size 96, dropout 0.5, weight_decay 1e-3, temporal frame dropout 25%.
* Velocity features: frame-to-frame diffs appended (146 pos + 146 vel = 292-dim).
* Adam eps 1e-3, label_smoothing 0.1.
* Mixup (alpha=0.4) and scale jitter (±20%) added to training augmentation.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.lstm import SignGRU

# ── Paths ──────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = _REPO_ROOT / "data" / "splits"
CKPT_DIR   = _REPO_ROOT / "checkpoints"

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN      = 100   # frames fed to the model per forward pass
INPUT_DIM    = 292   # 146 position + 146 velocity (frame-to-frame diff)
BATCH_SIZE   = 32
EPOCHS       = 300
LR           = 1e-3
WEIGHT_DECAY = 1e-3
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.5
EARLY_STOP_PATIENCE = 30

# Augmentation
NOISE_STD       = 0.07
FLIP_PROB       = 0.5
FRAME_DROP_PROB = 0.25   # probability of zeroing out any single frame (temporal dropout)
SCALE_JITTER    = 0.2    # coordinate scale drawn from U[1-SCALE_JITTER, 1+SCALE_JITTER]
MIXUP_ALPHA     = 0.4    # Beta distribution parameter for mixup; 0 disables

# Test-time augmentation: average logits over this many temporal crops at val.
TTA_CROPS = 4

with open(SPLITS_DIR / "meta.json") as _f:
    _meta = json.load(_f)
NUM_CLASSES = _meta["num_classes"]
STORED_LEN  = _meta.get("stored_len", _meta.get("seq_len", SEQ_LEN))


# ── Temporal crop helper ───────────────────────────────────────────────────────
def temporal_crop(
    x_stored: torch.Tensor,
    lengths: torch.Tensor,
    seq_len: int,
    mode: str = "random",
    crop_idx: int = 0,
    num_crops: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop (B, STORED_LEN, D) → (B, seq_len, D).

    Args:
        mode: 'random'  — random start within the active region (training).
              'center'  — middle of the active region.
              'uniform' — evenly-spaced crops indexed by crop_idx/num_crops (TTA).
    Returns:
        (cropped_batch, crop_lengths) where crop_lengths[i] = min(lengths[i], seq_len).
    """
    B, S, D = x_stored.shape
    out = torch.zeros(B, seq_len, D, dtype=x_stored.dtype, device=x_stored.device)
    crop_lens = torch.zeros(B, dtype=torch.long, device=x_stored.device)

    for i in range(B):
        L = min(int(lengths[i].item()), S)
        if L > seq_len:
            if mode == "random":
                start = torch.randint(0, L - seq_len + 1, (1,)).item()
            elif mode == "uniform":
                stride = max(1, (L - seq_len) // max(1, num_crops - 1))
                start = min(crop_idx * stride, L - seq_len)
            else:  # center
                start = (L - seq_len) // 2
            out[i] = x_stored[i, start : start + seq_len]
            crop_lens[i] = seq_len
        else:
            out[i, :L] = x_stored[i, :L]
            crop_lens[i] = L

    return out, crop_lens


# ── Augmentation ───────────────────────────────────────────────────────────────
def augment_batch(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Augment a (B, T, 146) batch in-place; padding frames are left untouched.

    Feature layout (146-dim):
      [0:63]    rh coords   | [63]     rh_pres
      [64:127]  lh coords   | [127]    lh_pres
      [128:146] pose coords
    """
    B, T, _ = x.shape
    pad_mask = torch.arange(T, device=x.device)[None] >= lengths[:, None]  # (B, T)

    # Temporal frame dropout: zero out random active frames.
    # Forces the model to classify from partial observations rather than
    # memorising specific frame positions in the training sequences.
    frame_drop = (torch.rand(B, T, device=x.device) < FRAME_DROP_PROB) & ~pad_mask
    x[frame_drop] = 0.0

    # Gaussian jitter on coordinate dims only; skip presence-bit indices
    noise = torch.randn_like(x) * NOISE_STD
    noise[:, :, 63]  = 0   # rh_pres
    noise[:, :, 127] = 0   # lh_pres
    noise[pad_mask]  = 0   # don't jitter padding
    noise[frame_drop] = 0  # already zeroed; keep consistent
    x = x + noise

    # Scale jitter: random per-sample scale on coordinate dims only (not presence bits).
    scale = 1.0 + (torch.rand(B, 1, 1, device=x.device) * 2 - 1) * SCALE_JITTER
    x[:, :, :63]    *= scale
    x[:, :, 64:127] *= scale
    x[:, :, 128:]   *= scale

    # Horizontal flip for a random subset of the batch
    flip = torch.rand(B, device=x.device) < FLIP_PROB
    x[flip, :, 0:63:3]   *= -1   # rh x-coords  (0,3,6,...,60)
    x[flip, :, 64:127:3] *= -1   # lh x-coords  (64,67,...,124)
    x[flip, :, 128::3]   *= -1   # pose x-coords (128,131,...,143)

    return x


def add_velocity(x: torch.Tensor) -> torch.Tensor:
    """Append frame-to-frame differences to (B, T, 146) → (B, T, 292).

    Velocity must be computed after augmentation so flips/scales stay consistent.
    The first frame gets a zero velocity (no prior frame available).
    """
    vel = torch.diff(x, dim=1)                          # (B, T-1, 146)
    vel = torch.cat([torch.zeros_like(x[:, :1, :]), vel], dim=1)  # (B, T, 146)
    return torch.cat([x, vel], dim=2)                   # (B, T, 292)


# ── Dataset ────────────────────────────────────────────────────────────────────
class LandmarkDataset(Dataset):
    """Loads stored (N, STORED_LEN, 144) arrays and builds 146-dim features.

    Returns (x_stored, y, length) tuples — temporal cropping happens in the
    training/evaluation loops so both random crops (train) and TTA (val) are
    possible without rebuilding the dataset.
    """

    def __init__(self, split: str) -> None:
        data = np.load(SPLITS_DIR / f"{split}.npz")
        raw = torch.from_numpy(data["X"]).float()          # (N, STORED_LEN, 144)

        # Build 146-dim feature: both hands with per-hand presence bits.
        rh_pres = raw[:, :, 0:63].any(dim=2, keepdim=True).float()    # (N, S, 1)
        lh_pres = raw[:, :, 63:126].any(dim=2, keepdim=True).float()  # (N, S, 1)
        self.X = torch.cat([
            raw[:, :, 0:63],   rh_pres,   # 64 dims — right hand
            raw[:, :, 63:126], lh_pres,   # 64 dims — left hand
            raw[:, :, 126:144],           # 18 dims — pose
        ], dim=2)                          # (N, STORED_LEN, 146)

        self.y       = torch.from_numpy(data["y"]).long()
        self.lengths = torch.from_numpy(data["lengths"]).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.lengths[idx]


# ── Class weights ──────────────────────────────────────────────────────────────
def compute_class_weights(dataset: LandmarkDataset) -> torch.Tensor:
    labels  = dataset.y.numpy()
    counts  = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


# ── Evaluation with TTA ────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_crops: int = TTA_CROPS,
) -> tuple[float, float]:
    """Evaluate with uniform multi-crop TTA.

    For each sample, `num_crops` evenly-spaced temporal crops are averaged.
    Sequences shorter than SEQ_LEN get the same crop every time (no variance).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x_stored, y_batch, len_batch in loader:
        x_stored = x_stored.to(device)
        y_batch  = y_batch.to(device)
        len_batch = len_batch.to(device)

        all_logits = []
        for crop_i in range(num_crops):
            x_crop, crop_lens = temporal_crop(
                x_stored, len_batch, SEQ_LEN,
                mode="uniform", crop_idx=crop_i, num_crops=num_crops,
            )
            x_crop = add_velocity(x_crop)
            all_logits.append(model(x_crop, lengths=crop_lens))

        logits = torch.stack(all_logits).mean(0)   # (B, num_classes)

        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)

    return total_loss / total, correct / total


# ── Training loop ──────────────────────────────────────────────────────────────
def train(resume: bool = True) -> None:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_ds = LandmarkDataset("train")
    val_ds   = LandmarkDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=False)
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    print(f"Input dim: {INPUT_DIM}  SEQ_LEN: {SEQ_LEN}  STORED_LEN: {STORED_LEN}")

    model = SignGRU(
        input_size=INPUT_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15,
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    # Early stopping and checkpointing are driven by val_loss, not val_acc.
    # With only 238 val samples, accuracy is very noisy (1% = 2.4 samples);
    # val_loss is a much smoother signal and reflects genuine model improvement.
    best_val_loss = float("inf")
    best_val_acc  = 0.0       # tracked for display only
    epochs_no_imp = 0
    start_epoch   = 1

    ckpt_path = CKPT_DIR / "best.pt"
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt.get("config", {})
        arch_match = (
            cfg.get("input_size")   == INPUT_DIM
            and cfg.get("hidden_size") == HIDDEN_SIZE
            and cfg.get("num_classes") == NUM_CLASSES
        )
        if arch_match:
            model.load_state_dict(ckpt["model_state"])
            best_val_loss = ckpt.get("val_loss", float("inf"))
            best_val_acc  = ckpt.get("val_acc", 0.0)
            start_epoch   = ckpt["epoch"] + 1
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            print(f"Resumed from epoch {ckpt['epoch']}  "
                  f"(best val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.3f})")
        else:
            print("Checkpoint architecture mismatch — starting fresh")
    else:
        print("Starting fresh")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x_stored, y_batch, len_batch in train_loader:
            x_stored  = x_stored.to(device)
            y_batch   = y_batch.to(device)
            len_batch = len_batch.to(device)

            # Random temporal crop — each epoch sees a different window for
            # sequences longer than SEQ_LEN, providing temporal diversity.
            x_crop, crop_lens = temporal_crop(
                x_stored, len_batch, SEQ_LEN, mode="random",
            )

            x_crop = augment_batch(x_crop, crop_lens)
            x_crop = add_velocity(x_crop)

            if MIXUP_ALPHA > 0:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                idx = torch.randperm(x_crop.size(0), device=device)
                x_crop = lam * x_crop + (1 - lam) * x_crop[idx]
                y_b    = y_batch[idx]
                logits = model(x_crop, lengths=crop_lens)
                loss   = lam * criterion(logits, y_batch) + (1 - lam) * criterion(logits, y_b)
            else:
                logits = model(x_crop, lengths=crop_lens)
                loss   = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * len(y_batch)
            train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            train_total   += len(y_batch)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train loss {train_loss:.4f}  acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f}  acc {val_acc:.3f} | "
            f"lr {current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_imp = 0
            ckpt = {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_acc":         val_acc,
                "val_loss":        val_loss,
                "config": {
                    "input_size":   INPUT_DIM,
                    "hidden_size":  HIDDEN_SIZE,
                    "num_layers":   NUM_LAYERS,
                    "dropout":      DROPOUT,
                    "num_classes":  NUM_CLASSES,
                },
            }
            torch.save(ckpt, CKPT_DIR / "best.pt")
            print(f"  saved best checkpoint  (val_loss={val_loss:.4f}  val_acc={val_acc:.3f})")
        else:
            epochs_no_imp += 1

        if epochs_no_imp >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}  best val_acc={best_val_acc:.3f}")
    print(f"Checkpoint: {CKPT_DIR / 'best.pt'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true",
                        help="Start training from scratch instead of loading best.pt")
    args = parser.parse_args()
    train(resume=not args.no_resume)
