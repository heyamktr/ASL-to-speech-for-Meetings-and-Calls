"""Training loop for the LSTM sign classifier.

Run from the model-training/ directory:
    python -m src.train
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.lstm import SignLSTM

# Paths
_REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = _REPO_ROOT / "data" / "splits"
CKPT_DIR   = _REPO_ROOT / "checkpoints"

# Hyperparameters
BATCH_SIZE   = 32
EPOCHS       = 300
LR           = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.4
EARLY_STOP_PATIENCE = 30

# Read num_classes from the split metadata so this works for both 20 and 100
with open(SPLITS_DIR / "meta.json") as _f:
    NUM_CLASSES = json.load(_f)["num_classes"]

# Augmentation strengths
NOISE_STD = 0.02   # Gaussian noise on coordinates
FLIP_PROB = 0.5    # probability of horizontally mirroring the hand


# 1. Dataset
class LandmarkDataset(Dataset):
    def __init__(self, split: str, augment: bool = False) -> None:
        data = np.load(SPLITS_DIR / f"{split}.npz")
        raw = torch.from_numpy(data["X"]).float()          # (N, 100, 144)

        # Use right hand + presence bit + pose = 82-dim.
        # Left-hand block dropped: it is 80% zeros at this data scale (~10
        # samples/class) and consistently hurts training. Pose (66.7% signal)
        # captures arm/shoulder context that differentiates many signs.
        # Layout: [0:63]=rh coords, [63]=rh present, [64:82]=pose (18-dim)
        rh_pres = raw[:, :, 0:63].any(dim=2, keepdim=True).float()   # (N,100,1)
        self.X = torch.cat([
            raw[:, :, 0:63], rh_pres, raw[:, :, 126:144],
        ], dim=2)                                                       # (N,100,82)
        self.y = torch.from_numpy(data["y"]).long()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].clone()   # (100, 82)

        if self.augment:
            detected = x.any(dim=1)   # (100,) True where any feature non-zero

            # Jitter on coordinate dims only; preserve presence bit at index 63.
            noise = torch.randn_like(x) * NOISE_STD
            noise[:, 63] = 0
            x[detected] += noise[detected]

            # Horizontal flip: negate x-coords in right hand and pose blocks.
            if torch.rand(1).item() < FLIP_PROB:
                x[:, 0:63:3] *= -1   # rh x-coords:   cols 0,3,6,...,60
                x[:, 64::3]  *= -1   # pose x-coords:  cols 64,67,70,73,76,79

        return x, self.y[idx]


# 2. Class weights
def compute_class_weights(dataset: LandmarkDataset) -> torch.Tensor:
    labels  = dataset.y.numpy()
    counts  = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


# 3. Evaluation
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        correct    += (logits.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


# 4. Training loop
def train(resume: bool = True) -> None:
    """Train the model.

    resume=True (default): pick up from the last saved checkpoint so you
    don't repeat epochs already completed. Set to False to start fresh.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_ds = LandmarkDataset("train", augment=True)
    val_ds   = LandmarkDataset("val",   augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    model = SignLSTM(
        num_classes=NUM_CLASSES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc  = 0.0
    epochs_no_imp = 0
    start_epoch   = 1

    ckpt_path = CKPT_DIR / "best.pt"
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_val_acc = ckpt["val_acc"]
        start_epoch  = ckpt["epoch"] + 1
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        print(f"Resumed from epoch {ckpt['epoch']}  (best val_acc={best_val_acc:.3f})")
    else:
        print("Starting fresh")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
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
            best_val_acc  = val_acc
            epochs_no_imp = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "config": {
                    "num_classes": NUM_CLASSES,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                },
            }
            torch.save(ckpt, CKPT_DIR / "best.pt")
            print(f"  saved best checkpoint  (val_acc={val_acc:.3f})")
        else:
            epochs_no_imp += 1

        if epochs_no_imp >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint: {CKPT_DIR / 'best.pt'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true",
                        help="Start training from scratch instead of loading best.pt")
    args = parser.parse_args()
    train(resume=not args.no_resume)
