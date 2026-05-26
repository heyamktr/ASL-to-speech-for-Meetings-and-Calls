"""Export a trained PyTorch checkpoint to ONNX for the backend.

Run from the model-training/ directory:
    python -m src.export_onnx
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from src.models.lstm import SignGRU

_REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = _REPO_ROOT / "checkpoints"
EXPORTS_DIR = _REPO_ROOT / "exports"
SPLITS_DIR = _REPO_ROOT / "data" / "splits"

DEFAULT_SEQ_LEN = 100
RAW_FEATURE_DIM = 144


def load_model(ckpt_path: Path, device: torch.device) -> tuple[SignGRU, dict]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]

    model = SignGRU(
        input_size=cfg["input_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        num_classes=cfg["num_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(CKPT_DIR / "best.pt"))
    parser.add_argument("--output", default=str(EXPORTS_DIR / "asl_model.onnx"))
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt_path = Path(args.ckpt)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(ckpt_path, device)
    cfg = ckpt["config"]

    dummy_x = torch.zeros(1, args.seq_len, cfg["input_size"], dtype=torch.float32)
    dummy_lengths = torch.tensor([args.seq_len], dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_x, dummy_lengths),
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=False,
        input_names=["features", "lengths"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch"},
            "lengths": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    label_map_src = SPLITS_DIR / "label_map.json"
    label_map_dst = output_path.parent / "label_map.json"
    if label_map_src.exists():
        shutil.copyfile(label_map_src, label_map_dst)

    meta = {
        "checkpoint": str(ckpt_path),
        "onnx_model": str(output_path),
        "seq_len": args.seq_len,
        "raw_feature_dim": RAW_FEATURE_DIM,
        "model_input_dim": cfg["input_size"],
        "num_classes": cfg["num_classes"],
        "hidden_size": cfg["hidden_size"],
        "num_layers": cfg["num_layers"],
        "dropout": cfg["dropout"],
        "label_map": str(label_map_dst) if label_map_src.exists() else None,
    }
    with open(output_path.parent / "export_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Exported ONNX model to: {output_path}")
    if label_map_src.exists():
        print(f"Copied label map to: {label_map_dst}")
    print(f"Wrote metadata to: {output_path.parent / 'export_meta.json'}")


if __name__ == "__main__":
    main()
