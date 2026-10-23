#!/usr/bin/env python3
"""
Train the text-only beat/miss classifier.

Usage:
    python src/models/train.py --data-root . --freeze-encoder --epochs 10 --batch-size 8
    python src/models/train.py --data-root . --epochs 15 --batch-size 8 --lr 2e-5
    python src/models/train.py --data-root . --freeze-n-layers 8 --epochs 15
    python src/models/train.py --data-root . --lora-r 8 --lr 2e-4 --head-lr 2e-4 --epochs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import EarningsCallDataset, get_dataloaders  # noqa: E402
from model import EarningsTextClassifier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train")


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup to the base LR, then cosine decay to 0."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    losses, probs, labels = [], [], []
    for batch in loader:
        inputs = model.tokenize(batch["text"], device)
        y = batch["label"].to(device)
        logits = model(**inputs)
        losses.append(criterion(logits, y).item() * len(y))
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        labels.extend(y.cpu().tolist())

    labels_a, probs_a = np.array(labels), np.array(probs)
    preds = (probs_a >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels_a, preds, labels=[0, 1]).ravel()
    return {
        "loss": float(sum(losses) / max(1, len(labels))),
        "f1": float(f1_score(labels_a, preds, zero_division=0)),
        "auc": float(roc_auc_score(labels_a, probs_a)) if len(set(labels)) > 1 else float("nan"),
        "precision": float(precision_score(labels_a, preds, zero_division=0)),
        "recall": float(recall_score(labels_a, preds, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


METRICS = ("f1", "auc", "precision", "recall", "specificity")


def train_one(args, ds: EarningsCallDataset, seed: int, out_dir: Path, device: torch.device) -> Dict:
    """Train + evaluate one model with *seed*; returns best-epoch val and test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dl, val_dl, test_dl = get_dataloaders(
        ds, batch_size=args.batch_size, seed=seed,
        split=args.split, val_start=args.val_start, test_start=args.test_start,
    )

    model = EarningsTextClassifier(
        max_length=args.max_length,
        freeze_encoder=args.freeze_encoder,
        freeze_n_layers=args.freeze_n_layers,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

    # Inverse-frequency class weights from the training split only
    train_labels = ds.get_labels()[train_dl.dataset.indices]
    counts = np.bincount(train_labels, minlength=2).astype(np.float64)
    class_weights = torch.tensor(len(train_labels) / (2 * counts + 1e-8), dtype=torch.float32).to(device)
    logger.info("Class weights (train split): %s", class_weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    head_lr = args.head_lr if args.head_lr is not None else args.lr * 10
    param_groups = [{"params": model.head_parameters(), "lr": head_lr}]
    enc_params = model.encoder_parameters()
    if enc_params:
        param_groups.append({"params": enc_params, "lr": args.lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_dl)
    scheduler = warmup_cosine(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"
    history = []
    best_f1, bad_epochs = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for batch in train_dl:
            inputs = model.tokenize(batch["text"], device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(**inputs), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(y)

        train_loss = running / len(train_dl.dataset)
        val = evaluate(model, val_dl, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        logger.info(
            "Epoch %d/%d (%.0fs) train_loss=%.4f val_loss=%.4f F1=%.3f AUC=%.3f P=%.3f R=%.3f spec=%.3f",
            epoch, args.epochs, time.time() - t0, train_loss,
            val["loss"], val["f1"], val["auc"], val["precision"], val["recall"], val["specificity"],
        )

        if val["f1"] > best_f1:
            best_f1, bad_epochs = val["f1"], 0
            # Save only trainable weights (head + any LoRA adapters); the frozen
            # encoder is reloaded from the Hub, so this is ~1.5 MB instead of ~440 MB
            trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
            torch.save({"model_state": trainable, "model_name": model.model_name,
                        "args": vars(args), "seed": seed, "epoch": epoch, "val": val}, best_path)
            logger.info("  New best val F1 %.3f — saved %s", best_f1, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info("Early stopping after %d epochs without F1 improvement", bad_epochs)
                break

    # Final test evaluation with the best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    # strict=False: the checkpoint holds only trainable weights; frozen ones are already loaded
    missing_trainable = {n for n, p in model.named_parameters() if p.requires_grad} - set(ckpt["model_state"])
    unexpected = model.load_state_dict(ckpt["model_state"], strict=False).unexpected_keys
    if missing_trainable or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing_trainable} unexpected={unexpected}")
    test = evaluate(model, test_dl, criterion, device)
    logger.info(
        "TEST seed %d (best epoch %d): F1=%.3f AUC=%.3f P=%.3f R=%.3f spec=%.3f  [tp=%d fp=%d tn=%d fn=%d]",
        seed, ckpt["epoch"], test["f1"], test["auc"], test["precision"], test["recall"], test["specificity"],
        test["tp"], test["fp"], test["tn"], test["fn"],
    )

    with open(out_dir / "training_history.json", "w") as f:
        json.dump({"args": vars(args), "seed": seed, "history": history,
                   "best_epoch": ckpt["epoch"], "val": ckpt["val"], "test": test}, f, indent=2)
    return {"seed": seed, "best_epoch": ckpt["epoch"], "val": ckpt["val"], "test": test}


def main() -> None:
    p = argparse.ArgumentParser(description="Train the earnings beat/miss text classifier")
    p.add_argument("--data-root", default=".")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5, help="Encoder / LoRA-adapter LR")
    p.add_argument("--head-lr", type=float, default=None, help="Head LR (default: 10x --lr)")
    p.add_argument("--lora-r", type=int, default=0, help="LoRA rank; 0 disables LoRA")
    p.add_argument("--lora-alpha", type=float, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--freeze-n-layers", type=int, default=0)
    p.add_argument("--patience", type=int, default=3, help="Early-stopping patience on val F1")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                   help="Train once per seed and report mean ± std")
    p.add_argument("--split", choices=["temporal", "random"], default="temporal")
    p.add_argument("--val-start", default="2024-01-01", help="Temporal split: first val date")
    p.add_argument("--test-start", default="2025-01-01", help="Temporal split: first test date")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    ds = EarningsCallDataset(data_root=args.data_root)
    root = Path(args.output_dir)
    runs = [train_one(args, ds, seed, root / f"seed_{seed}", device) for seed in args.seeds]

    summary = {"args": vars(args), "runs": runs}
    for part in ("val", "test"):
        summary[part] = {
            m: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}
            for m in METRICS
            for v in [[r[part][m] for r in runs]]
        }
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("%d seeds, %s split", len(runs), args.split)
    for part in ("val", "test"):
        logger.info("  %-4s %s", part.upper(), "  ".join(
            f"{m}={summary[part][m]['mean']:.3f}±{summary[part][m]['std']:.3f}" for m in METRICS))
    logger.info("Saved %s", root / "summary.json")


if __name__ == "__main__":
    main()
