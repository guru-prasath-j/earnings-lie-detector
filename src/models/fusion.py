#!/usr/bin/env python3
"""
Fusion model: combine the company's earnings *history* with the filing's *text*
embedding in one small neural network, trained with a hand-written PyTorch loop.

Inputs per sample (same table and temporal split as ``baselines.py``):
  history — 5 numbers: prior-quarter miss/surprise, last-4-quarter stats
  embed   — 384-dim MiniLM embedding of the whole MD&A

Two ways to fuse them (``--fusion``):

  early:  [history ; embed] (389) ──> MLP ──> logit
          One network sees everything at once.  The 384 text dims can drown out
          the 5 history dims — same problem the PCA blocks solved in baselines.py.

  late:   history (5)   ──> Linear ──> ReLU ──> 8  ┐
                                                   ├─ concat ──> MLP ──> logit
          embed   (384) ──> Linear ──> ReLU ──> 16 ┘
          Each input gets its own small "branch" that squeezes it to a few
          learned features first, so both sides arrive at the head on equal footing.

``--blocks history`` or ``--blocks embed`` trains a single-branch model, the
neural counterpart of the logistic-regression baselines.

The training loop is deliberately written out by hand (no DataLoader, no Trainer):
the whole dataset fits on the GPU as two tensors, so one epoch is just
"shuffle the row indices, walk through them in mini-batches".

Usage:
    python src/models/fusion.py --data-root .                         # late fusion, 5 seeds
    python src/models/fusion.py --data-root . --fusion early
    python src/models/fusion.py --data-root . --blocks history
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import HISTORY_COLS, auc_ci, build_table, metrics, temporal_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fusion")


class FusionNet(nn.Module):
    """One branch per input block, concatenated, then a small MLP head → 1 logit.

    ``branch_dims`` maps block name → branch output size.  ``early=True`` skips
    the branches and feeds the raw concatenated inputs straight to the head.
    """

    def __init__(self, in_dims: Dict[str, int], branch_dims: Dict[str, int], hidden: int = 32,
                 dropout: float = 0.3, input_dropout: float = 0.2, early: bool = False) -> None:
        super().__init__()
        self.blocks = list(in_dims)
        self.early = early
        if early:
            self.branches = nn.ModuleDict()
            head_in = sum(in_dims.values())
        else:
            # Input dropout on the wide text block: randomly hides some of the 384 dims
            # each step so the branch can't memorise a few of them.
            self.branches = nn.ModuleDict({
                b: nn.Sequential(nn.Dropout(input_dropout if in_dims[b] > 32 else 0.0),
                                 nn.Linear(in_dims[b], branch_dims[b]), nn.ReLU())
                for b in self.blocks
            })
            head_in = sum(branch_dims[b] for b in self.blocks)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.early:
            z = torch.cat([x[b] for b in self.blocks], dim=1)
        else:
            z = torch.cat([self.branches[b](x[b]) for b in self.blocks], dim=1)
        return self.head(z).squeeze(1)  # (batch,) logits; sigmoid(logit) = P(miss)


@torch.no_grad()
def predict(model: nn.Module, X: Dict[str, torch.Tensor]) -> np.ndarray:
    model.eval()  # turns dropout off
    return torch.sigmoid(model(X)).cpu().numpy()


def load_tensors(data_root: str, blocks: List[str], val_start: str, test_start: str, device: torch.device
                 ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Feature blocks per split, standardised on train only, already on ``device``."""
    df, emb_cols = build_table(data_root)
    parts = temporal_split(df, val_start, test_start)
    cols = {"history": HISTORY_COLS, "embed": emb_cols}
    X: Dict[str, Dict[str, torch.Tensor]] = {part: {} for part in parts}
    for b in blocks:
        scaler = StandardScaler().fit(parts["train"][cols[b]])
        for part, frame in parts.items():
            X[part][b] = torch.tensor(scaler.transform(frame[cols[b]]), dtype=torch.float32, device=device)
    y = {part: torch.tensor(frame["label"].to_numpy(), dtype=torch.float32, device=device)
         for part, frame in parts.items()}
    return X, y


def train_one(args, X: Dict[str, Dict[str, torch.Tensor]], y: Dict[str, torch.Tensor],
              seed: int, device: torch.device, eval_test: bool = True,
              on_epoch: Optional[Callable[[Dict], None]] = None) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """Train one model; returns (result, probabilities).

    ``eval_test=False`` never touches the test set (used by the sweep).
    ``on_epoch`` is called with each epoch's record (e.g. to log it to W&B).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dims = {b: X["train"][b].shape[1] for b in args.blocks}
    branch_dims = {"history": args.history_dim, "embed": args.embed_dim}
    model = FusionNet(in_dims, branch_dims, hidden=args.hidden, dropout=args.dropout,
                      input_dropout=args.input_dropout, early=args.fusion == "early").to(device)

    # pos_weight = (#beats / #misses): each miss counts ~6x in the loss
    y_tr = y["train"]
    pos_weight = (len(y_tr) - y_tr.sum()) / y_tr.sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n = len(y_tr)
    best_auc, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        # ---- train: one pass over the shuffled training rows ----
        model.train()  # dropout on
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            xb = {b: X["train"][b][idx] for b in args.blocks}
            yb = y_tr[idx]

            logits = model(xb)              # 1. forward
            loss = criterion(logits, yb)    # 2. loss (mean over the batch)
            optimizer.zero_grad()           # 3. clear last step's gradients
            loss.backward()                 # 4. backprop: fill .grad on every parameter
            optimizer.step()                # 5. update weights using .grad
            total += loss.item() * len(idx)

        # ---- validate ----
        p_val = predict(model, X["val"])
        val_auc = roc_auc_score(y["val"].cpu().numpy(), p_val)
        history.append({"epoch": epoch, "train_loss": total / n, "val_auc": float(val_auc)})
        if on_epoch:
            on_epoch(history[-1])

        # ---- early stopping on val AUC (smooth, threshold-free; val F1 is too jumpy) ----
        if val_auc > best_auc:
            best_auc, best_epoch, bad_epochs = val_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())  # snapshot, not a reference
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    model.load_state_dict(best_state)  # roll back to the best epoch
    parts = ("val", "test") if eval_test else ("val",)
    probs = {part: predict(model, X[part]) for part in parts}
    result = {"seed": seed, "best_epoch": best_epoch, "epochs_run": epoch, "history": history}
    for part in parts:
        result[part] = metrics(y[part].cpu().numpy(), probs[part])
    msg = "seed %d: best epoch %d/%d  VAL auc=%.3f f1=%.3f" % (
        seed, best_epoch, epoch, result["val"]["auc"], result["val"]["f1"])
    if eval_test:
        msg += " | TEST auc=%.3f f1=%.3f" % (result["test"]["auc"], result["test"]["f1"])
    logger.info(msg)
    return result, probs


def main() -> None:
    p = argparse.ArgumentParser(description="History + text-embedding fusion model")
    p.add_argument("--data-root", default=".")
    p.add_argument("--blocks", nargs="+", default=["history", "embed"], choices=["history", "embed"])
    p.add_argument("--fusion", choices=["late", "early"], default="late")
    p.add_argument("--history-dim", type=int, default=8, help="History branch output size")
    p.add_argument("--embed-dim", type=int, default=16, help="Text branch output size")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--input-dropout", type=float, default=0.2, help="Dropout on the 384-dim embedding input")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20, help="Epochs without val-AUC gain before stopping")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--val-start", default="2024-01-01")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--output-dir", default=None, help="Default: checkpoints/fusion/<config>")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = "+".join(args.blocks) + ("" if len(args.blocks) == 1 else f"_{args.fusion}")
    out_dir = Path(args.output_dir or f"checkpoints/fusion/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_tensors(args.data_root, args.blocks, args.val_start, args.test_start, device)

    runs, probs = [], []
    for seed in args.seeds:
        r, pr = train_one(args, X, y, seed, device)
        runs.append(r)
        probs.append(pr)

    # Seed ensemble: average the probabilities of all seeds' models
    y_np = {part: y[part].cpu().numpy() for part in ("val", "test")}
    ens = {part: np.mean([pr[part] for pr in probs], axis=0) for part in ("val", "test")}
    ensemble = {"val": metrics(y_np["val"], ens["val"]), "test": metrics(y_np["test"], ens["test"])}
    ensemble["test"]["auc_ci95"] = auc_ci(y_np["test"], ens["test"])

    def agg(part: str, key: str) -> Dict[str, float]:
        v = [r[part][key] for r in runs]
        return {"mean": float(np.mean(v)), "std": float(np.std(v))}

    summary = {
        "args": vars(args),
        "val": {k: agg("val", k) for k in ("auc", "f1")},
        "test": {k: agg("test", k) for k in ("auc", "f1")},
        "seed_ensemble": ensemble,
        "runs": runs,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lo, hi = ensemble["test"]["auc_ci95"]
    logger.info("=" * 60)
    logger.info("%s, %d seeds", name, len(args.seeds))
    logger.info("  per seed  VAL auc=%.3f±%.3f | TEST auc=%.3f±%.3f f1=%.3f±%.3f",
                summary["val"]["auc"]["mean"], summary["val"]["auc"]["std"],
                summary["test"]["auc"]["mean"], summary["test"]["auc"]["std"],
                summary["test"]["f1"]["mean"], summary["test"]["f1"]["std"])
    logger.info("  ensemble  VAL auc=%.3f | TEST auc=%.3f [%.2f–%.2f] f1=%.3f",
                ensemble["val"]["auc"], ensemble["test"]["auc"], lo, hi, ensemble["test"]["f1"])
    logger.info("Saved %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
