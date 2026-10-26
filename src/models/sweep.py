#!/usr/bin/env python3
"""
Weights & Biases hyperparameter sweep for the late-fusion model (``fusion.py``).

How a W&B sweep works:
  1. ``wandb.sweep(config)`` registers a search space on the W&B server and returns a sweep id.
  2. ``wandb.agent(sweep_id, fn)`` loops: ask the server for the next hyperparameters,
     run ``fn`` (which reads them from ``wandb.config``), report the metric back.
  3. With ``method: bayes`` the server fits a model of "hyperparameters → metric" to the
     runs so far and proposes points likely to score well, instead of sampling blindly.

What we optimise: **mean val AUC over 3 seeds**.  One seed's val AUC varies by ±0.03 —
as much as most hyperparameter changes — so a single-seed objective would mostly reward luck.

The sweep never evaluates the test set.  Picking the best of ~50 configs on a 21-miss
val set already makes the best val score optimistic; test is checked once, afterwards,
on the winning config only (``--best``).

Usage:
    wandb login                                              # once per machine
    python src/models/sweep.py --data-root . --count 50      # new sweep, 50 runs (~5 min)
    python src/models/sweep.py --data-root . --sweep-id ID --count 20   # add runs to a sweep
    python src/models/sweep.py --best ENTITY/PROJECT/ID      # print the fusion.py command for the winner
    python src/models/sweep.py --data-root . --dry-run       # one run with W&B disabled (no login)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion import load_tensors, train_one  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sweep")

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_auc_mean", "goal": "maximize"},
    "parameters": {
        "lr": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-4, "max": 3e-1},
        "history_dim": {"values": [4, 8, 16]},
        "embed_dim": {"values": [4, 8, 16, 32]},
        "hidden": {"values": [8, 16, 32, 64]},
        "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.6},
        "input_dropout": {"distribution": "uniform", "min": 0.0, "max": 0.5},
        "batch_size": {"values": [16, 32, 64]},
    },
}

# Not searched: model shape and stopping rule stay as in fusion.py
FIXED = {"blocks": ["history", "embed"], "fusion": "late", "epochs": 200, "patience": 20}

# fusion.py's defaults — used by --dry-run
DEFAULTS = {"lr": 1e-3, "weight_decay": 1e-2, "history_dim": 8, "embed_dim": 16, "hidden": 32,
            "dropout": 0.3, "input_dropout": 0.2, "batch_size": 32}


def make_run_fn(X, y, seeds, device, **init_kwargs):
    """The function each agent step calls: train ``seeds`` models with this run's config.

    Inside ``wandb.agent``, ``wandb.init()`` receives that run's hyperparameters from the server.
    """

    def run_fn() -> None:
        with wandb.init(**init_kwargs) as run:
            args = SimpleNamespace(**FIXED, **dict(run.config))
            # Per-seed learning curves, all plotted against "epoch"
            run.define_metric("epoch")
            run.define_metric("seed_*", step_metric="epoch")

            val_aucs, best_epochs = [], []
            for seed in seeds:
                def log_epoch(rec, seed=seed):
                    run.log({"epoch": rec["epoch"],
                             f"seed_{seed}/train_loss": rec["train_loss"],
                             f"seed_{seed}/val_auc": rec["val_auc"]})

                result, _ = train_one(args, X, y, seed, device, eval_test=False, on_epoch=log_epoch)
                val_aucs.append(result["val"]["auc"])
                best_epochs.append(result["best_epoch"])

            run.summary["val_auc_mean"] = float(np.mean(val_aucs))
            run.summary["val_auc_std"] = float(np.std(val_aucs))
            run.summary["best_epoch_mean"] = float(np.mean(best_epochs))
            logger.info("val AUC %.3f ± %.3f  config=%s", np.mean(val_aucs), np.std(val_aucs), dict(run.config))

    return run_fn


def print_best(sweep_path: str) -> None:
    """Fetch the sweep's best run and print the fusion.py command that tests it."""
    sweep = wandb.Api().sweep(sweep_path)
    runs = sorted((r for r in sweep.runs if "val_auc_mean" in r.summary),
                  key=lambda r: r.summary["val_auc_mean"], reverse=True)
    logger.info("%d finished runs. Top 5 by mean val AUC:", len(runs))
    for r in runs[:5]:
        logger.info("  %s  val %.3f ± %.3f  %s", r.name, r.summary["val_auc_mean"],
                    r.summary["val_auc_std"], {k: r.config[k] for k in SWEEP_CONFIG["parameters"]})
    cfg = runs[0].config
    flags = " ".join(f"--{k.replace('_', '-')} {cfg[k]:.6g}" if isinstance(cfg[k], float)
                     else f"--{k.replace('_', '-')} {cfg[k]}" for k in SWEEP_CONFIG["parameters"])
    print(f"\npython src/models/fusion.py --data-root . {flags} "
          f"--output-dir checkpoints/fusion/late_swept")


def main() -> None:
    p = argparse.ArgumentParser(description="W&B sweep over late-fusion hyperparameters")
    p.add_argument("--data-root", default=".")
    p.add_argument("--project", default="earnings-lie-detector")
    p.add_argument("--entity", default=None, help="W&B user/team (default: your account)")
    p.add_argument("--count", type=int, default=50, help="Runs for this agent")
    p.add_argument("--sweep-id", default=None, help="Join an existing sweep instead of creating one")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--val-start", default="2024-01-01")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--best", metavar="ENTITY/PROJECT/SWEEP_ID", default=None,
                   help="Don't train; report the best run of this sweep")
    p.add_argument("--dry-run", action="store_true", help="One run with fusion.py defaults, W&B disabled")
    args = p.parse_args()

    if args.best:
        print_best(args.best)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = load_tensors(args.data_root, FIXED["blocks"], args.val_start, args.test_start, device)
    if args.dry_run:
        make_run_fn(X, y, args.seeds, device, mode="disabled", config=DEFAULTS)()
        return

    run_fn = make_run_fn(X, y, args.seeds, device)

    sweep_id = args.sweep_id or wandb.sweep(SWEEP_CONFIG, project=args.project, entity=args.entity)
    wandb.agent(sweep_id, run_fn, count=args.count, project=args.project, entity=args.entity)


if __name__ == "__main__":
    main()
