#!/usr/bin/env python3
"""
Package the swept late-fusion model for the Hugging Face Hub.

Steps:
  1. Train it exactly as evaluated: sweep-winner config (``evaluate.FUSION_CFG``),
     train < 2024, early-stopped on 2024, seed 42.
  2. Wrap it in ``HubFusionNet`` with the train-set scaler statistics, so it takes raw features.
  3. ``save_pretrained`` → ``config.json`` + ``model.safetensors``; add a model card
     (README.md) and ``fusion_net.py`` (the class code needed to load it).
  4. Reload with ``from_pretrained`` and check predictions are identical.
  5. Only with ``--push``: create the repo (private unless ``--public``) and upload the folder.

Usage:
    python src/models/push_to_hub.py --data-root .                       # build + verify locally
    hf auth login                                                        # once per machine
    python src/models/push_to_hub.py --data-root . --repo-id USER/earnings-miss-fusion --push
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import HISTORY_COLS  # noqa: E402
from evaluate import FUSION_CFG  # noqa: E402
from fusion import load_tensors, train_one  # noqa: E402
from fusion_net import HubFusionNet  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("push_to_hub")

HISTORY_DOC = {
    "prev_miss": "1 if the company missed the EPS estimate at its previous announcement, else 0",
    "prev_surprise": "previous announcement's EPS surprise in %, clipped to [-100, 100]",
    "miss_rate_4q": "share of misses over the last 4 announcements",
    "mean_surprise_4q": "mean EPS surprise % over the last 4 announcements",
    "std_surprise_4q": "std of EPS surprise % over the last 4 announcements (0 if < 2)",
}


def model_card(repo_id: str, license_id: str, m: dict, pooled: dict) -> str:
    rows = "\n".join(f"| {i} | `{c}` | {HISTORY_DOC[c]} |" for i, c in enumerate(HISTORY_COLS))
    p = pooled.get("fusion_swept", {})
    h = pooled.get("history", {})
    pooled_txt = (f"| this model (3 seeds) | {p['auc']:.3f} | {p['delta_vs_history']:+.3f} "
                  f"[{p['delta_ci95'][0]:+.3f}, {p['delta_ci95'][1]:+.3f}] | {p['prec_top10']:.0%} |\n"
                  f"| logistic regression, history only | {h['auc']:.3f} | — | {h['prec_top10']:.0%} |"
                  if p else "_(run `src/models/evaluate.py` to fill this table)_")
    return f"""---
license: {license_id}
library_name: pytorch
tags:
- tabular-classification
- finance
- earnings
- sec-filings
- pytorch_model_hub_mixin
metrics:
- roc_auc
---

# Earnings miss predictor (history + MD&A embedding fusion)

Predicts the probability that a US company will **miss the consensus EPS estimate at its next
quarterly earnings announcement**, from:

- the company's own recent earnings track record (5 numbers), and
- a 384-dim embedding of the MD&A section of its most recent 10-Q, filed ~90 days before that announcement.

A small late-fusion network (~{m['n_params']:,} parameters): history 5 → 16, embedding 384 → 4 (each
Linear + ReLU), concatenated → MLP(16) → 1 logit. It was trained as a learning project; see the
[GitHub repo](https://github.com/guru-prasath-j/earnings-lie-detector) for the full pipeline and docs.

> **Read the limitations below.** The useful signal comes from the company's track record. The text
> embedding mostly identifies *which company* wrote the filing; it is not a "lie detector".

## Usage

```python
# pip install torch huggingface_hub; download fusion_net.py from this repo
from fusion_net import HubFusionNet

model = HubFusionNet.from_pretrained("{repo_id}")
p_miss = model.predict_proba(history, embedding)   # history (n, 5), embedding (n, 384) -> (n,)
```

Inputs are **raw** (unscaled). The model standardises them with training-set statistics stored as buffers.

**History features** (in this column order; computed only from announcements *before* the one predicted):

| # | Feature | Definition |
|---|---|---|
{rows}

**Embedding:** MD&A text → 256-token chunks → `{m['embedder']}` mean-pooled per chunk → average of chunk
vectors (384-dim, not normalised). See `src/models/text_features.py::document_embeddings`.

## Training data

- 661 (10-Q, next announcement) pairs from 49 large US companies, 2020–2026; 13.9% misses.
- Text: MD&A sections from SEC EDGAR 10-Q filings. Labels: EPS actual vs estimate from Yahoo Finance
  (`yfinance`). The data itself is **not** included here.
- Temporal split: train 2020–2023 (370, 48 misses), validation 2024 (122, 21; early stopping and the
  W&B hyperparameter sweep), test 2025–2026 (169, 23).
- Class-weighted BCE (`pos_weight` ≈ 6.7), AdamW, early stopping on validation AUC. Seed 42.

## Evaluation

This checkpoint (single split):

| | AUC | F1 @ 0.5 | Precision | Recall |
|---|---|---|---|---|
| Validation 2024 | {m['val']['auc']:.3f} | {m['val']['f1']:.3f} | {m['val']['precision']:.3f} | {m['val']['recall']:.3f} |
| Test 2025–26 | {m['test']['auc']:.3f} | {m['test']['f1']:.3f} | {m['test']['precision']:.3f} | {m['test']['recall']:.3f} |

The test set has only 23 misses (95% bootstrap AUC range roughly ±0.11). A more reliable estimate is
the **rolling-origin evaluation** over test years 2023, 2024 and 2025–26 (403 samples, 58 misses):

| Model | Pooled AUC | ΔAUC vs history [95% paired bootstrap] | Precision in top 10% |
|---|---|---|---|
{pooled_txt}

Base rate 14.4%. Flagging the riskiest 10% of announcements each year finds misses at roughly 2.5–3× the base rate.

## Limitations

- **Text signal is mostly company identity.** In company-held-out cross-validation, an embedding-only
  model drops from AUC 0.67 to 0.48 (chance), while history-only holds at 0.68. Adding the embedding
  to history gives a small gain whose confidence interval includes zero.
- Small sample (49 companies, mostly large caps); not validated on other universes or periods.
- Probabilities come from class-weighted training, so they are **not calibrated** (they overstate the
  miss rate). Use them for ranking.
- The hyperparameters were tuned on 2024, which is also a test year in the rolling evaluation.
- **Not investment advice.** Educational project only.
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Package the fusion model for the Hugging Face Hub")
    p.add_argument("--data-root", default=".")
    p.add_argument("--out-dir", default="checkpoints/hub/earnings-miss-fusion")
    p.add_argument("--repo-id", default="USER/earnings-miss-fusion", help="Hub repo, e.g. alice/earnings-miss-fusion")
    p.add_argument("--license", default="mit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--push", action="store_true", help="Upload to the Hub (needs `hf auth login`)")
    p.add_argument("--public", action="store_true", help="Create the repo public (default: private)")
    args = p.parse_args()

    device = torch.device("cpu")  # tiny model; CPU keeps saved tensors device-free
    cfg = SimpleNamespace(**FUSION_CFG)

    # 1. train exactly as evaluated
    X, y, scalers = load_tensors(args.data_root, cfg.blocks, "2024-01-01", "2025-01-01", device, return_scalers=True)
    result, probs, net = train_one(cfg, X, y, args.seed, device, return_model=True)

    # 2. wrap with the scaler statistics
    hub = HubFusionNet(blocks=cfg.blocks, in_dims={b: X["train"][b].shape[1] for b in cfg.blocks},
                       branch_dims={"history": cfg.history_dim, "embed": cfg.embed_dim},
                       hidden=cfg.hidden, dropout=cfg.dropout, input_dropout=cfg.input_dropout,
                       history_features=HISTORY_COLS)
    hub.net.load_state_dict(net.state_dict())
    for b, sc in scalers.items():
        hub.set_scaler(b, sc.mean_, sc.scale_)

    # Raw test features (undo the standardisation) must give the same probabilities
    raw = {b: scalers[b].inverse_transform(X["test"][b].numpy()) for b in cfg.blocks}
    p_hub = hub.predict_proba(raw["history"], raw["embed"])
    assert np.allclose(p_hub, probs["test"], atol=1e-5), "wrapper changed the predictions"

    # 3. save weights + config, then card and class code
    out = Path(args.out_dir)
    hub.save_pretrained(out)
    pooled_path = Path("checkpoints/evaluation.json")
    pooled = json.loads(pooled_path.read_text())["pooled"] if pooled_path.exists() else {}
    m = {"val": result["val"], "test": result["test"], "embedder": hub.embedder,
         "n_params": sum(t.numel() for t in hub.net.parameters())}
    (out / "README.md").write_text(model_card(args.repo_id, args.license, m, pooled))
    shutil.copy(Path(__file__).with_name("fusion_net.py"), out / "fusion_net.py")

    # 4. round trip
    reloaded = HubFusionNet.from_pretrained(out)
    assert np.allclose(reloaded.predict_proba(raw["history"], raw["embed"]), p_hub, atol=1e-6)
    logger.info("Saved and verified %s: %s", out, sorted(f.name for f in out.iterdir()))
    logger.info("VAL auc=%.3f  TEST auc=%.3f f1=%.3f  (%d params)",
                result["val"]["auc"], result["test"]["auc"], result["test"]["f1"], m["n_params"])

    # 5. upload
    if args.push:
        if args.repo_id.startswith("USER/"):
            sys.exit("Pass --repo-id <your-username>/<name>")
        api = HfApi()
        url = api.create_repo(args.repo_id, private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, folder_path=out, commit_message="Upload earnings miss fusion model")
        logger.info("Pushed to %s (%s)", url, "public" if args.public else "private")


if __name__ == "__main__":
    main()
