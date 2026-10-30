#!/usr/bin/env python3
"""
Package a model for the Hugging Face Hub.

``--model miss`` (default, since 2026-10-29): logistic regression on history + sector + guidance
rules, the best model on 188 companies (``docs/13``).  ``--model fusion``: the earlier swept
late-fusion net (history + MD&A embedding, 49 companies; ``docs/11``).

Steps (both models):
  1. Train exactly as evaluated, and report the held-out test.
     miss: C chosen on 2024 val (train < 2024), test 2025–26; then **refit on all years** with that C
     for release.  fusion: sweep-winner config, train < 2024, early-stopped on 2024, seed 42.
  2. Wrap it in ``HubMissModel`` / ``HubFusionNet`` with the train-set scaler statistics.
  3. ``save_pretrained`` → ``config.json`` + ``model.safetensors``; add a model card (README.md)
     and the class code needed to load it.
  4. Reload with ``from_pretrained`` and check predictions are identical (miss: also from raw text).
  5. Only with ``--push``: create the repo (private unless ``--public``) and upload the folder,
     deleting the other model's code file from the repo.

Usage:
    python src/models/push_to_hub.py --data-root .                       # build + verify locally
    hf auth login                                                        # once per machine
    python src/models/push_to_hub.py --data-root . --repo-id USER/lie-detect --push
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
from miss_model import HubMissModel, history_vector  # noqa: E402

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


# Picked by mean val AUC 2023/2024 (docs/14): 0.692 vs 0.682 without guidance, 0.677 without market data
MISS_MODEL = "history+sector+analyst+price+guidance"


def miss_card(repo_id: str, license_id: str, m: dict, ev: dict) -> str:
    pooled, held, folds = ev["pooled"], ev["company_holdout"], ev["folds"]
    me, hs, h = pooled[MISS_MODEL], pooled["history+sector"], pooled["history"]
    vs = me["vs_history+sector"]
    hv = held[MISS_MODEL]["vs_history+sector"]
    hist_rows = "\n".join(f"| `{c}` | {HISTORY_DOC[c]} |" for c in HISTORY_COLS)
    year_rows = "\n".join(f"| {f} | {v['n'][2]:,} / {v['miss'][2]} | {v['history+sector']['auc']:.3f} | "
                          f"{v[MISS_MODEL]['auc']:.3f} | {v[MISS_MODEL]['auc'] - v['history+sector']['auc']:+.3f} |"
                          for f, v in folds.items())
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

# Earnings miss predictor (track record, sector, analysts, price, guidance)

Predicts the probability that a US company will **miss the consensus EPS estimate at its next
quarterly earnings announcement**, about 90 days ahead, from:

- its own recent **earnings surprises** (5 numbers),
- its **GICS sector** (10 sectors; Real Estate not supported),
- **analyst actions** in the 90 days before: rating changes and price targets raised or cut (4 numbers),
- the stock's **run-up** vs the S&P 500 (SPY) over 1 and 3 months, and its volatility (3 numbers), and
- what its latest **earnings press release** says about guidance: raised, cut, maintained,
  narrowed, introduced or withdrawn, and how hedged the outlook is (10 numbers from regex rules).

It is a **logistic regression** ({m['n_features']} features → 1 logit) wrapped as a PyTorch module. It was built as a
learning project; the [GitHub repo](https://github.com/guru-prasath-j/earnings-lie-detector) has the
full pipeline and the docs, including every idea that did *not* work.

> **What the evaluation says:** most of the skill comes from the track record and the sector. On companies
> the model never saw, analyst, price and guidance features add **{hv['delta_vs_history+sector']:+.3f} AUC** over those two
> [{hv['delta_ci95'][0]:+.3f}, {hv['delta_ci95'][1]:+.3f}]. Forward in time the gain is **{vs['delta_vs_history+sector']:+.3f}**
> [{vs['delta_ci95'][0]:+.3f}, {vs['delta_ci95'][1]:+.3f}]: it helped in 2023 and 2024 but not in 2025–26 (table below).

## Usage

```python
# pip install torch huggingface_hub numpy pandas yfinance
# download miss_model.py, guidance_rules.py and market_features.py from this repo
from miss_model import HubMissModel

model = HubMissModel.from_pretrained("{repo_id}")
p_miss = model.predict(
    past_surprises=[4.2, -1.5, 2.8, 6.0],   # EPS surprise % of earlier announcements, oldest → newest
    sector="Industrials",                     # one of model.sectors
    release_text=open("press_release.txt").read(),   # latest earnings press release (8-K Exhibit 99.1)
    ticker="CAT",                             # analyst actions and prices are fetched from Yahoo Finance
    announcement_date="2026-11-04",           # the announcement being predicted (YYYY-MM-DD)
)
```

Pass `market_inputs=(analyst_actions, prices, spy_prices)` instead of `ticker` to use your own data
(format: `market_features.market_vector`). `model.predict_proba(X)` takes a raw feature matrix in
`model.features` order; the model standardises inputs with training-set statistics stored as buffers.

**History features** (computed from announcements *before* the one predicted):

| Feature | Definition |
|---|---|
{hist_rows}

**Sector:** one-hot over `model.sectors` ({", ".join(m["sectors"])}).

**Analyst and price features:** `market_features.market_vector`. Windows end 2 days before the
announcement date, so the release and the reactions to it never leak in. Analyst actions (last 90 days):
log count, upgrades − downgrades, share of price targets raised − lowered, mean % target change. Price:
1- and 3-month return minus SPY's, 3-month volatility.

**Guidance features:** `guidance_rules.guidance_vector(text)` finds sentences mentioning
guidance / outlook / forecast (minus boilerplate) and takes the direction verb closest to that noun.
About 85% precise on spot checks. Use the text of the press release, without tables.

## Training data

- {m['n']:,} (press release, next announcement) pairs from {m['n_companies']} S&P 500 companies, 2017–2026;
  {m['miss_rate']:.1%} misses. The release is filed on announcement day; it is paired with the *next*
  announcement (83–98 days later), never its own.
- Text: 8-K Item 2.02 Exhibit 99.1 from SEC EDGAR. Labels, analyst actions and prices: Yahoo Finance.
  The data itself is **not** included here.
- Logistic regression with balanced class weights; C = {m['C']} chosen on validation 2024 (train < 2024).
  **The released weights are refit on all years** with that C.

## Evaluation

Rolling-origin test: each year predicted by a model trained only on earlier years
({ev['_n']:,} announcements, {ev['_miss']} misses):

| Model | Pooled AUC | ΔAUC vs history [95%] | Precision in top 10% |
|---|---|---|---|
| history only | {h['auc']:.3f} | — | {h['prec_top10']:.0%} |
| history + sector | {hs['auc']:.3f} | {hs['delta_vs_history']:+.3f} [{hs['delta_ci95'][0]:+.3f}, {hs['delta_ci95'][1]:+.3f}] | {hs['prec_top10']:.0%} |
| **this model** | **{me['auc']:.3f}** | {me['delta_vs_history']:+.3f} [{me['delta_ci95'][0]:+.3f}, {me['delta_ci95'][1]:+.3f}] | {me['prec_top10']:.0%} |

By test year:

| Test year | Announcements / misses | history + sector | this model | Δ |
|---|---|---|---|---|
{year_rows}

Base rate {ev['_base']:.1%}. On companies held out of training (5-fold by company): AUC
{held[MISS_MODEL]['by_company']:.3f}, {hv['delta_vs_history+sector']:+.3f} [{hv['delta_ci95'][0]:+.3f}, {hv['delta_ci95'][1]:+.3f}] over history + sector. The analyst + price
part of that gain (+0.019) is unchanged when each test row's own year is left out of training, so it
is not a same-quarter market shock.

## Limitations

- **Modest skill.** AUC ≈ 0.68 ranks misses better than chance, not reliably. Flagging the riskiest
  10% finds misses at {me["prec_top10"] / ev["_base"]:.1f}× the base rate.
- **The most recent test period (2025–26) shows no gain** over history + sector. With three test years
  we can't tell a change in markets from noise.
- **Depends on Yahoo Finance** at prediction time. Yahoo's analyst-action history is used as if it were
  complete and unrevised; that isn't guaranteed.
- **The text adds little** on its own; an LLM reading the same guidance passages did no better.
- **Survivorship:** companies are today's S&P 500 members (plus 49 large caps); REITs are excluded
  because their EPS labels mix FFO and GAAP bases.
- Probabilities come from class-weighted training, so they are **not calibrated** (they overstate the
  miss rate). Use them for ranking.
- **Not investment advice.** Educational project only.
"""


def build_miss_model(args) -> None:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from baselines import build_table
    from evaluate import lr_fold
    from market_features import ANALYST_COLS, PRICE_COLS, _prices, market_vector, prepare_analyst, yahoo_inputs
    from text_features import GUIDANCE_COLS
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from universe import load_sectors

    df, _ = build_table(args.data_root, text="press")
    sector = pd.get_dummies(df.ticker.map(load_sectors()), prefix="sec", dtype=float)
    df = pd.concat([df, sector], axis=1)
    sec_cols = list(sector.columns)
    cols = {"history": HISTORY_COLS, "sector": sec_cols, "guidance": GUIDANCE_COLS,
            "analyst": ANALYST_COLS, "price": PRICE_COLS}
    blocks = MISS_MODEL.split("+")
    feats = [c for b in blocks for c in cols[b]]  # same order as evaluate.make_blocks

    # 1a. evaluation protocol: train < 2024, C on 2024 val, test 2025–26 (same code as evaluate.py)
    tr, va, te = df[df.date < "2024-01-01"], df[(df.date >= "2024-01-01") & (df.date < "2025-01-01")], df[df.date >= "2025-01-01"]
    _, p_te, C = lr_fold(MISS_MODEL, tr, va, te, cols)
    test_auc = roc_auc_score(te.label, p_te)
    # 1b. release: refit on all years with that C
    sc = StandardScaler().fit(df[feats])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(sc.transform(df[feats]), df.label)

    # 2. wrap; px_vol_63 of a new listing is filled with the training median (as market_features does)
    market_dir = Path(args.data_root) / "data" / "raw" / "market"
    vol_fill = float(df["px_vol_63"].median())
    hub = HubMissModel(features=feats, sectors=[c[len("sec_"):] for c in sec_cols], C=C, blocks=blocks,
                       fills={"px_rel_63": 0.0, "px_rel_21": 0.0, "px_vol_63": vol_fill})
    with torch.no_grad():
        hub.linear.weight.copy_(torch.as_tensor(clf.coef_, dtype=torch.float32))
        hub.linear.bias.copy_(torch.as_tensor(clf.intercept_, dtype=torch.float32))
        hub.mean.copy_(torch.as_tensor(sc.mean_, dtype=torch.float32))
        hub.std.copy_(torch.as_tensor(sc.scale_, dtype=torch.float32))
    p_sk = clf.predict_proba(sc.transform(df[feats]))[:, 1]
    assert np.allclose(hub.predict_proba(df[feats].to_numpy()), p_sk, atol=1e-5), "wrapper changed the predictions"

    # Raw path: past surprises + sector name + release text + market inputs → same features as the table
    labels = pd.read_csv(Path(args.data_root) / "data" / "processed" / "labels.csv").sort_values(["ticker", "date"])
    sectors, spy, inputs = load_sectors(), _prices(market_dir, "SPY"), {}
    rows = df.sample(300, random_state=0)
    for _, r in rows.iterrows():
        if r.ticker not in inputs:
            f = market_dir / f"{r.ticker}_analyst.csv"
            inputs[r.ticker] = (prepare_analyst(pd.read_csv(f)) if f.exists() else pd.DataFrame(),
                                _prices(market_dir, r.ticker), spy)
        past = labels[(labels.ticker == r.ticker) & (labels.date < r.date)].eps_surprise_pct.tolist()
        raw = hub.feature_vector(past, sectors[r.ticker], r.text, announcement_date=r.date, market_inputs=inputs[r.ticker])
        assert np.allclose(raw, r[feats].to_numpy(dtype=float), atol=1e-6), (r.ticker, r.date)
    logger.info("Raw-input path matches the training table on %d sampled rows", len(rows))

    # Live Yahoo fetch (what a user gets) vs the cached files, on a few recent rows
    live = rows[rows.date >= "2024-01-01"].head(5)
    for _, r in live.iterrows():
        a = market_vector(*yahoo_inputs(r.ticker), r.date)
        diff = max(abs(a[c] - r[c]) for c in ANALYST_COLS + PRICE_COLS)
        logger.info("  live vs cached %s %s: max |diff| %.2g", r.ticker, r.date, diff)

    # 3. save weights + config, card, class code
    out = Path(args.out_dir or "checkpoints/hub/earnings-miss-model")
    hub.save_pretrained(out)
    ev_raw = json.loads(Path("checkpoints/evaluation_press.json").read_text())
    base = ev_raw["pooled"]["_base_rate"]
    n_test = sum(ev_raw["folds"][f]["n"][2] for f in ev_raw["folds"])
    miss_test = sum(ev_raw["folds"][f]["miss"][2] for f in ev_raw["folds"])
    ev = {**ev_raw, "_base": base, "_n": n_test, "_miss": miss_test}
    m = {"n_features": len(feats), "sectors": hub.sectors, "n": len(df), "n_companies": df.ticker.nunique(),
         "miss_rate": df.label.mean(), "C": C, "test_auc": test_auc, "test_n": len(te), "test_miss": int(te.label.sum())}
    (out / "README.md").write_text(miss_card(args.repo_id, args.license, m, ev))
    for f in ("miss_model.py", "guidance_rules.py", "market_features.py"):
        shutil.copy(Path(__file__).with_name(f), out / f)

    # 4. round trip
    reloaded = HubMissModel.from_pretrained(out)
    assert np.allclose(reloaded.predict_proba(df[feats].to_numpy()), p_sk, atol=1e-5)
    assert reloaded.features == feats and reloaded.sectors == hub.sectors and reloaded.blocks == blocks
    logger.info("Saved and verified %s: %s", out, sorted(f.name for f in out.iterdir()))
    logger.info("%s  C=%s  test 2025-26 AUC=%.3f (train<2024)  released: refit on %d rows, %d companies",
                MISS_MODEL, C, test_auc, len(df), df.ticker.nunique())
    _push(args, out, f"Upload earnings miss model ({MISS_MODEL})", delete=["fusion_net.py"])


def _push(args, out: Path, message: str, delete: list) -> None:
    if not args.push:
        return
    if args.repo_id.startswith("USER/"):
        sys.exit("Pass --repo-id <your-username>/<name>")
    api = HfApi()
    url = api.create_repo(args.repo_id, private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, folder_path=out, commit_message=message, delete_patterns=delete)
    logger.info("Pushed to %s (new repos are %s; existing repos keep their visibility)", url,
                "public" if args.public else "private")


def main() -> None:
    p = argparse.ArgumentParser(description="Package a model for the Hugging Face Hub")
    p.add_argument("--model", default="miss", choices=["miss", "fusion"],
                   help="miss = LR history+sector+guidance (188 companies); fusion = earlier fusion net")
    p.add_argument("--data-root", default=".")
    p.add_argument("--out-dir", default=None,
                   help="default checkpoints/hub/earnings-miss-model (miss) / earnings-miss-fusion (fusion)")
    p.add_argument("--repo-id", default="USER/lie-detect", help="Hub repo, e.g. alice/lie-detect")
    p.add_argument("--license", default="mit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--push", action="store_true", help="Upload to the Hub (needs `hf auth login`)")
    p.add_argument("--public", action="store_true", help="Create the repo public (default: private)")
    args = p.parse_args()
    if args.model == "miss":
        build_miss_model(args)
    else:
        build_fusion(args)


def build_fusion(args) -> None:
    args.out_dir = args.out_dir or "checkpoints/hub/earnings-miss-fusion"
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
    _push(args, out, "Upload earnings miss fusion model", delete=["miss_model.py", "guidance_rules.py"])


if __name__ == "__main__":
    main()
