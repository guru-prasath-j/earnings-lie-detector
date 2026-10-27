#!/usr/bin/env python3
"""
Evaluation deep dive: how sure can we be about the model comparisons?

The single 2025–26 test window has only 23 misses, so every AUC has a ±0.11 bootstrap
range and no model difference is clearly real.  This script asks four questions:

1. Rolling-origin evaluation — repeat train → val → test for several test years,
   each fold trained only on the past:
       fold 2023:    train 2020–21, val 2022, test 2023
       fold 2024:    train 2020–22, val 2023, test 2024
       fold 2025–26: train 2020–23, val 2024, test 2025–26   (= the usual split)
   Hyperparameters (C, decision threshold) are picked on each fold's val year.
   Pooling the three test years gives ~400 test samples / 58 misses instead of 23.

2. Paired bootstrap — is model A really better than history alone?  Resample the
   *same* rows for both models and get a confidence interval for the AUC difference.
   Much tighter than comparing two separate intervals, because both models share
   the "easy" and "hard" rows of each resample.

3. Decisions, not just ranking — AUC ignores the threshold.  We report F1 at 0.5 vs
   at a threshold tuned on val, and *precision in the top 10%*: if we flag the 10%
   riskiest announcements each year, what share are real misses?  (Base rate ~14%.)

4. Company-held-out — 5-fold cross-validation where folds are split *by ticker*
   (GroupKFold) vs at random.  If a model only recognises *which company* wrote the
   filing, its AUC collapses when the test companies were never seen in training.

Text blocks: ``embed`` (MiniLM, whole MD&A) and its within-company versions ``resid``
(minus the company's earlier-filing mean) and ``delta`` (minus its previous filing),
which remove company identity.  Every model is compared against ``history``.
   (Ignores time order; it is a diagnostic, not a forecast.)

Usage:
    python src/models/evaluate.py --data-root .                 # 10-Q MD&A, ~1 min
    python src/models/evaluate.py --data-root . --text press    # 8-K earnings press releases
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import HISTORY_COLS, build_table  # noqa: E402
from fusion import train_one  # noqa: E402
from text_features import GUIDANCE_COLS, within_company_embeddings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate")
logging.getLogger("fusion").setLevel(logging.WARNING)  # silence per-seed lines

FOLDS = [  # (name, val year start, test start, test end)
    ("2023", "2022-01-01", "2023-01-01", "2024-01-01"),
    ("2024", "2023-01-01", "2024-01-01", "2025-01-01"),
    ("2025-26", "2024-01-01", "2025-01-01", "2100-01-01"),
]
LR_MODELS = [
    "history", "embed", "history+embed_pca16",
    # within-company text (company "accent" removed; see text_features.within_company_embeddings)
    "resid", "history+resid_pca16", "delta", "history+delta_pca16",
    # rule-based outlook features: guidance raised / cut / maintained / withdrawn, hedging
    "guidance", "history+guidance", "history+guidance+embed_pca16",
]
CS = (0.001, 0.01, 0.1, 1.0, 10.0)
# Winner of W&B sweep 0zsaf6qz (picked on the 2024 val year — mildly optimistic for the 2023/2024 folds)
FUSION_CFG = dict(blocks=["history", "embed"], fusion="late", epochs=200, patience=20,
                  lr=2.274e-3, weight_decay=0.2011, history_dim=16, embed_dim=4, hidden=16,
                  dropout=0.532, input_dropout=0.4957, batch_size=64)


# ---------------------------------------------------------------- features + models
def make_blocks(model: str, fit: pd.DataFrame, frames: List[pd.DataFrame],
                cols: Dict[str, List[str]]) -> List[np.ndarray]:
    """Feature matrices for ``frames``; every scaler/PCA is fit on ``fit`` only.

    ``model`` is "+"-joined blocks, each a key of ``cols`` (history, embed, resid, delta, guidance),
    optionally with "_pcaN" to compress it to N principal components.
    """
    out = [[] for _ in frames]
    for block in model.split("+"):
        base, _, n_pca = block.partition("_pca")
        c = cols[base]
        sc = StandardScaler().fit(fit[c])
        mats = [sc.transform(f[c]) for f in [fit] + frames]
        if n_pca:
            pca = PCA(int(n_pca), random_state=0).fit(mats[0])
            mats = [pca.transform(m) for m in mats]
            sc2 = StandardScaler().fit(mats[0])
            mats = [sc2.transform(m) for m in mats]
        for o, m in zip(out, mats[1:]):
            o.append(m)
    return [np.hstack(o) for o in out]


def lr_fold(model: str, tr, va, te, cols) -> Tuple[np.ndarray, np.ndarray, float]:
    """Pick C on val AUC; return (val prob, test prob, chosen C)."""
    X_tr, X_va, X_te = make_blocks(model, tr, [tr, va, te], cols)
    best = None
    for C in CS:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(X_tr, tr.label)
        p_va = clf.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(va.label, p_va)
        if best is None or auc > best[0]:
            best = (auc, p_va, clf.predict_proba(X_te)[:, 1], C)
    return best[1], best[2], best[3]


def fusion_fold(tr, va, te, cols, device, seeds=(42, 43, 44)) -> Tuple[np.ndarray, np.ndarray, None]:
    """Swept late-fusion net, early-stopped on val; probabilities averaged over seeds."""
    X = {"train": {}, "val": {}, "test": {}}
    for b in FUSION_CFG["blocks"]:
        c = cols[b]
        sc = StandardScaler().fit(tr[c])
        for part, f in (("train", tr), ("val", va), ("test", te)):
            X[part][b] = torch.tensor(sc.transform(f[c]), dtype=torch.float32, device=device)
    y = {part: torch.tensor(f.label.to_numpy(), dtype=torch.float32, device=device)
         for part, f in (("train", tr), ("val", va), ("test", te))}
    probs = [train_one(SimpleNamespace(**FUSION_CFG), X, y, s, device)[1] for s in seeds]
    return np.mean([p["val"] for p in probs], 0), np.mean([p["test"] for p in probs], 0), None


# ---------------------------------------------------------------- metrics
def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold that maximises F1 on (val) data."""
    cands = np.unique(p)
    f1s = [f1_score(y, p >= t, zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def precision_top_k(y: np.ndarray, p: np.ndarray, frac: float = 0.10) -> float:
    k = max(1, int(round(frac * len(y))))
    return float(y[np.argsort(-p)[:k]].mean())


def bootstrap(y, p_by_model: Dict[str, np.ndarray], ref: str, n_boot=2000, seed=0) -> Dict:
    """AUC CI per model and paired CI of AUC(model) − AUC(ref) on the same resamples."""
    rng = np.random.default_rng(seed)
    aucs = {m: [] for m in p_by_model}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        for m, p in p_by_model.items():
            aucs[m].append(roc_auc_score(y[i], p[i]))
    out = {}
    for m in p_by_model:
        a = np.array(aucs[m])
        d = a - np.array(aucs[ref])
        out[m] = {"auc_ci95": np.percentile(a, [2.5, 97.5]).round(3).tolist(),
                  f"delta_vs_{ref}": float(roc_auc_score(y, p_by_model[m]) - roc_auc_score(y, p_by_model[ref])),
                  "delta_ci95": np.percentile(d, [2.5, 97.5]).round(3).tolist(),
                  "p_delta_gt_0": float((d > 0).mean())}
    return out


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="Rolling-origin, paired-bootstrap and company-held-out evaluation")
    p.add_argument("--data-root", default=".")
    p.add_argument("--text", default="mda", choices=["mda", "press"],
                   help="mda = 10-Q MD&A; press = 8-K earnings press releases (src/data/press_releases.py)")
    p.add_argument("--output", default=None, help="default checkpoints/evaluation.json (mda) / evaluation_press.json")
    p.add_argument("--no-fusion", action="store_true", help="Skip the neural model (LR only)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output = args.output or ("checkpoints/evaluation.json" if args.text == "mda"
                                  else f"checkpoints/evaluation_{args.text}.json")
    df, emb_cols = build_table(args.data_root, text=args.text)
    resid, delta = within_company_embeddings(df, emb_cols)
    df = pd.concat([df, resid, delta], axis=1)
    cols = {"history": HISTORY_COLS, "embed": emb_cols,
            "resid": list(resid.columns), "delta": list(delta.columns), "guidance": GUIDANCE_COLS}
    models = LR_MODELS + ([] if args.no_fusion else ["fusion_swept"])
    report: Dict = {"folds": {}, "pooled": {}, "company_holdout": {}}

    # ---- 1 + 3: rolling origin -----------------------------------------------------
    pooled = {m: {"y": [], "p": [], "pred_tuned": []} for m in models}
    top10 = {m: [] for m in models}  # (n flagged, n misses) per fold
    for name, val_start, test_start, test_end in FOLDS:
        tr = df[df.date < val_start]
        va = df[(df.date >= val_start) & (df.date < test_start)]
        te = df[(df.date >= test_start) & (df.date < test_end)]
        fold = {"n": [len(tr), len(va), len(te)], "miss": [int(tr.label.sum()), int(va.label.sum()), int(te.label.sum())]}
        for m in models:
            p_va, p_te, C = (fusion_fold(tr, va, te, cols, device) if m == "fusion_swept"
                             else lr_fold(m, tr, va, te, cols))
            t = best_f1_threshold(va.label.to_numpy(), p_va)
            y_te = te.label.to_numpy()
            k = max(1, int(round(0.10 * len(y_te))))
            top10[m].append((k, int(y_te[np.argsort(-p_te)[:k]].sum())))
            fold[m] = {"C": C, "auc": float(roc_auc_score(y_te, p_te)),
                       "f1@0.5": float(f1_score(y_te, p_te >= 0.5, zero_division=0)),
                       "threshold": t, "f1@tuned": float(f1_score(y_te, p_te >= t, zero_division=0)),
                       "prec_top10": precision_top_k(y_te, p_te)}
            pooled[m]["y"].append(y_te)
            pooled[m]["p"].append(p_te)
            pooled[m]["pred_tuned"].append(p_te >= t)
        report["folds"][name] = fold
        logger.info("fold %-7s train %d/%d val %d/%d test %d/%d (n/miss) — AUC: %s", name,
                    fold["n"][0], fold["miss"][0], fold["n"][1], fold["miss"][1], fold["n"][2], fold["miss"][2],
                    "  ".join(f"{m}={fold[m]['auc']:.3f}" for m in models))

    # ---- 2: pooled metrics + paired bootstrap ---------------------------------------
    y_all = np.concatenate(pooled[models[0]]["y"])
    p_all = {m: np.concatenate(pooled[m]["p"]) for m in models}
    boot = bootstrap(y_all, p_all, ref="history")
    for m in models:
        pred_t = np.concatenate(pooled[m]["pred_tuned"])
        n_flag, n_hit = map(sum, zip(*top10[m]))
        report["pooled"][m] = {
            "auc": float(roc_auc_score(y_all, p_all[m])),
            "mean_fold_auc": float(np.mean([report["folds"][f][m]["auc"] for f, *_ in FOLDS])),
            "avg_precision": float(average_precision_score(y_all, p_all[m])),
            "f1@0.5": float(f1_score(y_all, p_all[m] >= 0.5)),
            "f1@tuned": float(f1_score(y_all, pred_t)),
            "precision@tuned": float(y_all[pred_t].mean()) if pred_t.any() else 0.0,
            "recall@tuned": float(pred_t[y_all == 1].mean()),
            "prec_top10": n_hit / n_flag, "top10_hits": f"{n_hit}/{n_flag}",
            **boot[m],
        }
    base = float(y_all.mean())
    report["pooled"]["_base_rate"] = base

    # ---- 4: company-held-out vs random CV (LR, C as chosen on 2024 val in the 2025-26 fold) ----
    fixed_C = {m: report["folds"]["2025-26"][m]["C"] for m in LR_MODELS}
    y = df.label.to_numpy()
    splitters = {"random": StratifiedKFold(5, shuffle=True, random_state=0).split(df, y),
                 "by_company": GroupKFold(5).split(df, y, groups=df.ticker)}
    unseen_prob = {}
    for split_name, splits in splitters.items():
        splits = list(splits)
        for m in LR_MODELS:
            prob = np.zeros(len(df))
            for tr_i, te_i in splits:
                tr, te = df.iloc[tr_i], df.iloc[te_i]
                X_tr, X_te = make_blocks(m, tr, [tr, te], cols)
                clf = LogisticRegression(C=fixed_C[m], class_weight="balanced", max_iter=5000).fit(X_tr, tr.label)
                prob[te_i] = clf.predict_proba(X_te)[:, 1]
            report["company_holdout"].setdefault(m, {})[split_name] = float(roc_auc_score(y, prob))
            if split_name == "by_company":
                unseen_prob[m] = prob
    # Paired bootstrap on unseen companies too: every row is a test row here (2.5× the rolling-origin test)
    for m, r in bootstrap(y, unseen_prob, ref="history").items():
        report["company_holdout"][m]["by_company_delta_ci95"] = r["delta_ci95"]
        report["company_holdout"][m]["by_company_p_delta_gt_0"] = r["p_delta_gt_0"]

    # ---- print ----------------------------------------------------------------------
    logger.info("=" * 78)
    logger.info("POOLED rolling-origin test, %d samples / %d misses (base rate %.3f)",
                len(y_all), int(y_all.sum()), base)
    logger.info("%-28s %6s %8s %8s %6s %7s %7s %10s  %s", "model", "AUC", "AUC 95%", "", "AP",
                "F1@.5", "F1@val", "top10%", "ΔAUC vs history [95%]  P(Δ>0)")
    for m in models:
        r = report["pooled"][m]
        lo, hi = r["auc_ci95"]
        dlo, dhi = r["delta_ci95"]
        logger.info("%-28s %6.3f %8.2f–%.2f %6.3f %7.3f %7.3f %5.2f (%s)  %+.3f [%+.3f, %+.3f]  %.2f",
                    m, r["auc"], lo, hi, r["avg_precision"], r["f1@0.5"], r["f1@tuned"],
                    r["prec_top10"], r["top10_hits"], r["delta_vs_history"], dlo, dhi, r["p_delta_gt_0"])
    logger.info("Company-held-out (5-fold CV, all years; AUC random → by company; by-company ΔAUC vs history):")
    for m, r in report["company_holdout"].items():
        dlo, dhi = r["by_company_delta_ci95"]
        logger.info("  %-28s %.3f → %.3f  %+.3f [%+.3f, %+.3f]  P(Δ>0)=%.2f", m, r["random"], r["by_company"],
                    r["by_company"] - report["company_holdout"]["history"]["by_company"], dlo, dhi,
                    r["by_company_p_delta_gt_0"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()
