#!/usr/bin/env python3
"""
Reference models for the beat/miss task, evaluated on the same samples and
temporal split as ``train.py``.

  history  — logistic regression on the company's own past earnings surprises
             (prior quarter's miss/surprise, last-4-quarter averages).  No text.
  tfidf    — TF-IDF over the *full* MD&A text (no 512-token cut) + logistic regression.
  combined — both feature sets together.

If FinBERT can't beat ``history``, the text adds nothing yet; if ``tfidf``
beats FinBERT, truncation is discarding useful text.

Usage:
    python src/models/baselines.py --data-root .
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from dataset import EarningsCallDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("baselines")

HISTORY_COLS = ["prev_miss", "prev_surprise", "miss_rate_4q", "mean_surprise_4q", "std_surprise_4q"]


def history_features(labels_csv: Path) -> pd.DataFrame:
    """Per (ticker, date) features built only from *earlier* announcements."""
    df = pd.read_csv(labels_csv).sort_values(["ticker", "date"])
    g = df.groupby("ticker")
    # shift(1): each row only sees announcements strictly before it
    prev_label = g["label"].shift(1)
    prev_surp = g["eps_surprise_pct"].shift(1).clip(-100, 100)
    df["prev_miss"] = prev_label
    df["prev_surprise"] = prev_surp
    df["miss_rate_4q"] = prev_label.groupby(df["ticker"]).transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["mean_surprise_4q"] = prev_surp.groupby(df["ticker"]).transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["std_surprise_4q"] = prev_surp.groupby(df["ticker"]).transform(lambda s: s.rolling(4, min_periods=2).std())
    # Fill gaps (first quarters of a ticker) with neutral values
    df[HISTORY_COLS] = df[HISTORY_COLS].fillna({"prev_miss": 0, "prev_surprise": 0, "miss_rate_4q": 0,
                                               "mean_surprise_4q": 0, "std_surprise_4q": 0})
    df["ticker"] = df["ticker"].str.upper()
    df["date"] = df["date"].astype(str)
    return df.set_index(["ticker", "date"])[HISTORY_COLS]


def metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, prob)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def fit_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, Cs=(0.01, 0.1, 1.0, 10.0)) -> Dict:
    """Pick C by validation AUC, report val + test metrics for that C."""
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        clf.fit(X_tr, y_tr)
        val = metrics(y_va, clf.predict_proba(X_va)[:, 1])
        if best is None or val["auc"] > best["val"]["auc"]:
            best = {"C": C, "val": val, "test": metrics(y_te, clf.predict_proba(X_te)[:, 1])}
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="History-only and TF-IDF reference models")
    p.add_argument("--data-root", default=".")
    p.add_argument("--val-start", default="2024-01-01")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--output", default="checkpoints/baselines.json")
    args = p.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        ds = EarningsCallDataset(data_root=args.data_root)
    root = Path(args.data_root)
    hist = history_features(root / "data" / "processed" / "labels.csv")

    rows = []
    for s in ds.samples:
        rows.append({
            "date": s["date"][:10],
            "label": s["label"],
            "text": ds.transcripts[s["transcript_key"]],
            **hist.loc[(s["ticker"], s["date"])].to_dict(),
        })
    df = pd.DataFrame(rows)
    parts = {
        "train": df[df.date < args.val_start],
        "val": df[(df.date >= args.val_start) & (df.date < args.test_start)],
        "test": df[df.date >= args.test_start],
    }
    y = {k: v["label"].to_numpy() for k, v in parts.items()}
    logger.info("Split — " + ", ".join(f"{k}: {len(v)} ({int(v.label.sum())} miss)" for k, v in parts.items()))

    # History features
    scaler = StandardScaler().fit(parts["train"][HISTORY_COLS])
    H = {k: scaler.transform(v[HISTORY_COLS]) for k, v in parts.items()}

    # TF-IDF on full text, fit on train only
    tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3, max_df=0.9,
                            sublinear_tf=True, stop_words="english")
    T = {"train": tfidf.fit_transform(parts["train"]["text"])}
    T["val"] = tfidf.transform(parts["val"]["text"])
    T["test"] = tfidf.transform(parts["test"]["text"])

    results = {
        "history": fit_eval(H["train"], y["train"], H["val"], y["val"], H["test"], y["test"]),
        "tfidf": fit_eval(T["train"], y["train"], T["val"], y["val"], T["test"], y["test"]),
        "combined": fit_eval(
            *[m for k in ("train", "val", "test") for m in (hstack([T[k], csr_matrix(H[k])]).tocsr(), y[k])]
        ),
    }

    logger.info("=" * 60)
    for name, r in results.items():
        logger.info("%-9s C=%-5g VAL f1=%.3f auc=%.3f | TEST f1=%.3f auc=%.3f P=%.3f R=%.3f",
                    name, r["C"], r["val"]["f1"], r["val"]["auc"],
                    r["test"]["f1"], r["test"]["auc"], r["test"]["precision"], r["test"]["recall"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()
