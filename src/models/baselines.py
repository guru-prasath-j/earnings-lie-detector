#!/usr/bin/env python3
"""
Reference models for the beat/miss task, evaluated on the same samples and
temporal split as ``train.py``.

Feature blocks (see ``text_features.py``), each fed to logistic regression alone
and in combination:

  history — the company's own past earnings surprises (prior quarter's
            miss/surprise, last-4-quarter averages).  No text.
  change  — how much the filing differs from the company's previous and
            year-ago filings, plus negative/uncertainty word rates.
  embed   — mean MiniLM embedding over the *whole* MD&A (384-dim).
  tfidf   — TF-IDF over the full MD&A text.

Any text block must improve on ``history`` alone to be worth keeping.

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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import EarningsCallDataset  # noqa: E402
from text_features import CHANGE_COLS, GUIDANCE_COLS, change_features, document_embeddings, guidance_features  # noqa: E402
from llm_guidance import LLM_COLS, llm_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("baselines")

HISTORY_COLS = ["prev_miss", "prev_surprise", "miss_rate_4q", "mean_surprise_4q", "std_surprise_4q"]

# Text sources: folder of {TICKER}.json, embedding cache, min days between text and the label it predicts
TEXT_SOURCES = {
    "mda": {"dir": "data/raw/transcripts", "cache": "embeddings_minilm.npz", "min_gap": 0, "llm": None},  # 10-Q MD&A
    "press": {"dir": "data/raw/press_releases", "cache": "embeddings_minilm_press.npz", "min_gap": 30,      # 8-K EX-99.1
              "llm": "data/processed/llm_guidance.jsonl"},  # LLM-scored guidance (llm_guidance.py), if run
}


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


def auc_ci(y: np.ndarray, prob: np.ndarray, n_boot: int = 2000, seed: int = 0) -> List[float]:
    """95% bootstrap interval for AUC — how much the score could move by chance."""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], prob[i]))
    return [float(v) for v in np.percentile(aucs, [2.5, 97.5])]


def fit_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, Cs=(0.001, 0.01, 0.1, 1.0, 10.0)) -> Dict:
    """Pick C by validation AUC, report val + test metrics for that C.

    Also returns the val/test probabilities (``_prob``) so models can be averaged.
    """
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        clf.fit(X_tr, y_tr)
        p_va, p_te = clf.predict_proba(X_va)[:, 1], clf.predict_proba(X_te)[:, 1]
        val = metrics(y_va, p_va)
        if best is None or val["auc"] > best["val"]["auc"]:
            best = {"C": C, "val": val, "test": metrics(y_te, p_te), "_prob": {"val": p_va, "test": p_te}}
    best["test"]["auc_ci95"] = auc_ci(y_te, best["_prob"]["test"])
    return best


def average_models(results: Dict, names: Tuple[str, ...], y_va: np.ndarray, y_te: np.ndarray) -> Dict:
    """Ensemble: average of the probabilities from separately trained models.

    Each block gets its own regularisation, so a 384-dim block can't swamp a 5-dim one.
    All members are class-balanced logistic regressions, so their probabilities are on
    the same scale and the usual 0.5 threshold still applies.
    """
    prob = {part: np.mean([results[n]["_prob"][part] for n in names], axis=0) for part in ("val", "test")}
    out = {"C": None, "val": metrics(y_va, prob["val"]), "test": metrics(y_te, prob["test"]), "_prob": prob}
    out["test"]["auc_ci95"] = auc_ci(y_te, prob["test"])
    return out


def build_table(data_root: str, text: str = "mda") -> Tuple[pd.DataFrame, List[str]]:
    """One row per sample: date, label, text, history, change, guidance and embedding columns.

    ``text`` picks the document source (see ``TEXT_SOURCES``).
    Returns the table and the names of the embedding columns.
    """
    root = Path(data_root)
    src = TEXT_SOURCES[text]
    with contextlib.redirect_stdout(io.StringIO()):
        ds = EarningsCallDataset(data_root=data_root, transcripts_dir=root / src["dir"],
                                 min_transcript_gap_days=src["min_gap"])
    hist = history_features(root / "data" / "processed" / "labels.csv")
    change = change_features(ds.transcripts)  # all filings, so "previous filing" is complete
    guidance = guidance_features(ds.transcripts)
    used = {s["transcript_key"] for s in ds.samples}
    emb = document_embeddings({k: ds.transcripts[k] for k in used},
                              root / "data" / "processed" / src["cache"])
    emb_cols = [f"emb_{i}" for i in range(len(next(iter(emb.values()))))]
    llm = None
    if src["llm"] and (root / src["llm"]).exists():
        llm = llm_features(root / src["llm"])
        missing = len(used - set(llm.index))
        if missing > 0.01 * len(used):  # e.g. new companies not scored yet: zeros would read as "no guidance"
            logger.warning("%d of %d documents have no LLM score → LLM columns left out", missing, len(used))
            llm = None
        elif missing:
            logger.warning("%d of %d documents have no LLM score (treated as no guidance)", missing, len(used))

    rows = []
    for s in ds.samples:
        tk = s["transcript_key"]
        rows.append({
            "ticker": s["ticker"],
            "date": s["date"][:10],
            "label": s["label"],
            "text": ds.transcripts[tk],
            **hist.loc[(s["ticker"], s["date"])].to_dict(),
            **change.loc[tk].to_dict(),
            **guidance.loc[tk].to_dict(),
            **({c: float(llm.at[tk, c]) if tk in llm.index else 0.0 for c in LLM_COLS} if llm is not None else {}),
            **dict(zip(emb_cols, emb[tk])),
        })
    return pd.DataFrame(rows), emb_cols


def temporal_split(df: pd.DataFrame, val_start: str, test_start: str) -> Dict[str, pd.DataFrame]:
    """Same split as ``get_dataloaders(split="temporal")``."""
    parts = {
        "train": df[df.date < val_start],
        "val": df[(df.date >= val_start) & (df.date < test_start)],
        "test": df[df.date >= test_start],
    }
    logger.info("Split — " + ", ".join(f"{k}: {len(v)} ({int(v.label.sum())} miss)" for k, v in parts.items()))
    return parts


def main() -> None:
    p = argparse.ArgumentParser(description="History-only and TF-IDF reference models")
    p.add_argument("--data-root", default=".")
    p.add_argument("--val-start", default="2024-01-01")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--output", default="checkpoints/baselines.json")
    args = p.parse_args()

    df, emb_cols = build_table(args.data_root)
    parts = temporal_split(df, args.val_start, args.test_start)
    y = {k: v["label"].to_numpy() for k, v in parts.items()}

    # Dense blocks, each standardised on train only
    def dense(cols):
        scaler = StandardScaler().fit(parts["train"][cols])
        return {k: csr_matrix(scaler.transform(v[cols])) for k, v in parts.items()}

    blocks = {"history": dense(HISTORY_COLS), "change": dense(CHANGE_COLS), "embed": dense(emb_cols)}

    # Embedding compressed to a few components (PCA fit on train) so it can sit next to history
    for n in (8, 16, 32):
        pca = PCA(n, random_state=0).fit(blocks["embed"]["train"].toarray())
        reduced = {k: pca.transform(v.toarray()) for k, v in blocks["embed"].items()}
        scaler = StandardScaler().fit(reduced["train"])
        blocks[f"embed_pca{n}"] = {k: csr_matrix(scaler.transform(v)) for k, v in reduced.items()}

    # TF-IDF on full text, fit on train only
    tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3, max_df=0.9,
                            sublinear_tf=True, stop_words="english")
    blocks["tfidf"] = {"train": tfidf.fit_transform(parts["train"]["text"])}
    blocks["tfidf"]["val"] = tfidf.transform(parts["val"]["text"])
    blocks["tfidf"]["test"] = tfidf.transform(parts["test"]["text"])

    combos = [
        ("history",), ("change",), ("embed",), ("tfidf",),
        ("history", "change"), ("history", "embed"), ("history", "tfidf"),
        ("history", "change", "embed"),
        ("history", "embed_pca8"), ("history", "embed_pca16"), ("history", "embed_pca32"),
    ]
    results = {}
    for combo in combos:
        X = {k: hstack([blocks[b][k] for b in combo]).tocsr() for k in ("train", "val", "test")}
        results["+".join(combo)] = fit_eval(X["train"], y["train"], X["val"], y["val"], X["test"], y["test"])
    results["avg(history,embed)"] = average_models(results, ("history", "embed"), y["val"], y["test"])

    logger.info("=" * 60)
    for name, r in results.items():
        lo, hi = r["test"]["auc_ci95"]
        logger.info("%-22s C=%-5s VAL f1=%.3f auc=%.3f | TEST f1=%.3f auc=%.3f [%.2f–%.2f] P=%.3f R=%.3f",
                    name, r["C"], r["val"]["f1"], r["val"]["auc"],
                    r["test"]["f1"], r["test"]["auc"], lo, hi, r["test"]["precision"], r["test"]["recall"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args": vars(args),
                   "results": {n: {k: v for k, v in r.items() if k != "_prob"} for n, r in results.items()}},
                  f, indent=2)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()
