"""
Text features for the beat/miss task, computed per filing (``(ticker, filing_date)``).

1. **Change features** — how much a filing differs from the same company's
   earlier filings ("Lazy Prices", Cohen, Malloy & Nguyen 2020): firms tend to
   rewrite their reports when something changes, and those changes have
   preceded bad news.  Compared against both the previous filing and the
   filing from ~1 year earlier (same fiscal quarter, so seasonal structure
   matches).  No model or fitting involved.

2. **Document embeddings** — the *whole* MD&A split into 256-token chunks,
   each chunk embedded with a frozen sentence-embedding model
   (``all-MiniLM-L6-v2``, 384-dim), then averaged.  Cached to disk.

3. **Within-company embeddings** — the embedding minus the same company's
   earlier filings (their running mean, or just the previous one), which
   removes the company's recognisable writing style ("identity").

4. **Guidance features** — rule-based reading of what a press release says
   about its outlook: was guidance raised, cut, maintained, narrowed,
   (re)introduced or withdrawn, and how hedged are the outlook sentences.

Only text from the filing itself and from *earlier* filings is used, so there
is no look-ahead.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Key = Tuple[str, str]  # (ticker, filing date YYYY-MM-DD)

# Small hand-picked word lists (a stand-in for the Loughran-McDonald finance dictionary).
NEGATIVE_WORDS = {
    "decline", "declined", "declines", "decrease", "decreased", "decreases", "loss", "losses",
    "impairment", "impairments", "weak", "weaker", "weakness", "adverse", "adversely", "challenging",
    "challenges", "headwinds", "unfavorable", "lower", "difficult", "deterioration", "litigation",
    "restructuring", "shortfall", "downturn", "pressure", "pressures", "slowdown", "delays", "delayed",
}
UNCERTAINTY_WORDS = {
    "uncertain", "uncertainty", "uncertainties", "may", "might", "could", "possible", "possibly",
    "approximately", "risk", "risks", "volatility", "volatile", "depend", "depends", "unpredictable",
    "fluctuations", "contingent", "believe", "anticipate", "expect", "estimate", "estimates",
}

_WORD_RE = re.compile(r"[a-z]+")

CHANGE_COLS = [
    "has_prev", "sim_prev", "jaccard_prev", "log_len_ratio_prev",
    "has_yoy", "sim_yoy", "jaccard_yoy", "log_len_ratio_yoy",
    "neg_rate", "unc_rate", "d_neg_rate_prev", "d_unc_rate_prev",
]


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = np.sqrt(sum(v * v for v in a.values()))
    nb = np.sqrt(sum(v * v for v in b.values()))
    return float(dot / (na * nb)) if na and nb else 0.0


def change_features(transcripts: Dict[Key, str]) -> pd.DataFrame:
    """One row per filing with similarity / length / tone change vs earlier filings."""
    docs: Dict[Key, dict] = {}
    for key, text in transcripts.items():
        toks = _tokens(text)
        n = max(1, len(toks))
        counts = Counter(toks)
        docs[key] = {
            "counts": counts,
            "vocab": set(counts),
            "n": n,
            "neg_rate": 1000 * sum(counts[w] for w in NEGATIVE_WORDS) / n,
            "unc_rate": 1000 * sum(counts[w] for w in UNCERTAINTY_WORDS) / n,
        }

    by_ticker: Dict[str, List[Key]] = {}
    for key in sorted(docs):
        by_ticker.setdefault(key[0], []).append(key)

    def compare(cur: dict, old: Optional[dict], prefix: str) -> dict:
        if old is None:
            return {f"has_{prefix}": 0.0}
        return {
            f"has_{prefix}": 1.0,
            f"sim_{prefix}": _cosine(cur["counts"], old["counts"]),
            f"jaccard_{prefix}": len(cur["vocab"] & old["vocab"]) / max(1, len(cur["vocab"] | old["vocab"])),
            f"log_len_ratio_{prefix}": float(np.log(cur["n"] / old["n"])),
        }

    rows = []
    for ticker, keys in by_ticker.items():
        dates = [pd.Timestamp(k[1]) for k in keys]
        for i, key in enumerate(keys):
            cur = docs[key]
            prev = docs[keys[i - 1]] if i > 0 else None
            # Same quarter last year: an earlier filing 300–430 days back
            yoy = None
            for j in range(i - 1, -1, -1):
                gap = (dates[i] - dates[j]).days
                if 300 <= gap <= 430:
                    yoy = docs[keys[j]]
                    break
                if gap > 430:
                    break
            row = {"ticker": ticker, "date": key[1], "neg_rate": cur["neg_rate"], "unc_rate": cur["unc_rate"]}
            row.update(compare(cur, prev, "prev"))
            row.update(compare(cur, yoy, "yoy"))
            if prev is not None:
                row["d_neg_rate_prev"] = cur["neg_rate"] - prev["neg_rate"]
                row["d_unc_rate_prev"] = cur["unc_rate"] - prev["unc_rate"]
            rows.append(row)

    df = pd.DataFrame(rows).set_index(["ticker", "date"])
    df = df.reindex(columns=CHANGE_COLS)
    # Missing comparisons → neutral values; the has_* flags tell the model they're missing
    return df.fillna({c: df[c].median() for c in CHANGE_COLS if c not in ("has_prev", "has_yoy")}).fillna(0.0)


def document_embeddings(
    transcripts: Dict[Key, str],
    cache_path: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_tokens: int = 256,
    batch_size: int = 64,
) -> Dict[Key, np.ndarray]:
    """Mean of chunk embeddings over the full text of each filing (cached)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cached = {(str(t), str(d)): v for t, d, v in zip(data["tickers"], data["dates"], data["embeddings"])}
        if (data["model"].item() == model_name and int(data["chunk_tokens"]) == chunk_tokens
                and set(transcripts) <= set(cached)):
            logger.info("Loaded %d cached embeddings from %s", len(cached), cache_path)
            return cached

    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    out: Dict[Key, np.ndarray] = {}
    with torch.no_grad():
        for key, text in tqdm(sorted(transcripts.items()), desc="Embedding filings"):
            enc = tok(text, max_length=chunk_tokens, truncation=True, padding=True,
                      return_overflowing_tokens=True, return_tensors="pt")
            vecs = []
            for s in range(0, enc["input_ids"].shape[0], batch_size):
                ids = enc["input_ids"][s:s + batch_size].to(device)
                mask = enc["attention_mask"][s:s + batch_size].to(device)
                hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
                # Mean pooling over real tokens, then L2-normalise (how MiniLM is meant to be used)
                m = mask.unsqueeze(-1).float()
                emb = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
                vecs.append(torch.nn.functional.normalize(emb, dim=-1).cpu())
            out[key] = torch.cat(vecs).mean(0).numpy().astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(out)
    np.savez(cache_path, tickers=np.array([k[0] for k in keys]), dates=np.array([k[1] for k in keys]),
             embeddings=np.stack([out[k] for k in keys]), model=np.array(model_name),
             chunk_tokens=np.array(chunk_tokens))
    logger.info("Saved %d embeddings to %s", len(out), cache_path)
    return out


def within_company_embeddings(df: pd.DataFrame, emb_cols: List[str], group_col: str = "ticker",
                              order_col: str = "date") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove each company's "accent" from its filing embeddings.

    Returns two frames aligned with ``df`` (columns ``resid_i`` and ``delta_i``):
      resid — embedding minus the mean of the same company's *earlier* rows
      delta — embedding minus the company's previous row
    Only earlier rows are used, so there is no look-ahead.  A company's first row has
    no history and gets 0 ("typical").
    """
    order = df.sort_values([group_col, order_col]).index
    E = df.loc[order, emb_cols].to_numpy(dtype=np.float64)
    groups = df.loc[order, group_col].to_numpy()
    g = pd.DataFrame(E).groupby(groups)
    n_before = g.cumcount().to_numpy()[:, None]               # earlier rows of the same company
    sum_before = g.cumsum().to_numpy() - E                    # their sum (cumsum includes this row)
    with np.errstate(invalid="ignore", divide="ignore"):
        resid = np.where(n_before > 0, E - sum_before / n_before, 0.0)
    prev = g.shift(1).to_numpy()
    delta = np.nan_to_num(E - prev, nan=0.0)
    n = len(emb_cols)
    resid = pd.DataFrame(resid, index=order, columns=[f"resid_{i}" for i in range(n)]).reindex(df.index)
    delta = pd.DataFrame(delta, index=order, columns=[f"delta_{i}" for i in range(n)]).reindex(df.index)
    return resid, delta


# ---- guidance features ---------------------------------------------------------------
# The rules live in guidance_rules.py (self-contained, shipped with the Hub model); re-exported here.
from guidance_rules import (  # noqa: E402,F401
    _BOILERPLATE_RE, _DIRECTION_RES, _OUTLOOK_RE, _SENT_RE, GUIDANCE_COLS, HEDGE_WORDS, MAX_VERB_GAP, guidance_vector,
)


def guidance_features(transcripts: Dict[Key, str]) -> pd.DataFrame:
    """One row per document: counts of outlook sentences by direction, plus their hedging rate.

    ``g_net`` = sign(raise − cut): +1 more raises, −1 more cuts, 0 otherwise.
    """
    rows = [{"ticker": t, "date": d, **guidance_vector(text)} for (t, d), text in transcripts.items()]
    return pd.DataFrame(rows).set_index(["ticker", "date"])[GUIDANCE_COLS]
