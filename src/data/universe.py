"""
Company universe: the original 49 tickers plus an expansion sample from the S&P 500
==================================================================================
Every result so far is limited by the number of misses (~250 on press releases), so
the expansion adds companies, not features.  New companies are drawn from the
*current* S&P 500 (datasets/s-and-p-500-companies on GitHub), stratified by GICS
sector so that the sample is not just more mega-cap tech:

  * one ticker per CIK (GOOG/GOOGL, FOX/FOXA, NWS/NWSA → the first listed)
  * ``BRK.B`` style tickers → ``BRK-B`` (the form SEC and Yahoo use)
  * ``n`` new companies, split across sectors in proportion to the remaining pool,
    sampled with a fixed seed

Real Estate is excluded (``exclude`` column): REITs report FFO, and Yahoo's EPS
actual and estimate are often on different bases, so equity REITs "missed" 45–62%
of quarters (EQIX by 25–49% quarter after quarter) — label noise, not signal.

The result is written to ``data/universe.csv`` (committed, so every machine scrapes
the same set even after the index membership changes).

Caveat: today's index members are survivors, so the sample over-represents
companies that did well enough to stay in the index.  The original 49 have the
same bias.

Usage:
    python src/data/universe.py                       # → data/universe.csv (49 original + 150 new)
    python src/data/universe.py --list expansion      # print the new tickers (for --tickers $(...))
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scraper import HEADERS, get_default_tickers  # noqa: E402

SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
EXCLUDED_SECTORS = {"Real Estate": "REIT EPS labels unreliable (FFO vs GAAP EPS basis)"}
UNIVERSE_PATH = Path(__file__).resolve().parents[2] / "data" / "universe.csv"


def build_universe(n_new: int = 150, seed: int = 0) -> pd.DataFrame:
    sp = pd.read_csv(io.StringIO(requests.get(SP500_URL, headers=HEADERS, timeout=30).text))
    sp["ticker"] = sp["Symbol"].str.replace(".", "-", regex=False)
    sp = sp.drop_duplicates("CIK").rename(columns={"Security": "name", "GICS Sector": "sector", "CIK": "cik"})

    original = get_default_tickers()
    orig = pd.DataFrame({"ticker": original}).merge(sp[["ticker", "name", "sector", "cik"]], on="ticker", how="left")
    orig["batch"] = "original"

    pool = sp[~sp["ticker"].isin(original) & ~sp["cik"].isin(orig["cik"].dropna())]
    # Largest-remainder allocation of n_new across sectors, proportional to pool size
    share = pool["sector"].value_counts() / len(pool) * n_new
    alloc = np.floor(share).astype(int)
    alloc[(share - alloc).sort_values(ascending=False).index[: n_new - alloc.sum()]] += 1
    rng = np.random.default_rng(seed)
    picks = [g.iloc[rng.choice(len(g), alloc[s], replace=False)] for s, g in pool.groupby("sector")]
    new = pd.concat(picks)[["ticker", "name", "sector", "cik"]].sort_values("ticker")
    new["batch"] = "expansion"
    u = pd.concat([orig, new], ignore_index=True)
    u["exclude"] = u["sector"].map(EXCLUDED_SECTORS).fillna("")
    return u


def load_universe(batch: str | None = None) -> list[str]:
    u = pd.read_csv(UNIVERSE_PATH, keep_default_na=False)
    u = u[u["exclude"] == ""]
    return u["ticker"].tolist() if batch is None else u.loc[u["batch"] == batch, "ticker"].tolist()


def load_sectors() -> dict:
    """ticker → GICS sector, for every company in the universe file."""
    u = pd.read_csv(UNIVERSE_PATH, keep_default_na=False)
    return dict(zip(u["ticker"], u["sector"]))


def main() -> None:
    p = argparse.ArgumentParser(description="Build / list the company universe")
    p.add_argument("--n-new", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--list", choices=["all", "original", "expansion"], default=None,
                   help="print tickers from the existing data/universe.csv instead of rebuilding it")
    args = p.parse_args()

    if args.list:
        print(" ".join(load_universe(None if args.list == "all" else args.list)))
        return
    u = build_universe(args.n_new, args.seed)
    UNIVERSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    u.to_csv(UNIVERSE_PATH, index=False)
    print(f"Saved {len(u)} companies to {UNIVERSE_PATH}")
    print(u.groupby(["sector", "batch"]).size().unstack(fill_value=0).to_string())
    print(f"Excluded: {(u['exclude'] != '').sum()} ({', '.join(EXCLUDED_SECTORS)})")


if __name__ == "__main__":
    main()
