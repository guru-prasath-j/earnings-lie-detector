"""
Point-in-time market data from Yahoo Finance: analyst actions and daily prices
==============================================================================
Yahoo's estimate tables (EPS trend, revisions, dispersion, number of analysts) are
**today's snapshot only**, so using them for past announcements would leak the future.
Two things Yahoo does keep with timestamps:

  * ``upgrades_downgrades`` — every analyst rating action since ~2012 (firm, from/to
    grade, up / down / main / init, price target raised / lowered and its old and new
    value).  A cut price target usually comes with a cut estimate, so this is the
    closest free proxy for estimate revisions.
  * daily prices — the stock's run-up before an announcement reflects what investors
    already know.  SPY is fetched too, as the market benchmark.

Output: ``data/raw/market/{TICKER}_analyst.csv`` and ``{TICKER}_prices.csv`` (Close only).

Usage:
    python src/data/market_data.py                      # every company in data/universe.csv + SPY, ~5 min
    python src/data/market_data.py --tickers AAPL JPM
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import load_universe  # noqa: E402

OUT_DIR = Path("data/raw/market")
START = "2015-01-01"  # labels start 2017; 90-day windows need a little earlier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def fetch(ticker: str, out_dir: Path) -> None:
    t = yf.Ticker(ticker)
    px = t.history(start=START, auto_adjust=True)
    if px.empty:
        logger.warning("%s: no prices", ticker)
    else:
        px.index = px.index.tz_localize(None).normalize()
        px[["Close"]].to_csv(out_dir / f"{ticker}_prices.csv", index_label="date")
    if ticker == "SPY":
        return
    try:
        ud = t.upgrades_downgrades
    except Exception as exc:  # yfinance raises on tickers without coverage
        logger.warning("%s: no analyst actions (%s)", ticker, type(exc).__name__)
        ud = None
    if ud is None or ud.empty:
        ud = pd.DataFrame(columns=["Firm", "ToGrade", "FromGrade", "Action", "priceTargetAction",
                                   "currentPriceTarget", "priorPriceTarget"])
        ud.index.name = "GradeDate"
    ud.to_csv(out_dir / f"{ticker}_analyst.csv")
    logger.info("%s: %d price days, %d analyst actions", ticker, len(px), len(ud))


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch analyst actions and daily prices (Yahoo Finance)")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--output-dir", default=str(OUT_DIR))
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for tk in (args.tickers or load_universe()) + ["SPY"]:
        try:
            fetch(tk, out)
        except Exception as exc:
            logger.warning("%s failed: %s", tk, exc)
        time.sleep(0.3)


if __name__ == "__main__":
    main()
