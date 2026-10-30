"""
Point-in-time analyst and price features for each earnings announcement — numpy and pandas only,
so this file is also shipped next to the model on the Hugging Face Hub.

Everything is computed from data dated at least ``SAFETY_DAYS`` before the announcement day, so neither
the release itself, the analyst reactions to it, nor the announcement-day price move can leak in, even
when Yahoo's announcement date is a day or two late (the press-release step found such offsets).

Analyst block (``ANALYST_COLS``), actions in the ``WINDOW_DAYS`` calendar days before:
  * ``an_n``          log(1 + number of rating actions)
  * ``an_net_grade``  upgrades − downgrades
  * ``an_pt_net``     (price targets raised − lowered) / actions with a target change
  * ``an_pt_chg``     mean % change of the price target (new / old − 1), clipped to ±50%

Price block (``PRICE_COLS``), closes up to the last trading day before the window end:
  * ``px_rel_63`` / ``px_rel_21``  return over ~3 months / 1 month minus SPY's (clipped to ±100% / ±50%)
  * ``px_vol_63``                   std of daily returns over ~3 months

Training reads the files written by ``src/data/market_data.py`` (``data/raw/market``);
``yahoo_inputs(ticker)`` fetches the same inputs live (needs ``yfinance``) for a new prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ANALYST_COLS = ["an_n", "an_net_grade", "an_pt_net", "an_pt_chg"]
PRICE_COLS = ["px_rel_63", "px_rel_21", "px_vol_63"]
WINDOW_DAYS = 90
SAFETY_DAYS = 2  # windows end this many days before the announcement date
PRICE_START = "2015-01-01"


def prepare_analyst(an: pd.DataFrame) -> pd.DataFrame:
    """Yahoo ``upgrades_downgrades`` (``GradeDate`` as a column) → adds ``day`` and ``pt_chg``."""
    an = an.copy()
    if an.empty:
        return an
    an["GradeDate"] = pd.to_datetime(an["GradeDate"])
    if getattr(an["GradeDate"].dt, "tz", None) is not None:
        an["GradeDate"] = an["GradeDate"].dt.tz_localize(None)
    an["day"] = an["GradeDate"].dt.normalize()
    prior, cur = an["priorPriceTarget"].fillna(0), an["currentPriceTarget"].fillna(0)
    an["pt_chg"] = np.where((prior > 0) & (cur > 0), cur / prior.where(prior > 0, 1) - 1, np.nan)
    return an


def _ret(px: pd.Series, end: int, n: int) -> float:
    return float(px.iloc[end] / px.iloc[end - n] - 1) if end - n >= 0 else np.nan


def market_vector(an: pd.DataFrame, px: pd.Series, spy: pd.Series, date: str) -> Dict[str, float]:
    """``ANALYST_COLS`` + ``PRICE_COLS`` for one announcement on ``date`` (YYYY-MM-DD).

    ``an`` from ``prepare_analyst``; ``px`` / ``spy`` daily closes indexed by tz-naive dates.
    Price features are NaN without ~3 months of history (callers fill them).
    """
    day = pd.Timestamp(date) - pd.Timedelta(days=SAFETY_DAYS - 1)  # data strictly before this day
    r: Dict[str, float] = {}
    w = an[(an["day"] < day) & (an["day"] >= day - pd.Timedelta(days=WINDOW_DAYS))] if not an.empty else an
    if len(w):
        up, down = (w["Action"] == "up").sum(), (w["Action"] == "down").sum()
        raised = (w["priceTargetAction"] == "Raises").sum()
        lowered = (w["priceTargetAction"] == "Lowers").sum()
        chg = w["pt_chg"].dropna()
        r.update(an_n=float(np.log1p(len(w))), an_net_grade=float(up - down),
                 an_pt_net=float(raised - lowered) / max(1, raised + lowered),
                 an_pt_chg=float(np.clip(chg.mean(), -0.5, 0.5)) if len(chg) else 0.0)
    else:
        r.update(an_n=0.0, an_net_grade=0.0, an_pt_net=0.0, an_pt_chg=0.0)
    e = px.index.searchsorted(day) - 1
    s = spy.index.searchsorted(day) - 1
    if e >= 63 and s >= 63:
        r["px_rel_63"] = float(np.clip(_ret(px, e, 63) - _ret(spy, s, 63), -1, 1))
        r["px_rel_21"] = float(np.clip(_ret(px, e, 21) - _ret(spy, s, 21), -0.5, 0.5))
        r["px_vol_63"] = float(px.iloc[e - 63:e + 1].pct_change().std())
    else:
        r.update(px_rel_63=np.nan, px_rel_21=np.nan, px_vol_63=np.nan)
    return r


def _prices(market_dir: Path, ticker: str) -> pd.Series:
    f = market_dir / f"{ticker}_prices.csv"
    if not f.exists():
        return pd.Series(dtype=float)
    return pd.read_csv(f, index_col="date", parse_dates=True)["Close"].sort_index()


def market_features(keys: pd.DataFrame, market_dir: Path) -> pd.DataFrame:
    """``keys`` has columns ``ticker``, ``date`` (announcement, YYYY-MM-DD); one output row per key."""
    spy = _prices(market_dir, "SPY")
    rows = []
    for ticker, g in keys.groupby("ticker"):
        px = _prices(market_dir, ticker)
        f = market_dir / f"{ticker}_analyst.csv"
        an = prepare_analyst(pd.read_csv(f)) if f.exists() else pd.DataFrame()
        rows += [{"ticker": ticker, "date": d, **market_vector(an, px, spy, d)} for d in g["date"]]
    out = pd.DataFrame(rows).set_index(["ticker", "date"])[ANALYST_COLS + PRICE_COLS]
    # no price history yet (new listing) → neutral values
    return out.fillna({"px_rel_63": 0.0, "px_rel_21": 0.0}).fillna({"px_vol_63": out["px_vol_63"].median()})


def yahoo_inputs(ticker: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Live (analyst actions, prices, SPY prices) from Yahoo Finance, in the form ``market_vector`` takes."""
    import yfinance as yf

    def closes(t: str) -> pd.Series:
        px = yf.Ticker(t).history(start=PRICE_START, auto_adjust=True)["Close"]
        px.index = px.index.tz_localize(None).normalize()
        return px.sort_index()

    try:
        ud = yf.Ticker(ticker).upgrades_downgrades
    except Exception:  # no analyst coverage
        ud = None
    an = prepare_analyst(ud.reset_index()) if ud is not None and not ud.empty else pd.DataFrame()
    return an, closes(ticker), closes("SPY")
