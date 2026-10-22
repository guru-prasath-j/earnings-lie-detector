#!/usr/bin/env python3
"""
Ground Truth Labeler for Earnings Call Lie Detector.

Labels each 10-Q filing with a binary outcome using two strategies:

  1. **yfinance (primary)** — earnings surprise: compares analyst consensus
     EPS estimate vs actual reported EPS.  Beat/met → 0, missed → 1.
  2. **Text analysis (fallback)** — extracts revenue and net-income figures
     from the MD&A text and compares year-over-year.  Both declined → 1.

Usage:
    python labeler.py --transcripts-dir data/raw/transcripts
    python labeler.py --method yfinance --tickers AAPL MSFT
    python labeler.py --method text --transcripts-dir data/raw/transcripts
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("labeler")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TranscriptRecord:
    """One loaded 10-Q filing."""
    ticker: str
    company: str
    date: str            # filing / call date  (YYYY-MM-DD)
    quarter: str         # e.g. "Q1"
    year: int
    text: str
    path: Path


@dataclass
class LabelRow:
    """One output row."""
    ticker: str = ""
    company: str = ""
    date: str = ""
    quarter: str = ""
    year: int = 0
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    eps_surprise_pct: Optional[float] = None
    revenue: Optional[float] = None
    revenue_yoy_pct: Optional[float] = None
    label: Optional[int] = None
    label_source: str = ""          # "yfinance" | "text_analysis"
    confidence_score: float = 0.0


# ---------------------------------------------------------------------------
# Helpers — date / quarter logic
# ---------------------------------------------------------------------------

def _date_to_fiscal_quarter(date_str: str) -> Tuple[int, int]:
    """Return (quarter_number, year) for a date string.

    Simple calendar-quarter mapping.  Most companies' fiscal quarters
    align with calendar quarters, and the few that don't (e.g. MSFT,
    NKE) are close enough for matching purposes.
    """
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    q = (dt.month - 1) // 3 + 1
    return q, dt.year


def _quarter_label(q: int) -> str:
    return f"Q{q}"


def _prev_quarter(q: int, y: int) -> Tuple[int, int]:
    """Return the previous calendar quarter."""
    if q == 1:
        return 4, y - 1
    return q - 1, y


def _same_quarter_last_year(q: int, y: int) -> Tuple[int, int]:
    return q, y - 1


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

_ticker_cache: Dict[str, yf.Ticker] = {}


def _get_ticker(symbol: str) -> yf.Ticker:
    """Get a cached yfinance Ticker object."""
    if symbol not in _ticker_cache:
        _ticker_cache[symbol] = yf.Ticker(symbol)
    return _ticker_cache[symbol]


def _fetch_earnings_history(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch earnings history via yfinance.

    Tries ``Ticker.get_earnings_dates()`` first (~10+ years of history),
    then ``Ticker.earnings_history`` (last 4 quarters only) as a fallback.

    Returns a DataFrame with at least columns:
        epsActual, epsEstimate, report_date
    or None on failure.
    """
    tk = _get_ticker(symbol)

    # --- attempt 1: get_earnings_dates ---
    try:
        ed = tk.get_earnings_dates(limit=60)
        if ed is not None and not ed.empty:
            rename_map = {}
            for col in ed.columns:
                lc = col.lower().replace(" ", "")
                if "epsactual" in lc or col == "Reported EPS":
                    rename_map[col] = "epsActual"
                elif "epsestimate" in lc or col == "EPS Estimate":
                    rename_map[col] = "epsEstimate"
                elif "surprise" in lc:
                    rename_map[col] = "surprisePercent"
            ed = ed.rename(columns=rename_map)
            if "epsActual" in ed.columns and "epsEstimate" in ed.columns:
                if isinstance(ed.index, pd.DatetimeIndex):
                    ed["report_date"] = ed.index
                return ed
    except Exception as exc:
        logger.debug("get_earnings_dates failed for %s: %s", symbol, exc)

    # --- attempt 2: earnings_history attribute ---
    try:
        eh = tk.earnings_history
        if eh is not None and not eh.empty:
            # Normalise column names — yfinance versions differ
            rename_map = {}
            for col in eh.columns:
                lc = col.lower().replace(" ", "")
                if "actual" in lc and "eps" in lc:
                    rename_map[col] = "epsActual"
                elif "estimate" in lc and "eps" in lc:
                    rename_map[col] = "epsEstimate"
                elif "surprise" in lc and "pct" in lc.replace("%", "pct"):
                    rename_map[col] = "surprisePercent"
                elif "surprise" in lc:
                    rename_map[col] = "epsDifference"
            eh = eh.rename(columns=rename_map)

            # Make sure we have the basics
            if "epsActual" in eh.columns and "epsEstimate" in eh.columns:
                # Attach the index as a date column if it's a DatetimeIndex
                if isinstance(eh.index, pd.DatetimeIndex):
                    eh["report_date"] = eh.index
                elif "quarter" in eh.columns:
                    # Some versions give a "quarter" column like "4Q2023"
                    pass
                return eh
    except Exception as exc:
        logger.debug("earnings_history failed for %s: %s", symbol, exc)

    return None


def _match_earnings_row(
    earnings_df: pd.DataFrame,
    filing_date_str: str,
    quarter: int,
    year: int,
) -> Optional[pd.Series]:
    """Find the earnings row that matches a given filing quarter.

    10-Q filing dates don't always coincide exactly with the earnings
    report date, so we use a window-based match:
      1. Try to match by quarter label if one exists.
      2. Fall back to nearest report_date within ±90 days of filing date.
    """

    filing_dt = datetime.strptime(filing_date_str[:10], "%Y-%m-%d")

    # Strategy 1: match by quarter column if present
    if "quarter" in earnings_df.columns:
        # Quarter strings can look like "4Q2023", "Q4 2023", etc.
        for idx, row in earnings_df.iterrows():
            q_str = str(row.get("quarter", ""))
            # Extract quarter number and year from the string
            m = re.search(r"(\d)[Qq]\s*(\d{4})", q_str)
            if not m:
                m = re.search(r"[Qq](\d)\s*(\d{4})", q_str)
            if m:
                rq, ry = int(m.group(1)), int(m.group(2))
                if rq == quarter and ry == year:
                    return row

    # Strategy 2: closest report_date within ±90 days
    if "report_date" in earnings_df.columns:
        df_copy = earnings_df.copy()
        df_copy["report_date"] = pd.to_datetime(df_copy["report_date"], errors="coerce")
        df_copy = df_copy.dropna(subset=["report_date", "epsActual"])

        if not df_copy.empty:
            df_copy["_delta"] = (df_copy["report_date"] - filing_dt).abs()
            best = df_copy.loc[df_copy["_delta"].idxmin()]
            if best["_delta"] <= timedelta(days=90):
                return best

    return None


def _fetch_quarterly_revenue(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch quarterly revenue from the income statement."""
    tk = _get_ticker(symbol)
    try:
        stmt = tk.quarterly_income_stmt
        if stmt is None or stmt.empty:
            return None

        # Find revenue row — yfinance uses several possible labels
        revenue_labels = [
            "Total Revenue", "Revenue", "Net Revenue",
            "Total Net Revenue", "Net Sales", "Sales",
        ]
        for label in revenue_labels:
            if label in stmt.index:
                rev_series = stmt.loc[label]
                # Columns are dates
                df = pd.DataFrame({
                    "date": rev_series.index,
                    "revenue": rev_series.values,
                })
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date", "revenue"])
                return df.sort_values("date")
        return None
    except Exception as exc:
        logger.debug("quarterly_income_stmt failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# yfinance labeling
# ---------------------------------------------------------------------------

def label_with_yfinance(
    record: TranscriptRecord,
) -> Optional[LabelRow]:
    """Label a single filing using yfinance earnings surprise.

    Returns a ``LabelRow`` on success, ``None`` if data is unavailable.
    """
    q, y = _date_to_fiscal_quarter(record.date)

    earnings_df = _fetch_earnings_history(record.ticker)
    if earnings_df is None:
        return None

    match = _match_earnings_row(earnings_df, record.date, q, y)
    if match is None:
        return None

    eps_actual = match.get("epsActual")
    eps_estimate = match.get("epsEstimate")

    if pd.isna(eps_actual) or pd.isna(eps_estimate):
        return None

    eps_actual = float(eps_actual)
    eps_estimate = float(eps_estimate)

    # Surprise percentage
    if abs(eps_estimate) > 1e-6:
        surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
    else:
        surprise_pct = 0.0 if abs(eps_actual - eps_estimate) < 1e-6 else 100.0

    # Label: miss → 1, beat/met → 0
    label = 1 if eps_actual < eps_estimate else 0

    # Revenue (optional enrichment)
    revenue = None
    revenue_yoy_pct = None
    rev_df = _fetch_quarterly_revenue(record.ticker)
    if rev_df is not None and len(rev_df) >= 2:
        filing_dt = datetime.strptime(record.date[:10], "%Y-%m-%d")
        rev_df["_delta"] = (rev_df["date"] - filing_dt).abs()
        nearest_idx = rev_df["_delta"].idxmin()
        nearest = rev_df.loc[nearest_idx]
        if nearest["_delta"] <= timedelta(days=90):
            revenue = float(nearest["revenue"])
            # Find same quarter last year for YoY
            target_date = nearest["date"] - pd.DateOffset(years=1)
            rev_df["_delta_yoy"] = (rev_df["date"] - target_date).abs()
            yoy_idx = rev_df["_delta_yoy"].idxmin()
            yoy_row = rev_df.loc[yoy_idx]
            if yoy_row["_delta_yoy"] <= timedelta(days=120) and abs(float(yoy_row["revenue"])) > 0:
                revenue_yoy_pct = ((revenue - float(yoy_row["revenue"]))
                                   / abs(float(yoy_row["revenue"]))) * 100

    # Confidence: high when we have a solid EPS match
    confidence = 0.85
    if revenue is not None:
        confidence = 0.95

    return LabelRow(
        ticker=record.ticker,
        company=record.company,
        date=record.date,
        quarter=_quarter_label(q),
        year=y,
        eps_actual=eps_actual,
        eps_estimate=eps_estimate,
        eps_surprise_pct=round(surprise_pct, 4),
        revenue=revenue,
        revenue_yoy_pct=round(revenue_yoy_pct, 4) if revenue_yoy_pct is not None else None,
        label=label,
        label_source="yfinance",
        confidence_score=confidence,
    )


# ---------------------------------------------------------------------------
# Text-based labeling (fallback)
# ---------------------------------------------------------------------------

# Regex patterns to extract dollar amounts from MD&A text
_DOLLAR_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?",
    re.IGNORECASE,
)

_REVENUE_CONTEXT_RE = re.compile(
    r"(?:total\s+)?(?:net\s+)?(?:revenue|sales|net\s+sales)\s+"
    r"(?:was|were|of|totaled|totalling|reached|amounted\s+to)\s+"
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?",
    re.IGNORECASE,
)

_NET_INCOME_CONTEXT_RE = re.compile(
    r"(?:net\s+(?:income|earnings)|income\s+from\s+(?:continuing\s+)?operations)\s+"
    r"(?:was|were|of|totaled|totalling|reached|amounted\s+to)\s+"
    r"\$?\s*([\d,]+(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?",
    re.IGNORECASE,
)

_DECREASE_RE = re.compile(
    r"(?:revenue|sales|net\s+sales).*?"
    r"(?:decreas|declin|fell|drop|lower|reduc|contract|down)\w*"
    r".*?(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE | re.DOTALL,
)

_INCOME_DECREASE_RE = re.compile(
    r"(?:net\s+(?:income|earnings)|operating\s+(?:income|earnings)).*?"
    r"(?:decreas|declin|fell|drop|lower|reduc|contract|down)\w*"
    r".*?(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE | re.DOTALL,
)

_INCREASE_RE = re.compile(
    r"(?:revenue|sales|net\s+sales).*?"
    r"(?:increas|grew|growth|rose|higher|up|expand|improv)\w*"
    r".*?(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE | re.DOTALL,
)


def _normalise_amount(value_str: str, unit: Optional[str]) -> float:
    """Convert a parsed dollar string to a float in raw dollars."""
    value = float(value_str.replace(",", ""))
    if unit:
        u = unit.upper()[0]
        if u == "B":
            value *= 1e9
        elif u == "M":
            value *= 1e6
        elif u in ("K", "T"):
            value *= 1e3
    return value


def _extract_text_metrics(text: str) -> Dict[str, Any]:
    """Pull revenue and income signals from MD&A text.

    Returns dict with keys: revenue, net_income, revenue_change_pct,
    income_change_pct, revenue_declined, income_declined.
    """
    result: Dict[str, Any] = {
        "revenue": None,
        "net_income": None,
        "revenue_change_pct": None,
        "income_change_pct": None,
        "revenue_declined": None,
        "income_declined": None,
    }

    # Truncate to first 15k chars — MD&A summary is near the top
    snippet = text[:15000]

    # --- absolute revenue ---
    m = _REVENUE_CONTEXT_RE.search(snippet)
    if m:
        result["revenue"] = _normalise_amount(m.group(1), m.group(2))

    # --- absolute net income ---
    m = _NET_INCOME_CONTEXT_RE.search(snippet)
    if m:
        result["net_income"] = _normalise_amount(m.group(1), m.group(2))

    # --- revenue change signals ---
    m_dec = _DECREASE_RE.search(snippet)
    m_inc = _INCREASE_RE.search(snippet)
    if m_dec:
        result["revenue_change_pct"] = -float(m_dec.group(1))
        result["revenue_declined"] = True
    elif m_inc:
        result["revenue_change_pct"] = float(m_inc.group(1))
        result["revenue_declined"] = False

    # --- income change signals ---
    m_inc_dec = _INCOME_DECREASE_RE.search(snippet)
    if m_inc_dec:
        result["income_change_pct"] = -float(m_inc_dec.group(1))
        result["income_declined"] = True
    else:
        # If we found revenue increasing, assume income didn't decline
        # unless explicitly stated (conservative)
        result["income_declined"] = False

    return result


def label_with_text(
    record: TranscriptRecord,
    prior_records: Dict[Tuple[str, int, int], TranscriptRecord],
) -> Optional[LabelRow]:
    """Label a filing by analysing MD&A text for performance signals.

    The approach:
    - Extract revenue / income change percentages from text
    - If both revenue AND income declined → label = 1
    - If text signals can't be extracted → return None

    ``prior_records`` maps (ticker, quarter, year) → TranscriptRecord
    for potential cross-reference.
    """
    q, y = _date_to_fiscal_quarter(record.date)
    metrics = _extract_text_metrics(record.text)

    # Need at least one clear signal
    if metrics["revenue_declined"] is None and metrics["income_declined"] is None:
        return None

    revenue_declined = metrics["revenue_declined"] or False
    income_declined = metrics["income_declined"] or False

    # Label logic: both declined → 1 (bad quarter, likely misleading MD&A)
    label = 1 if (revenue_declined and income_declined) else 0

    # Confidence is lower for text-based analysis
    confidence = 0.40
    if metrics["revenue_change_pct"] is not None:
        confidence += 0.15
    if metrics["income_change_pct"] is not None:
        confidence += 0.15
    if metrics["revenue"] is not None:
        confidence += 0.05

    return LabelRow(
        ticker=record.ticker,
        company=record.company,
        date=record.date,
        quarter=_quarter_label(q),
        year=y,
        eps_actual=None,
        eps_estimate=None,
        eps_surprise_pct=None,
        revenue=metrics["revenue"],
        revenue_yoy_pct=metrics["revenue_change_pct"],
        label=label,
        label_source="text_analysis",
        confidence_score=confidence,
    )


# ---------------------------------------------------------------------------
# Transcript loading
# ---------------------------------------------------------------------------

def _ticker_files(transcripts_dir: Path) -> Dict[str, List[Path]]:
    """Map ticker -> JSON files, supporting both on-disk layouts:

    - ``{transcripts_dir}/{ticker}/*.json``
    - ``{transcripts_dir}/{ticker}.json`` (flat files written by scraper.py)

    Entries starting with ``_`` (e.g. ``_all_transcripts.json``) are skipped
    so records are not loaded twice.
    """
    files: Dict[str, List[Path]] = {}
    if not transcripts_dir.exists():
        return files
    for entry in sorted(transcripts_dir.iterdir()):
        if entry.name.startswith("_"):
            continue
        if entry.is_dir():
            files.setdefault(entry.name.upper(), []).extend(sorted(entry.glob("*.json")))
        elif entry.suffix == ".json":
            files.setdefault(entry.stem.upper(), []).append(entry)
    return files


def load_transcripts(
    transcripts_dir: Path,
    tickers: Optional[List[str]] = None,
) -> List[TranscriptRecord]:
    """Load transcript JSON files from ``data/raw/transcripts/``.

    Files may live at ``{ticker}/*.json`` or ``{ticker}.json``. Each JSON can be either:
    - A list of records (from scraper v2): ``[{text, date, ...}, ...]``
    - A single record: ``{text, date, ...}``
    """
    records: List[TranscriptRecord] = []
    if not transcripts_dir.exists():
        logger.error("Transcripts directory not found: %s", transcripts_dir)
        return records

    ticker_files = _ticker_files(transcripts_dir)
    if tickers:
        allowed = {t.upper() for t in tickers}
        ticker_files = {t: fs for t, fs in ticker_files.items() if t in allowed}

    for ticker, json_files in ticker_files.items():
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", json_file, exc)
                continue

            # Normalise to a list of entries
            entries = data if isinstance(data, list) else [data]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                text = entry.get("text", entry.get("mda_text", ""))
                date_str = entry.get("date", entry.get("filing_date", ""))
                if not date_str or not text:
                    continue

                # Ensure date is a string in YYYY-MM-DD format
                date_str = str(date_str)[:10]
                try:
                    q, y = _date_to_fiscal_quarter(date_str)
                except ValueError:
                    continue

                company = entry.get("company", entry.get("name", ticker))
                records.append(TranscriptRecord(
                    ticker=ticker,
                    company=company,
                    date=date_str,
                    quarter=_quarter_label(q),
                    year=y,
                    text=text,
                    path=json_file,
                ))

    logger.info("Loaded %d transcript records from %s", len(records), transcripts_dir)
    return records


# ---------------------------------------------------------------------------
# Main labeling pipeline
# ---------------------------------------------------------------------------

def label_tickers_directly(
    tickers: List[str],
    output_path: Path,
    lookback_quarters: int = 40,
) -> pd.DataFrame:
    """Label tickers using yfinance earnings data only (no transcript files needed).

    This is the most reliable labeling mode: it pulls the last N quarters
    of EPS surprise data from yfinance and labels each as met/beat (0) or
    missed (1).  No transcript parsing is required.

    Args:
        tickers: List of ticker symbols.
        output_path: Path for the output CSV.
        lookback_quarters: How many past quarters of data to fetch per ticker.

    Returns:
        DataFrame of labeling results.
    """
    rows: List[LabelRow] = []

    for ticker in tqdm(tickers, desc="Labeling via yfinance"):
        try:
            earnings_df = _fetch_earnings_history(ticker)
            if earnings_df is None or earnings_df.empty:
                logger.warning("No earnings data for %s", ticker)
                continue

            # Process each row in the earnings history
            for idx, erow in earnings_df.head(lookback_quarters).iterrows():
                eps_actual = erow.get("epsActual")
                eps_estimate = erow.get("epsEstimate")

                if pd.isna(eps_actual) or pd.isna(eps_estimate):
                    continue

                eps_actual = float(eps_actual)
                eps_estimate = float(eps_estimate)

                # Determine date
                if "report_date" in erow and pd.notna(erow["report_date"]):
                    date_str = pd.Timestamp(erow["report_date"]).strftime("%Y-%m-%d")
                elif isinstance(idx, pd.Timestamp):
                    date_str = idx.strftime("%Y-%m-%d")
                else:
                    continue

                q, y = _date_to_fiscal_quarter(date_str)

                # Surprise percentage
                if abs(eps_estimate) > 1e-6:
                    surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
                else:
                    surprise_pct = 0.0 if abs(eps_actual - eps_estimate) < 1e-6 else 100.0

                # Label: miss → 1, beat/met → 0
                label = 1 if eps_actual < eps_estimate else 0

                rows.append(LabelRow(
                    ticker=ticker,
                    company=ticker,
                    date=date_str,
                    quarter=_quarter_label(q),
                    year=y,
                    eps_actual=eps_actual,
                    eps_estimate=eps_estimate,
                    eps_surprise_pct=round(surprise_pct, 4),
                    label=label,
                    label_source="yfinance_direct",
                    confidence_score=0.90,
                ))

        except Exception as exc:
            logger.error("Failed to process %s: %s", ticker, exc)
            continue

        time.sleep(0.3)  # rate-limit yfinance

    # Build DataFrame
    df = pd.DataFrame([
        {
            "ticker": r.ticker,
            "company": r.company,
            "date": r.date,
            "quarter": r.quarter,
            "year": r.year,
            "eps_actual": r.eps_actual,
            "eps_estimate": r.eps_estimate,
            "eps_surprise_pct": r.eps_surprise_pct,
            "revenue": r.revenue,
            "revenue_yoy_pct": r.revenue_yoy_pct,
            "label": r.label,
            "label_source": r.label_source,
            "confidence_score": r.confidence_score,
        }
        for r in rows
    ])

    # Deduplicate by (ticker, date)
    if not df.empty:
        df = df.drop_duplicates(subset=["ticker", "date"], keep="first")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    labeled = len(df)
    if labeled > 0:
        met = int((df["label"] == 0).sum())
        missed = int((df["label"] == 1).sum())
        logger.info("Saved %d rows to %s", labeled, output_path)
        logger.info(
            "Summary: %d labeled | %d met/beat (%.0f%%) | %d missed (%.0f%%)",
            labeled, met, 100 * met / labeled, missed, 100 * missed / labeled,
        )
    else:
        logger.warning("No labels produced. Check yfinance connectivity.")

    return df


def run_labeling(
    transcripts_dir: Path,
    output_path: Path,
    tickers: Optional[List[str]] = None,
    method: str = "both",
) -> pd.DataFrame:
    """Run the full labeling pipeline.

    Args:
        transcripts_dir: Root dir containing ``{ticker}/`` sub-folders.
        output_path: Path for the output CSV.
        tickers: Optional whitelist of tickers to process.
        method: One of ``"yfinance"``, ``"text"``, ``"both"``, ``"direct"``.
            ``"direct"`` skips transcript parsing entirely and labels
            from yfinance earnings data only (recommended for 10-Q MD&A
            filings that don't contain forward-looking guidance).

    Returns:
        DataFrame of labeling results.
    """
    # --- Direct mode: label from yfinance without needing transcripts ---
    if method == "direct":
        if not tickers:
            # Discover tickers from transcript directory / file names
            tickers = list(_ticker_files(transcripts_dir))
            if not tickers:
                logger.error("No tickers specified and none found in transcripts dir.")
                return pd.DataFrame()
        return label_tickers_directly(tickers, output_path)

    records = load_transcripts(transcripts_dir, tickers)
    if not records:
        logger.error("No transcript records found. Check --transcripts-dir.")
        return pd.DataFrame()

    # Build a lookup for text-based cross-referencing
    prior_map: Dict[Tuple[str, int, int], TranscriptRecord] = {}
    for rec in records:
        q, y = _date_to_fiscal_quarter(rec.date)
        prior_map[(rec.ticker, q, y)] = rec

    rows: List[LabelRow] = []
    yf_success = 0
    text_success = 0
    failed = 0

    # Group by ticker to batch yfinance calls
    ticker_groups: Dict[str, List[TranscriptRecord]] = {}
    for rec in records:
        ticker_groups.setdefault(rec.ticker, []).append(rec)

    for ticker, group in tqdm(ticker_groups.items(), desc="Labeling tickers"):
        # Pre-fetch yfinance data once per ticker (cached internally)
        if method in ("yfinance", "both"):
            try:
                _fetch_earnings_history(ticker)
                _fetch_quarterly_revenue(ticker)
            except Exception:
                pass
            # Small delay to avoid rate limits
            time.sleep(0.2)

        for rec in group:
            label_row: Optional[LabelRow] = None

            # Try yfinance first (or exclusively)
            if method in ("yfinance", "both"):
                try:
                    label_row = label_with_yfinance(rec)
                except Exception as exc:
                    logger.debug("yfinance labeling failed for %s %s: %s",
                                 rec.ticker, rec.date, exc)

            if label_row is not None:
                yf_success += 1
                rows.append(label_row)
                continue

            # Fall back to text analysis
            if method in ("text", "both"):
                try:
                    label_row = label_with_text(rec, prior_map)
                except Exception as exc:
                    logger.debug("Text labeling failed for %s %s: %s",
                                 rec.ticker, rec.date, exc)

            if label_row is not None:
                text_success += 1
                rows.append(label_row)
                continue

            # Neither method produced a label — record with label=None
            q, y = _date_to_fiscal_quarter(rec.date)
            rows.append(LabelRow(
                ticker=rec.ticker,
                company=rec.company,
                date=rec.date,
                quarter=_quarter_label(q),
                year=y,
                label=None,
                label_source="",
                confidence_score=0.0,
            ))
            failed += 1

    # Build DataFrame
    df = pd.DataFrame([
        {
            "ticker": r.ticker,
            "company": r.company,
            "date": r.date,
            "quarter": r.quarter,
            "year": r.year,
            "eps_actual": r.eps_actual,
            "eps_estimate": r.eps_estimate,
            "eps_surprise_pct": r.eps_surprise_pct,
            "revenue": r.revenue,
            "revenue_yoy_pct": r.revenue_yoy_pct,
            "label": r.label,
            "label_source": r.label_source,
            "confidence_score": r.confidence_score,
        }
        for r in rows
    ])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    labeled = df["label"].notna().sum()
    if labeled > 0:
        met = int((df["label"] == 0).sum())
        missed = int((df["label"] == 1).sum())
        logger.info("Saved %d rows to %s", len(df), output_path)
        logger.info(
            "Summary: %d labeled | %d met/beat (%.0f%%) | %d missed (%.0f%%) | "
            "%d unlabeled",
            labeled, met, 100 * met / labeled, missed, 100 * missed / labeled,
            len(df) - labeled,
        )
        logger.info(
            "Sources: yfinance=%d, text_analysis=%d, unlabeled=%d",
            yf_success, text_success, failed,
        )
    else:
        logger.warning("No labels produced. Check data and yfinance connectivity.")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ground truth labels for earnings call lie detector."
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path("data/raw/transcripts"),
        help="Root directory containing {ticker}/ sub-folders with JSON files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/labels.csv"),
        help="Path for the output labels CSV.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Whitelist of tickers to process (default: all).",
    )
    parser.add_argument(
        "--method",
        choices=["yfinance", "text", "both", "direct"],
        default="direct",
        help="Labeling method: direct (recommended — yfinance EPS data only, no "
             "transcript parsing needed), yfinance (EPS surprise matched to transcript "
             "dates), text (MD&A analysis), or both (yfinance first, text fallback). "
             "Default: direct.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("Labeler starting")
    logger.info("  Transcripts : %s", args.transcripts_dir)
    logger.info("  Output      : %s", args.output_path)
    logger.info("  Method      : %s", args.method)
    logger.info("  Tickers     : %s", args.tickers or "all")

    run_labeling(
        transcripts_dir=args.transcripts_dir,
        output_path=args.output_path,
        tickers=args.tickers,
        method=args.method,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()