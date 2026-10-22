# 02 — Labeler: did the company miss?

**File:** `src/data/labeler.py` · **Output:** `data/processed/labels.csv`

## What it does

For each company, it fetches every past earnings announcement with the analysts' EPS estimate and
the reported EPS, and writes one labelled row per announcement:

- `label = 1` (**miss**) if reported EPS < estimate
- `label = 0` (**met/beat**) otherwise

```bash
python src/data/labeler.py --transcripts-dir data/raw/transcripts          # default: --method direct
python src/data/labeler.py --method yfinance --tickers AAPL MSFT
```

Result: **1,899 rows, 258 misses (14%)**, roughly 2016–2026 for 49 tickers.

## Where the numbers come from

[yfinance](https://github.com/ranaroussi/yfinance) is an unofficial library for Yahoo Finance data.
`_fetch_earnings_history` tries:

1. `Ticker.get_earnings_dates(limit=60)` — about 10+ years of announcements with `EPS Estimate`,
   `Reported EPS` and `Surprise(%)`. **Used first.**
2. `Ticker.earnings_history` — only the last 4 quarters. Fallback.

Originally the order was reversed, which gave only 4 quarters per company (195 labels) — too
recent to overlap the 2020–2025 filings. Swapping them and raising `lookback_quarters` to 40 gave
1,899 labels.

## Methods (`--method`)

| Method | How a row is produced | Notes |
|---|---|---|
| `direct` (default) | One row per announcement from yfinance; transcripts are only used to discover the ticker list | What we use. Pairing with text happens later in `dataset.py` |
| `yfinance` | For each filing, find the nearest announcement within ±90 days | Pairs a filing with its *own* quarter → leaks the answer; not used |
| `text` | Extract revenue / net income from the MD&A and compare year-over-year | Rough fallback when no estimate data exists |

## Output columns

`ticker, company, date, quarter, year, eps_actual, eps_estimate, eps_surprise_pct, revenue,
revenue_yoy_pct, label, label_source, confidence_score`

- `date` is the **announcement date** — the key `dataset.py` joins on.
- `quarter`/`year` are calendar quarters of that date (not the company's fiscal quarter).

## Finding the transcripts

The scraper writes flat files (`AAPL.json`, each a list of records) while the labeler originally
expected folders (`AAPL/*.json`). `_ticker_files` now accepts both and skips files starting with `_`
(the combined `_all_transcripts.json`, which would double-count).

## Concepts to learn here

What an earnings surprise is; label design (what exactly is "ground truth"?); why the data source
and the amount of history it provides shape the whole project.
