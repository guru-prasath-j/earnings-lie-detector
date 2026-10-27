"""
SEC EDGAR earnings press-release scraper (8-K, Item 2.02, Exhibit 99.1)
=======================================================================
On the day a company announces quarterly results it files an 8-K with
**Item 2.02 "Results of Operations and Financial Condition"**; the press
release itself is attached as **Exhibit 99.1**.  Unlike the 10-Q MD&A, this is
where management talks up the quarter and (for many companies) gives guidance
for the next one.

Pipeline per company:
  1. All 8-K filings whose ``items`` include 2.02 (the submissions API keeps only
     ~1,000 recent filings, so older pages are fetched too — banks file thousands
     of forms a year)
  2. The filing's ``-index.htm`` page → the document typed ``EX-99.1``
     (fallback: a filename like ``ex991`` / ``exhibit99-1``, then any ``EX-99.*``)
  3. Download → strip iXBRL / HTML.  Tables (the financial statements) are
     dropped, keeping the narrative; if that leaves almost nothing, tables are kept

Output: ``data/raw/press_releases/{TICKER}.json``, a list of records with the
same ``date`` / ``text`` fields ``EarningsCallDataset`` reads, so
``EarningsCallDataset(transcripts_dir="data/raw/press_releases")`` pairs each
release with the *next* announcement (the release states the current quarter's
EPS, so it must never be paired with its own announcement).

Usage:
    python src/data/press_releases.py                      # 49 default tickers, 2016-07-01 → today, ~15 min
    python src/data/press_releases.py --tickers AAPL JPM
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scraper import (  # noqa: E402
    EDGAR_SUBMISSIONS_URL, HEADERS, REQUEST_DELAY, clean_ixbrl, download_filing,
    get_default_tickers, get_filing_base_url, load_company_tickers, normalize_text,
)

DEFAULT_OUTPUT_DIR = "data/raw/press_releases"
MIN_NARRATIVE_CHARS = 1500  # below this, the release is mostly tables → keep them

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_EX991_NAME = re.compile(r"ex(?:hibit)?[-_ ]?99[-_.]?0?1(?!\d)", re.IGNORECASE)


def _get_json(url: str) -> dict | None:
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    return resp.json() if resp.status_code == 200 else None


def earnings_8ks(cik: str, start_date: str, end_date: str) -> list[dict]:
    """8-K filings with Item 2.02 between the dates, from all submission pages."""
    data = _get_json(f"{EDGAR_SUBMISSIONS_URL}/CIK{cik}.json")
    if data is None:
        return []
    pages = [data["filings"]["recent"]]
    for page in data["filings"].get("files", []):
        if page["filingTo"] >= start_date:  # page overlaps the date range
            older = _get_json(f"{EDGAR_SUBMISSIONS_URL}/{page['name']}")
            if older:
                pages.append(older)

    out = []
    for p in pages:
        for form, items, acc, d, doc in zip(p["form"], p["items"], p["accessionNumber"],
                                            p["filingDate"], p["primaryDocument"]):
            if form == "8-K" and "2.02" in items.split(",") and start_date <= d <= end_date:
                out.append({"accessionNumber": acc, "filingDate": d, "items": items, "primaryDocument": doc})
    return sorted(out, key=lambda f: f["filingDate"])


def find_exhibit_991(cik: str, accession: str) -> str | None:
    """URL of the press-release exhibit, read from the filing's index page."""
    base = get_filing_base_url(cik, accession)
    html = download_filing(f"{base}/{accession}-index.htm")
    if html is None:
        return None
    docs = []  # (type, filename)
    for row in BeautifulSoup(html, "html.parser").select("table.tableFile tr"):
        cells = row.find_all("td")
        link = row.find("a")
        if len(cells) >= 4 and link:
            name = link["href"].rsplit("/", 1)[-1]
            if name.lower().endswith((".htm", ".html", ".txt")):
                docs.append((cells[3].get_text(strip=True).upper(), name))
    for pick in (lambda t, n: t == "EX-99.1",
                 lambda t, n: _EX991_NAME.search(n) is not None,
                 lambda t, n: t.startswith("EX-99")):
        for t, n in docs:
            if pick(t, n):
                return f"{base}/{n}"
    return None


def release_text(html: str) -> tuple[str, bool]:
    """Plain text of a press release; returns (text, tables_dropped)."""
    soup = BeautifulSoup(clean_ixbrl(html), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    def to_text(s: BeautifulSoup) -> str:
        # Join with spaces: inline spans (footnote superscripts, styled words) shouldn't break lines
        t = re.sub(r"\s+", " ", normalize_text(s.get_text(separator=" "))).strip()
        # Drop the SGML header ("EX-99.1 2 file.htm EX-99.1 Document") before the exhibit title
        m = re.search(r"\bExhibit\s+99", t[:600], re.IGNORECASE)
        return t[m.start():] if m else t

    full = to_text(soup)
    for table in soup("table"):
        table.decompose()
    narrative = to_text(soup)
    if len(narrative) >= MIN_NARRATIVE_CHARS:
        return narrative, True
    return full, False


def scrape_company(ticker: str, cik: str, name: str, start_date: str, end_date: str) -> list[dict]:
    filings = earnings_8ks(cik, start_date, end_date)
    logger.info("%s (%s): %d Item 2.02 8-Ks", ticker, name, len(filings))
    records = []
    for f in filings:
        url = find_exhibit_991(cik, f["accessionNumber"])
        html = download_filing(url) if url else None
        if html is None:
            logger.warning("  %s %s: no EX-99.1", ticker, f["filingDate"])
            continue
        text, tables_dropped = release_text(html)
        records.append({
            "company": ticker, "company_name": name, "cik": cik,
            "date": f["filingDate"], "form": "8-K", "items": f["items"],
            "accession_number": f["accessionNumber"], "exhibit_url": url,
            "tables_dropped": tables_dropped, "text": text, "text_length": len(text),
        })
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape 8-K Item 2.02 earnings press releases (EX-99.1)")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--start", default="2016-07-01", help="labels start in 2017; earlier releases feed history only")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ticker_map = load_company_tickers()
    total = 0
    for ticker in [t.upper() for t in (args.tickers or get_default_tickers())]:
        if ticker not in ticker_map:
            logger.warning("%s not in SEC ticker list, skipping", ticker)
            continue
        info = ticker_map[ticker]
        records = scrape_company(ticker, info["cik"], info["name"], args.start, args.end)
        if records:
            with open(out_dir / f"{ticker}.json", "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2, ensure_ascii=False)
        total += len(records)
    logger.info("Saved %d press releases to %s", total, out_dir)


if __name__ == "__main__":
    main()
