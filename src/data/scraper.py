"""
SEC EDGAR Transcript Scraper (v2 — fixed iXBRL + document resolution)
=====================================================================
Scrapes Management Discussion & Analysis (MD&A) sections from 10-Q filings
for S&P 500 companies. These sections contain forward-looking guidance
that we'll later compare against actual results.

Key fixes over v1:
  - Resolves the actual 10-Q document from the filing index (primaryDocument
    sometimes points to a cover page or XBRL viewer)
  - Handles iXBRL tags (<ix:nonnumeric>, <ix:nonFraction>, etc.)
  - Handles Unicode curly quotes (U+2019), em dashes, non-breaking spaces
  - Adds a debug mode to save raw HTML for inspection

Usage:
    python src/data/scraper.py --tickers AAPL MSFT GOOGL
    python src/data/scraper.py --all --start 2020-01-01 --end 2025-12-31
    python src/data/scraper.py --tickers AAPL --debug   # saves raw HTML for inspection
"""

import argparse
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# SEC requires a User-Agent header with your identity — update this with YOUR info
USER_AGENT = "EarningsLieDetector guruprasathjay37@gmail.com"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# SEC rate limit: max 10 requests/second — we stay safe at ~8/sec
REQUEST_DELAY = 0.12

# Base URLs
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Output directory
DEFAULT_OUTPUT_DIR = "data/raw/transcripts"

# Debug mode: save raw HTML to this directory for inspection
DEBUG_DIR = "data/raw/debug_html"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Ticker / CIK lookup
# ---------------------------------------------------------------------------

def load_company_tickers() -> dict[str, dict]:
    """
    Download SEC's master ticker-to-CIK mapping.
    Returns dict: {ticker: {"cik": "0001234567", "name": "Company Name"}}
    """
    logger.info("Downloading SEC company tickers list...")
    resp = requests.get(COMPANY_TICKERS_URL, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    ticker_map = {}
    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik = str(entry["cik_str"]).zfill(10)
        ticker_map[ticker] = {"cik": cik, "name": entry["title"]}

    logger.info(f"Loaded {len(ticker_map)} company tickers")
    return ticker_map


def get_default_tickers() -> list[str]:
    """
    A curated list of ~50 major S&P 500 companies across sectors.
    With ~20 quarterly filings each (2020-2025), this gives 500-1000 filings.
    """
    return [
        # Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "CRM",
        "ADBE", "INTC", "AMD", "ORCL", "CSCO", "QCOM", "AVGO",
        # Finance
        "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "AXP",
        # Healthcare
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO",
        # Consumer
        "WMT", "KO", "PEP", "PG", "COST", "NKE", "MCD", "SBUX",
        # Industrial / Energy
        "XOM", "CVX", "BA", "CAT", "GE", "UPS", "HON",
        # Telecom / Media
        "DIS", "NFLX", "CMCSA", "T", "VZ",
    ]


# ---------------------------------------------------------------------------
# 2. Fetch filing metadata from EDGAR
# ---------------------------------------------------------------------------

def get_company_filings(cik: str, filing_type: str = "10-Q") -> list[dict]:
    """
    Fetch the list of filings for a company from EDGAR's submissions API.
    """
    url = f"{EDGAR_SUBMISSIONS_URL}/CIK{cik}.json"
    time.sleep(REQUEST_DELAY)

    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        logger.warning(f"Failed to fetch filings for CIK {cik}: HTTP {resp.status_code}")
        return []

    data = resp.json()
    recent = data.get("filings", {}).get("recent", {})

    if not recent:
        return []

    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    filings = []
    for i in range(len(forms)):
        if forms[i] == filing_type:
            filings.append({
                "accessionNumber": accessions[i],
                "filingDate": dates[i],
                "primaryDocument": primary_docs[i],
                "form": forms[i],
            })

    return filings


def filter_filings_by_date(
    filings: list[dict],
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Filter filings to those within the date range."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    return [
        f for f in filings
        if start <= datetime.strptime(f["filingDate"], "%Y-%m-%d") <= end
    ]


# ---------------------------------------------------------------------------
# 3. Resolve the actual 10-Q document from the filing index
# ---------------------------------------------------------------------------

def get_filing_base_url(cik: str, accession_number: str) -> str:
    """Build the base URL for a filing's directory on EDGAR."""
    acc_no_dashes = accession_number.replace("-", "")
    return f"{EDGAR_ARCHIVES_URL}/{cik.lstrip('0')}/{acc_no_dashes}"


def resolve_10q_document(cik: str, accession_number: str, primary_doc: str) -> str:
    """
    Resolve the actual 10-Q document URL.

    The primaryDocument from the submissions API sometimes points to a cover
    page, an XBRL viewer, or other non-body documents. We fetch the filing
    index to find the actual 10-Q body HTML.

    Strategy:
    1. Try the filing index JSON to find the right document
    2. Fall back to the primaryDocument if the index approach fails
    """
    base_url = get_filing_base_url(cik, accession_number)

    # Try the filing index JSON first
    index_url = f"{base_url}/index.json"
    time.sleep(REQUEST_DELAY)

    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            index_data = resp.json()
            items = index_data.get("directory", {}).get("item", [])

            # Look for the main 10-Q document:
            # - Usually the largest .htm file
            # - Often named like ticker-YYYYMMDD.htm
            # - Has type "10-Q" in the index
            htm_files = []
            for item in items:
                name = item.get("name", "")
                doc_type = item.get("type", "")
                size = int(item.get("size", "0") or "0")

                # Skip non-HTML files, R-files (XBRL reports), and tiny files
                if not name.lower().endswith((".htm", ".html")):
                    continue
                if name.startswith("R") and name[1:].split(".")[0].isdigit():
                    continue  # R1.htm, R2.htm etc are XBRL viewer files

                htm_files.append({
                    "name": name,
                    "type": doc_type,
                    "size": size,
                    "url": f"{base_url}/{name}",
                })

            # Prefer files explicitly typed as "10-Q"
            for f in htm_files:
                if f["type"] == "10-Q":
                    logger.debug(f"  Resolved 10-Q doc from index type: {f['name']}")
                    return f["url"]

            # Otherwise pick the largest .htm file (the 10-Q body is always
            # the biggest HTML doc in the filing — cover pages are tiny)
            if htm_files:
                largest = max(htm_files, key=lambda x: x["size"])
                if largest["size"] > 50000:  # 10-Q bodies are usually > 100KB
                    logger.debug(f"  Resolved 10-Q doc by size: {largest['name']} ({largest['size']} bytes)")
                    return largest["url"]

    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        logger.debug(f"  Index resolution failed: {e}")

    # Fallback: use the primary document directly
    return f"{base_url}/{primary_doc}"


# ---------------------------------------------------------------------------
# 4. Download and parse the 10-Q filing HTML
# ---------------------------------------------------------------------------

def download_filing(url: str) -> str | None:
    """Download a filing's HTML content."""
    time.sleep(REQUEST_DELAY)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.warning(f"Failed to download {url}: {e}")
        return None


def normalize_text(text: str) -> str:
    """
    Normalize Unicode characters that break our regex matching.
    SEC filings are full of these:
      - U+2019 (right single quote / curly apostrophe) → '
      - U+2018 (left single quote) → '
      - U+201C / U+201D (curly double quotes) → "
      - U+2014 (em dash) → --
      - U+2013 (en dash) → -
      - U+00A0 (non-breaking space) → regular space
      - U+200B (zero-width space) → removed
    """
    replacements = {
        "’": "'",  # right single quote
        "‘": "'",  # left single quote
        "“": '"',  # left double quote
        "”": '"',  # right double quote
        "—": "--",  # em dash
        "–": "-",  # en dash
        " ": " ",  # non-breaking space
        "​": "",   # zero-width space
        "\xa0": " ",    # another form of NBSP
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_ixbrl(html: str) -> str:
    """
    Strip iXBRL tags from the HTML so BeautifulSoup can parse it cleanly.
    iXBRL wraps content in tags like <ix:nonnumeric>, <ix:nonFraction>, etc.
    These are in the 'ix' XML namespace and confuse HTML parsers.
    """
    # Remove all ix: namespace tags but keep their content
    # e.g., <ix:nonnumeric ...>some text</ix:nonnumeric> → some text
    html = re.sub(r"</?ix:[^>]*>", "", html)
    return html


def extract_mda_section(html: str, debug_info: str = "") -> str | None:
    """
    Extract the Management's Discussion & Analysis section from a 10-Q filing.

    This handles:
      - Standard HTML 10-Q filings
      - iXBRL filings (strips ix: tags first)
      - Various heading formats ("Item 2", "ITEM 2", with/without periods,
        dashes, em dashes)
      - Unicode characters (curly quotes, em dashes, NBSPs)
    """
    # Step 1: Strip iXBRL tags
    html = clean_ixbrl(html)

    # Step 2: Parse with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, and hidden elements
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Step 3: Get plain text and normalize Unicode
    text = soup.get_text(separator="\n")
    text = normalize_text(text)

    # Step 4: Search for MD&A section using flexible patterns
    # We search case-insensitively and allow lots of variation
    mda_start_patterns = [
        # "Item 2" with various separators and "Management's Discussion"
        r"Item\s*2\.?\s*[-—–.\s]*\s*Management'?s?\s*Discussion\s*and\s*Analysis",
        # Just "Management's Discussion and Analysis" as a standalone heading
        r"(?:^|\n)\s*Management'?s?\s*Discussion\s*and\s*Analysis\s*of\s*Financial\s*Condition",
        # Broader fallback
        r"Item\s*2[.\s]*Management",
    ]

    # End patterns: the section after MD&A
    mda_end_patterns = [
        r"Item\s*3\.?\s*[-—–.\s]*\s*Quantitative\s*and\s*Qualitative",
        r"Item\s*3\.?\s*[-—–.\s]*\s*Market\s*Risk",
        r"Item\s*4\.?\s*[-—–.\s]*\s*Controls?\s*and\s*Procedures?",
        # Table of contents "Item 3" that appears after MD&A content
        r"(?:^|\n)\s*Item\s*3\.",
    ]

    # Find the START of MD&A
    mda_start = None
    for pattern in mda_start_patterns:
        # Find ALL matches — we want the LAST one (the actual section,
        # not a table-of-contents reference)
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            # Use the last match — earlier matches are usually TOC entries
            mda_start = matches[-1].start()
            logger.debug(f"  MD&A start matched pattern: {pattern}")
            break

    if mda_start is None:
        # Last resort: try to find any "Management's Discussion" text
        match = re.search(
            r"Management'?s?\s+Discussion\s+and\s+Analysis",
            text,
            re.IGNORECASE,
        )
        if match:
            # Find the last occurrence
            matches = list(re.finditer(
                r"Management'?s?\s+Discussion\s+and\s+Analysis",
                text,
                re.IGNORECASE,
            ))
            mda_start = matches[-1].start()
        else:
            logger.debug(f"  No MD&A heading found in text ({len(text)} chars)")
            if debug_info:
                # In debug mode, show what text looks like around potential matches
                for snippet_pattern in ["Item 2", "item 2", "ITEM 2", "Management", "MANAGEMENT"]:
                    idx = text.lower().find(snippet_pattern.lower())
                    if idx >= 0:
                        snippet = text[max(0, idx-50):idx+100]
                        logger.info(f"  DEBUG found '{snippet_pattern}' at pos {idx}: ...{repr(snippet)}...")
            return None

    # Find the END of MD&A
    mda_end = len(text)
    search_from = mda_start + 200  # skip past the heading we matched

    for pattern in mda_end_patterns:
        match = re.search(pattern, text[search_from:], re.IGNORECASE)
        if match:
            candidate_end = search_from + match.start()
            # Only accept if there's substantial content between start and end
            if candidate_end - mda_start > 500:
                mda_end = candidate_end
                logger.debug(f"  MD&A end matched pattern: {pattern}")
                break

    mda_text = text[mda_start:mda_end].strip()

    # Clean up whitespace
    mda_text = re.sub(r"\n{3,}", "\n\n", mda_text)
    mda_text = re.sub(r"[ \t]+", " ", mda_text)
    # Remove page headers/footers that repeat throughout
    mda_text = re.sub(r"\n\s*Table of Contents\s*\n", "\n", mda_text)

    # Sanity check: MD&A should be substantial
    if len(mda_text) < 500:
        logger.debug(f"  MD&A too short: {len(mda_text)} chars")
        return None

    return mda_text


# ---------------------------------------------------------------------------
# 5. Determine the fiscal quarter from the filing date
# ---------------------------------------------------------------------------

def date_to_quarter(filing_date: str) -> str:
    """Convert a filing date to a fiscal quarter label like Q2-2023."""
    dt = datetime.strptime(filing_date, "%Y-%m-%d")
    # 10-Q filing dates are ~1 month after quarter end
    # Q1 filings: ~Apr/May, Q2: ~Jul/Aug, Q3: ~Oct/Nov
    # We estimate the quarter the filing covers (not the filing month)
    month = dt.month
    if month in (1, 2, 3, 4):
        q = 1
    elif month in (5, 6, 7):
        q = 2
    elif month in (8, 9, 10):
        q = 3
    else:
        q = 4
    return f"Q{q}-{dt.year}"


# ---------------------------------------------------------------------------
# 6. Main scraping loop
# ---------------------------------------------------------------------------

def scrape_company(
    ticker: str,
    cik: str,
    company_name: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    debug: bool = False,
) -> list[dict]:
    """Scrape all 10-Q MD&A sections for a single company."""
    logger.info(f"Scraping {ticker} ({company_name}) -- CIK: {cik}")

    filings = get_company_filings(cik, "10-Q")
    filings = filter_filings_by_date(filings, start_date, end_date)
    logger.info(f"  Found {len(filings)} 10-Q filings in date range")

    records = []

    for i, filing in enumerate(filings):
        # Resolve the actual 10-Q document (not cover page)
        url = resolve_10q_document(
            cik, filing["accessionNumber"], filing["primaryDocument"]
        )
        logger.info(f"  [{i+1}/{len(filings)}] {filing['filingDate']} -> {url.split('/')[-1]}")

        html = download_filing(url)
        if html is None:
            continue

        # Debug mode: save raw HTML for inspection
        if debug:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            debug_path = os.path.join(
                DEBUG_DIR, f"{ticker}_{filing['filingDate']}.html"
            )
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"  DEBUG: saved raw HTML to {debug_path}")

        mda_text = extract_mda_section(
            html,
            debug_info=f"{ticker}_{filing['filingDate']}" if debug else "",
        )

        if mda_text is None:
            logger.warning(f"  Could not extract MD&A from {filing['filingDate']}")

            # If primary doc failed, try other large HTML files in the filing
            if not debug:
                logger.info(f"  Trying fallback document resolution...")
                base_url = get_filing_base_url(cik, filing["accessionNumber"])
                fallback_url = try_fallback_documents(base_url, url)
                if fallback_url:
                    html = download_filing(fallback_url)
                    if html:
                        mda_text = extract_mda_section(html)
                        if mda_text:
                            url = fallback_url
                            logger.info(f"  Fallback succeeded: {fallback_url.split('/')[-1]}")

        if mda_text is None:
            continue

        record = {
            "company": ticker,
            "company_name": company_name,
            "cik": cik,
            "date": filing["filingDate"],
            "quarter": date_to_quarter(filing["filingDate"]),
            "form": filing["form"],
            "accession_number": filing["accessionNumber"],
            "filing_url": url,
            "mda_text": mda_text,
            "mda_length": len(mda_text),
        }
        records.append(record)
        logger.info(f"  Extracted MD&A: {len(mda_text):,} chars")

    # Save per-company JSON
    if records:
        company_file = os.path.join(output_dir, f"{ticker}.json")
        with open(company_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info(f"  Saved {len(records)} records to {company_file}")

    return records


def try_fallback_documents(base_url: str, already_tried: str) -> str | None:
    """
    When the primary document doesn't contain MD&A, try other HTML files
    in the filing directory. The actual 10-Q body is usually the largest.
    """
    index_url = f"{base_url}/index.json"
    time.sleep(REQUEST_DELAY)

    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return None

        index_data = resp.json()
        items = index_data.get("directory", {}).get("item", [])

        candidates = []
        for item in items:
            name = item.get("name", "")
            size = int(item.get("size", "0") or "0")
            url = f"{base_url}/{name}"

            if url == already_tried:
                continue
            if not name.lower().endswith((".htm", ".html")):
                continue
            if name.startswith("R") and len(name) < 10:
                continue
            if size < 50000:
                continue

            candidates.append({"url": url, "size": size, "name": name})

        if candidates:
            # Try the largest one we haven't tried yet
            best = max(candidates, key=lambda x: x["size"])
            logger.debug(f"  Fallback candidate: {best['name']} ({best['size']} bytes)")
            return best["url"]

    except (requests.RequestException, json.JSONDecodeError, KeyError):
        pass

    return None


def scrape_all(
    tickers: list[str],
    start_date: str,
    end_date: str,
    output_dir: str,
    debug: bool = False,
) -> list[dict]:
    """
    Main entry point: scrape MD&A sections for all given tickers.
    Saves per-company JSON files and a combined dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    ticker_map = load_company_tickers()

    all_records = []
    skipped = []
    stats = {"total_filings": 0, "extracted": 0, "failed": 0}

    for i, ticker in enumerate(tickers):
        ticker = ticker.upper()
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(tickers)}] Processing {ticker}")
        logger.info(f"{'='*60}")

        if ticker not in ticker_map:
            logger.warning(f"Ticker {ticker} not found in SEC database -- skipping")
            skipped.append(ticker)
            continue

        company_info = ticker_map[ticker]
        records = scrape_company(
            ticker=ticker,
            cik=company_info["cik"],
            company_name=company_info["name"],
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            debug=debug,
        )
        all_records.extend(records)

    # Save combined dataset
    combined_file = os.path.join(output_dir, "_all_transcripts.json")
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SCRAPING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total MD&A sections extracted: {len(all_records)}")
    logger.info(f"Companies scraped: {len(tickers) - len(skipped)}")
    logger.info(f"Companies skipped: {len(skipped)} -- {skipped}")
    logger.info(f"Output: {combined_file}")

    return all_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape earnings-related MD&A sections from SEC EDGAR 10-Q filings"
    )
    parser.add_argument(
        "--tickers", nargs="+",
        help="Specific tickers to scrape (e.g., AAPL MSFT GOOGL)",
    )
    parser.add_argument(
        "--all", action="store_true", dest="use_all",
        help="Scrape all default S&P 500 tickers (~50 companies)",
    )
    parser.add_argument(
        "--start", default="2020-01-01",
        help="Start date for filings (YYYY-MM-DD, default: 2020-01-01)",
    )
    parser.add_argument(
        "--end", default="2025-12-31",
        help="End date for filings (YYYY-MM-DD, default: 2025-12-31)",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save raw HTML files for inspection (to data/raw/debug_html/)",
    )

    args = parser.parse_args()

    if args.use_all:
        tickers = get_default_tickers()
    elif args.tickers:
        tickers = args.tickers
    else:
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        logger.info("No tickers specified -- using default test batch of 5")

    scrape_all(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()