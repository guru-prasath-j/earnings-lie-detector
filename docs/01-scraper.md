# 01 — Scraper: getting MD&A text from SEC EDGAR

**File:** `src/data/scraper.py` · **Output:** `data/raw/transcripts/{TICKER}.json`

## What it does

Downloads each company's 10-Q filings from the SEC's free EDGAR system and cuts out the MD&A section.

```bash
python src/data/scraper.py --all --start 2020-01-01 --end 2025-12-31
python src/data/scraper.py --tickers AAPL MSFT
python src/data/scraper.py --tickers AAPL --debug     # also saves raw HTML for inspection
```

The default list is ~50 large S&P 500 companies across sectors (`get_default_tickers`). 49 produced
data, 664 filings in total.

## How it works, step by step

1. **Ticker → CIK** (`load_company_tickers`). EDGAR identifies companies by a numeric CIK, not a
   ticker. SEC publishes a mapping file (`company_tickers.json`).
2. **List filings** (`get_company_filings`). `data.sec.gov/submissions/CIK##########.json` lists every
   filing; we keep form `10-Q` within the date range (`filter_filings_by_date`).
3. **Find the real document** (`resolve_10q_document`). The "primary document" sometimes points at a
   cover page or an XBRL viewer, so the scraper inspects the filing index and picks the actual report.
   `try_fallback_documents` tries alternatives if extraction fails.
4. **Clean the HTML** (`clean_ixbrl`, `normalize_text`). Modern filings are *inline XBRL*: HTML
   sprinkled with `<ix:...>` tags that tag numbers for machines. These are stripped, and curly quotes,
   em-dashes and non-breaking spaces are normalised so text patterns match.
5. **Cut out MD&A** (`extract_mda_section`). Regexes find the start ("Item 2. Management's Discussion
   and Analysis…") and the end (the next item, e.g. "Item 3. Quantitative and Qualitative
   Disclosures…"). Results under 500 characters are rejected as failed extractions.
6. **Save** one JSON list per ticker plus a combined `_all_transcripts.json`.

## Record format

```json
{
  "company": "AAPL", "company_name": "Apple Inc.", "cik": "0000320193",
  "date": "2025-08-01",            // filing date — when the text became public
  "quarter": "Q3-2025",            // rough guess from the filing month (see below)
  "form": "10-Q", "accession_number": "...", "filing_url": "...",
  "mda_text": "Item 2. Management's Discussion and Analysis ...",
  "mda_length": "19903"
}
```

## Things worth knowing

- **SEC etiquette.** SEC requires a `User-Agent` identifying you and limits request rate;
  `REQUEST_DELAY = 0.12 s` keeps us under ~10 requests/second.
- **`quarter` is approximate.** `date_to_quarter` guesses from the filing month and ignores fiscal
  years (Apple's fiscal Q3 ends in June). Downstream code uses `date` instead, which is exact.
- **Why there are only ~13–14 filings per company** for 6 years: companies file a 10-K (annual report),
  not a 10-Q, for their fourth quarter, and some extractions fail.
- **Bug fixed:** `int(item.get("size", "0"))` crashed when the index listed an empty size; it is now
  `int(item.get("size", "0") or "0")`.

## Concepts to learn here

Web scraping with `requests` + `BeautifulSoup`; regular expressions for semi-structured documents;
why real-world text needs Unicode normalisation; rate limiting against public APIs.
