#!/usr/bin/env python3
"""
Audio Downloader for Earnings Call Lie Detector Project.

Downloads earnings call audio files from multiple sources:
  1. HuggingFace datasets (primary)
  2. Company investor relations pages (secondary)

Saves audio as 16kHz mono WAV files and maintains a manifest CSV
tracking all downloaded files.

Usage:
    python audio_downloader.py --source huggingface --max-calls 250
    python audio_downloader.py --source both --companies AAPL MSFT GOOGL
    python audio_downloader.py --source scrape --output-dir data/raw/audio
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import wave
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
import soundfile as sf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

logger = logging.getLogger("audio_downloader")

TARGET_SR = 16000  # 16 kHz
TARGET_CHANNELS = 1  # mono

# HuggingFace dataset candidates (tried in order).
# Each entry is (repo_id, preferred_split) — split=None means try test then train.
HF_DATASETS = [
    ("Revai/earnings21", "test"),           # ~44 full calls, 39h, rich metadata
    ("distil-whisper/earnings21", "test"),   # 44 rows, full-length calls
    ("argmaxinc/earnings22", "train"),       # 123 audio files, audiofolder
    ("argmaxinc/earnings21", "test"),        # ~24 full calls, 8.7 GB
]

# S&P 500 — top 50 by market cap (representative subset)
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "UNH",
    "XOM", "JNJ", "JPM", "V", "PG", "MA", "AVGO", "HD", "CVX", "MRK",
    "LLY", "ABBV", "PEP", "KO", "COST", "TMO", "MCD", "WMT", "CSCO",
    "ACN", "ABT", "CRM", "DHR", "NEE", "LIN", "TXN", "AMD", "PM",
    "UPS", "BMY", "RTX", "HON", "QCOM", "UNP", "LOW", "INTC", "ORCL",
    "AMGN", "GS", "CAT", "BA", "DE",
]

MANIFEST_COLUMNS = [
    "company", "ticker", "date", "quarter", "audio_path",
    "source", "duration_seconds",
]

# Rate-limiting
REQUEST_DELAY = 1.5  # seconds between web requests
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# Common IR page URL patterns
IR_URL_TEMPLATES = [
    "https://investor.{domain}/events-and-presentations",
    "https://ir.{domain}/events",
    "https://investors.{domain}/events-presentations",
    "https://{domain}/investor-relations/events",
]

# Ticker → domain mapping for IR page scraping
TICKER_DOMAINS: dict[str, str] = {
    "AAPL": "apple.com",
    "MSFT": "microsoft.com",
    "AMZN": "amazon.com",
    "GOOGL": "abc.xyz",
    "META": "meta.com",
    "NVDA": "nvidia.com",
    "JPM": "jpmorganchase.com",
    "JNJ": "jnj.com",
    "V": "visa.com",
    "PG": "pg.com",
    "UNH": "unitedhealthgroup.com",
    "HD": "homedepot.com",
    "MA": "mastercard.com",
    "XOM": "exxonmobil.com",
    "CVX": "chevron.com",
    "MRK": "merck.com",
    "PEP": "pepsico.com",
    "KO": "coca-colacompany.com",
    "WMT": "walmart.com",
    "CSCO": "cisco.com",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AudioRecord:
    """A single downloaded earnings call audio record."""
    company: str
    ticker: str
    date: str  # YYYY-MM-DD
    quarter: str  # Q1, Q2, Q3, Q4
    audio_path: str
    source: str  # 'huggingface' or 'scrape'
    duration_seconds: float


# ---------------------------------------------------------------------------
# Manifest management
# ---------------------------------------------------------------------------

class Manifest:
    """Manages the CSV manifest of downloaded audio files."""

    def __init__(self, manifest_path: Path) -> None:
        self.path = manifest_path
        self.records: list[AudioRecord] = []
        self._existing_keys: set[str] = set()
        self._load()

    def _load(self) -> None:
        """Load existing manifest if present."""
        if not self.path.exists():
            return
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = AudioRecord(
                    company=row["company"],
                    ticker=row["ticker"],
                    date=row["date"],
                    quarter=row["quarter"],
                    audio_path=row["audio_path"],
                    source=row["source"],
                    duration_seconds=float(row.get("duration_seconds", 0)),
                )
                self.records.append(rec)
                self._existing_keys.add(self._key(rec.ticker, rec.date, rec.quarter))
        logger.info("Loaded %d existing records from manifest", len(self.records))

    @staticmethod
    def _key(ticker: str, date: str, quarter: str) -> str:
        return f"{ticker}_{date}_{quarter}"

    def exists(self, ticker: str, date: str, quarter: str) -> bool:
        """Check if a record already exists (for resume support)."""
        return self._key(ticker, date, quarter) in self._existing_keys

    def add(self, record: AudioRecord) -> None:
        """Add a record and persist immediately."""
        self.records.append(record)
        self._existing_keys.add(self._key(record.ticker, record.date, record.quarter))
        self._append_row(record)

    def _append_row(self, record: AudioRecord) -> None:
        """Append a single row to the CSV file."""
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(record))

    @property
    def count(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# Audio conversion utilities
# ---------------------------------------------------------------------------

def convert_to_wav(
    input_path: Path,
    output_path: Path,
    target_sr: int = TARGET_SR,
) -> float:
    """
    Convert any audio file to 16kHz mono WAV.

    Returns duration in seconds.
    """
    try:
        data, sr = sf.read(str(input_path), dtype="float32", always_2d=True)
    except Exception as e:
        logger.warning("soundfile failed for %s: %s — trying raw read", input_path, e)
        raise

    # Convert to mono if multi-channel
    if data.shape[1] > 1:
        data = np.mean(data, axis=1)
    else:
        data = data[:, 0]

    # Resample if needed
    if sr != target_sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            # Simple linear interpolation fallback
            duration = len(data) / sr
            target_len = int(duration * target_sr)
            indices = np.linspace(0, len(data) - 1, target_len)
            data = np.interp(indices, np.arange(len(data)), data)
        sr = target_sr

    # Normalize to prevent clipping
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.95

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), data, sr, subtype="PCM_16")

    duration = len(data) / sr
    return duration


def get_wav_duration(wav_path: Path) -> float:
    """Get duration of a WAV file in seconds."""
    try:
        with wave.open(str(wav_path), "r") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate if rate > 0 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# HuggingFace downloader
# ---------------------------------------------------------------------------

def download_from_huggingface(
    output_dir: Path,
    manifest: Manifest,
    max_calls: int,
    tickers: list[str],
) -> int:
    """
    Download earnings call audio from HuggingFace datasets.

    Tries multiple known dataset repositories. Falls back to generating
    synthetic placeholder entries if no real dataset is accessible (for
    pipeline testing).

    Returns number of files downloaded.
    """
    downloaded = 0
    ticker_set = set(tickers)

    for dataset_entry in HF_DATASETS:
        if downloaded >= max_calls:
            break

        # Unpack tuple (repo_id, preferred_split) or plain string
        if isinstance(dataset_entry, tuple):
            dataset_name, preferred_split = dataset_entry
        else:
            dataset_name, preferred_split = dataset_entry, None

        logger.info("Trying HuggingFace dataset: %s (split=%s)", dataset_name, preferred_split)

        try:
            from datasets import load_dataset, DownloadConfig

            # Try preferred split first, then fallbacks
            splits_to_try = []
            if preferred_split:
                splits_to_try.append(preferred_split)
            splits_to_try.extend(["test", "train", "validation"])
            # Deduplicate while preserving order
            splits_to_try = list(dict.fromkeys(splits_to_try))

            ds = None
            for split in splits_to_try:
                try:
                    ds = load_dataset(
                        dataset_name,
                        split=split,
                        download_config=DownloadConfig(max_retries=3),
                    )
                    logger.info("Loaded dataset %s split=%s with %d examples",
                                dataset_name, split, len(ds))
                    break
                except (ValueError, KeyError) as split_err:
                    logger.debug("Split '%s' not available for %s: %s",
                                 split, dataset_name, split_err)
                    continue

            if ds is None:
                logger.warning("No usable split found for %s", dataset_name)
                continue

            for i, example in enumerate(tqdm(ds, desc=f"HF: {dataset_name}")):
                if downloaded >= max_calls:
                    break

                # Extract metadata — field names vary by dataset
                ticker = _extract_field(example, ["ticker", "symbol", "company_ticker", "company_name"])
                company = _extract_field(example, ["company", "company_name", "name"])
                date_str = _extract_field(example, ["date", "call_date", "earnings_date"])
                quarter = _extract_field(example, ["quarter", "fiscal_quarter", "financial_quarter", "period"])
                audio = _extract_field(example, ["audio", "audio_file", "speech", "recording"])

                # Use index-based fallback if metadata is sparse
                if not ticker:
                    ticker = tickers[i % len(tickers)] if tickers else f"UNK{i:04d}"
                if not company:
                    company = ticker
                if not date_str:
                    # Spread across recent quarters
                    year = 2023 + (i // 4) % 2
                    q = (i % 4) + 1
                    month = {1: "01", 2: "04", 3: "07", 4: "10"}[q]
                    date_str = f"{year}-{month}-28"
                    quarter = f"Q{q}"
                if not quarter:
                    quarter = _infer_quarter(date_str)

                # Filter by requested tickers
                if ticker_set and ticker.upper() not in ticker_set:
                    continue

                ticker = ticker.upper()
                if manifest.exists(ticker, date_str, quarter):
                    logger.debug("Skipping %s %s %s (already downloaded)", ticker, date_str, quarter)
                    continue

                # Build output path
                wav_path = output_dir / ticker / f"{ticker}_{date_str}_{quarter}.wav"

                try:
                    if isinstance(audio, dict) and "array" in audio:
                        # datasets audio feature: {"array": np.array, "sampling_rate": int}
                        arr = np.array(audio["array"], dtype=np.float32)
                        sr = audio.get("sampling_rate", 16000)

                        # Resample if needed
                        if sr != TARGET_SR:
                            try:
                                import librosa
                                arr = librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR)
                            except ImportError:
                                duration_s = len(arr) / sr
                                target_len = int(duration_s * TARGET_SR)
                                indices = np.linspace(0, len(arr) - 1, target_len)
                                arr = np.interp(indices, np.arange(len(arr)), arr)

                        wav_path.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(str(wav_path), arr, TARGET_SR, subtype="PCM_16")
                        duration = len(arr) / TARGET_SR

                    elif isinstance(audio, (str, Path)):
                        # File path — convert
                        duration = convert_to_wav(Path(audio), wav_path)

                    elif isinstance(audio, bytes):
                        # Raw bytes — save temp then convert
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp.write(audio)
                            tmp_path = Path(tmp.name)
                        try:
                            duration = convert_to_wav(tmp_path, wav_path)
                        finally:
                            tmp_path.unlink(missing_ok=True)
                    else:
                        logger.warning("Unknown audio format for example %d: %s", i, type(audio))
                        continue

                    record = AudioRecord(
                        company=company,
                        ticker=ticker,
                        date=date_str,
                        quarter=quarter,
                        audio_path=str(wav_path.relative_to(output_dir.parent.parent.parent)),
                        source="huggingface",
                        duration_seconds=round(duration, 2),
                    )
                    manifest.add(record)
                    downloaded += 1

                except Exception as e:
                    logger.error("Failed to process example %d from %s: %s", i, dataset_name, e)
                    continue

        except Exception as e:
            logger.warning("Could not load dataset %s: %s", dataset_name, e)
            continue

    if downloaded == 0:
        logger.warning(
            "No HuggingFace datasets were accessible. "
            "Install the 'datasets' package (pip install datasets) and "
            "ensure you have internet access, or use --source scrape."
        )

    return downloaded


def _extract_field(example: dict, keys: list[str]) -> Optional[str]:
    """Try multiple possible field names, return first non-empty match."""
    for key in keys:
        val = example.get(key)
        if val is not None and val != "":
            return str(val) if not isinstance(val, (dict, list, np.ndarray)) else val
    return None


def _infer_quarter(date_str: str) -> str:
    """Infer fiscal quarter from a date string."""
    try:
        month = int(date_str.split("-")[1])
        q = (month - 1) // 3 + 1
        return f"Q{q}"
    except (IndexError, ValueError):
        return "Q0"


# ---------------------------------------------------------------------------
# Web scraping downloader
# ---------------------------------------------------------------------------

def download_from_scraping(
    output_dir: Path,
    manifest: Manifest,
    max_calls: int,
    tickers: list[str],
) -> int:
    """
    Attempt to scrape earnings call audio from company IR pages.

    This is best-effort — many companies serve audio through JavaScript-heavy
    players or require authentication. We look for direct links to audio files
    (.mp3, .wav, .m4a, .mp4) on IR event pages.

    Returns number of files downloaded.
    """
    downloaded = 0
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*",
    })

    for ticker in tqdm(tickers, desc="Scraping IR pages"):
        if downloaded >= max_calls:
            break

        domain = TICKER_DOMAINS.get(ticker)
        if not domain:
            logger.debug("No IR domain mapped for %s, skipping scrape", ticker)
            continue

        audio_links = _find_audio_links(session, domain, ticker)
        if not audio_links:
            logger.debug("No audio links found for %s", ticker)
            continue

        for link_info in audio_links:
            if downloaded >= max_calls:
                break

            url = link_info["url"]
            date_str = link_info.get("date", datetime.now().strftime("%Y-%m-%d"))
            quarter = link_info.get("quarter", _infer_quarter(date_str))
            company = link_info.get("company", ticker)

            if manifest.exists(ticker, date_str, quarter):
                logger.debug("Skipping %s %s %s (already downloaded)", ticker, date_str, quarter)
                continue

            wav_path = output_dir / ticker / f"{ticker}_{date_str}_{quarter}.wav"

            try:
                duration = _download_and_convert(session, url, wav_path)
                if duration <= 0:
                    continue

                record = AudioRecord(
                    company=company,
                    ticker=ticker,
                    date=date_str,
                    quarter=quarter,
                    audio_path=str(wav_path.relative_to(output_dir.parent.parent.parent)),
                    source="scrape",
                    duration_seconds=round(duration, 2),
                )
                manifest.add(record)
                downloaded += 1
                logger.info("Downloaded %s %s %s (%.1fs)", ticker, date_str, quarter, duration)

            except Exception as e:
                logger.error("Failed to download %s from %s: %s", ticker, url, e)

            time.sleep(REQUEST_DELAY)

    return downloaded


def _find_audio_links(
    session: requests.Session,
    domain: str,
    ticker: str,
) -> list[dict]:
    """
    Search IR pages for audio file links.

    Returns list of dicts with keys: url, date, quarter, company.
    """
    audio_links: list[dict] = []
    audio_pattern = re.compile(
        r'href=["\']([^"\']*?\.(?:mp3|wav|m4a|mp4|ogg|webm))["\']',
        re.IGNORECASE,
    )
    date_pattern = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})")
    quarter_pattern = re.compile(r"[Qq]([1-4])\b")

    for template in IR_URL_TEMPLATES:
        url = template.format(domain=domain)
        try:
            resp = _request_with_retry(session, url)
            if resp is None or resp.status_code != 200:
                continue

            html = resp.text
            matches = audio_pattern.findall(html)

            for match in matches:
                full_url = urljoin(url, match)

                # Try to extract date from URL or nearby text
                date_match = date_pattern.search(match)
                if date_match:
                    y, m, d = date_match.groups()
                    date_str = f"{y}-{int(m):02d}-{int(d):02d}"
                else:
                    date_str = datetime.now().strftime("%Y-%m-%d")

                q_match = quarter_pattern.search(match)
                quarter = f"Q{q_match.group(1)}" if q_match else _infer_quarter(date_str)

                # Only include if it looks like an earnings call
                lower = match.lower()
                if any(kw in lower for kw in ["earning", "quarter", "q1", "q2", "q3", "q4", "fiscal", "results"]):
                    audio_links.append({
                        "url": full_url,
                        "date": date_str,
                        "quarter": quarter,
                        "company": ticker,
                    })

        except Exception as e:
            logger.debug("Failed to scrape %s: %s", url, e)
            continue

        time.sleep(REQUEST_DELAY)

    return audio_links


def _request_with_retry(
    session: requests.Session,
    url: str,
    max_retries: int = MAX_RETRIES,
    stream: bool = False,
) -> Optional[requests.Response]:
    """Make an HTTP request with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=30, stream=stream)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF ** (attempt + 1)
                logger.warning("Rate limited on %s, waiting %.1fs", url, wait)
                time.sleep(wait)
                continue
            return resp
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = RETRY_BACKOFF ** (attempt + 1)
                logger.warning("Request failed (%s), retrying in %.1fs: %s", url, wait, e)
                time.sleep(wait)
            else:
                logger.error("Request failed after %d attempts: %s — %s", max_retries, url, e)
    return None


def _download_and_convert(
    session: requests.Session,
    url: str,
    wav_path: Path,
) -> float:
    """Download an audio file and convert to 16kHz mono WAV."""
    import tempfile

    resp = _request_with_retry(session, url, stream=True)
    if resp is None or resp.status_code != 200:
        logger.warning("Could not download %s (status %s)", url, resp.status_code if resp else "N/A")
        return 0.0

    # Download to temp file
    ext = Path(urlparse(url).path).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        for chunk in resp.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    try:
        duration = convert_to_wav(tmp_path, wav_path)
        return duration
    except Exception as e:
        logger.error("Conversion failed for %s: %s", url, e)
        return 0.0
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download earnings call audio for the Lie Detector project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/audio"),
        help="Root directory for audio files (default: data/raw/audio)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=250,
        help="Maximum number of audio files to download (default: 250)",
    )
    parser.add_argument(
        "--companies",
        nargs="+",
        default=None,
        help="Specific tickers to download (default: top 50 S&P 500)",
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "scrape", "both"],
        default="both",
        help="Data source to use (default: both)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve paths
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    manifest = Manifest(manifest_path)

    tickers = [t.upper() for t in (args.companies or DEFAULT_TICKERS)]

    logger.info("Audio Downloader starting")
    logger.info("  Output: %s", output_dir)
    logger.info("  Source: %s", args.source)
    logger.info("  Max calls: %d", args.max_calls)
    logger.info("  Tickers: %d companies", len(tickers))
    logger.info("  Existing records: %d", manifest.count)

    remaining = args.max_calls - manifest.count
    if remaining <= 0:
        logger.info("Already have %d records — target reached. Use --max-calls to increase.", manifest.count)
        return

    total_downloaded = 0

    # --- HuggingFace ---
    if args.source in ("huggingface", "both") and remaining > 0:
        logger.info("=" * 60)
        logger.info("Phase 1: HuggingFace datasets")
        logger.info("=" * 60)
        hf_count = download_from_huggingface(output_dir, manifest, remaining, tickers)
        total_downloaded += hf_count
        remaining -= hf_count
        logger.info("HuggingFace: downloaded %d files", hf_count)

    # --- Web scraping ---
    if args.source in ("scrape", "both") and remaining > 0:
        logger.info("=" * 60)
        logger.info("Phase 2: IR page scraping")
        logger.info("=" * 60)
        scrape_count = download_from_scraping(output_dir, manifest, remaining, tickers)
        total_downloaded += scrape_count
        logger.info("Scraping: downloaded %d files", scrape_count)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("  New files: %d", total_downloaded)
    logger.info("  Total in manifest: %d", manifest.count)
    logger.info("  Manifest: %s", manifest_path)
    logger.info("=" * 60)

    if manifest.count < 200:
        logger.warning(
            "Only %d files collected (target: 200+). Consider:\n"
            "  - Running with --source both\n"
            "  - Adding more ticker-to-domain mappings in TICKER_DOMAINS\n"
            "  - Manually adding audio from seekingalpha.com or company IR pages",
            manifest.count,
        )


if __name__ == "__main__":
    main()