#!/usr/bin/env python3
"""Audio feature extraction pipeline for the Earnings Call Lie Detector.

Extracts vocal stress indicators from earnings call audio using librosa.
Features capture pitch dynamics, speech rate, pause patterns, energy
profiles, MFCCs, and spectral characteristics — all shown to correlate
with deceptive speech in the psycholinguistics literature.

Output: ~20-dimensional feature vector per call, saved as both CSV
(for exploration) and .npz (for direct ML loading).

Usage:
    python -m src.data.audio_features \\
        --audio-dir data/raw/audio \\
        --manifest data/raw/audio/manifest.csv \\
        --output-path data/processed \\
        --n-workers 4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000          # target sample rate for all audio
MIN_AUDIO_SECONDS = 30        # skip files shorter than this
PAUSE_THRESHOLD_DB = -40      # dB below peak to count as silence
PAUSE_MIN_DURATION = 0.5      # seconds — minimum gap to count as a pause
CHUNK_SECONDS = 300           # process audio in 5-minute chunks to cap RAM
HOP_LENGTH = 512              # librosa hop length (32 ms at 16 kHz)
N_MFCC = 13                   # first N MFCC coefficients


# ---------------------------------------------------------------------------
# Feature container
# ---------------------------------------------------------------------------

@dataclass
class AudioFeatures:
    """One row of extracted features for a single earnings call."""

    # metadata (filled by caller, not extraction)
    ticker: str = ""
    date: str = ""
    quarter: str = ""
    audio_file: str = ""

    # pitch / F0
    pitch_mean: float = np.nan
    pitch_std: float = np.nan
    pitch_range: float = np.nan
    pitch_slope: float = np.nan          # linear trend over time

    # speech rate
    speech_rate: float = np.nan          # onsets per second

    # pauses
    pause_count: float = np.nan
    pause_mean_duration: float = np.nan
    pause_max_duration: float = np.nan
    pause_rate: float = np.nan           # pauses per minute

    # energy / RMS
    energy_mean: float = np.nan
    energy_std: float = np.nan
    energy_slope: float = np.nan

    # MFCCs (mean of first 13 coefficients)
    mfcc_01: float = np.nan
    mfcc_02: float = np.nan
    mfcc_03: float = np.nan
    mfcc_04: float = np.nan
    mfcc_05: float = np.nan
    mfcc_06: float = np.nan
    mfcc_07: float = np.nan
    mfcc_08: float = np.nan
    mfcc_09: float = np.nan
    mfcc_10: float = np.nan
    mfcc_11: float = np.nan
    mfcc_12: float = np.nan
    mfcc_13: float = np.nan

    # spectral
    spectral_centroid_mean: float = np.nan
    spectral_rolloff_mean: float = np.nan
    zero_crossing_rate: float = np.nan

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def feature_columns(cls) -> list[str]:
        """Return only the numeric feature column names (no metadata)."""
        meta = {"ticker", "date", "quarter", "audio_file"}
        return [f.name for f in fields(cls) if f.name not in meta]


# ---------------------------------------------------------------------------
# Low-level feature extractors
# ---------------------------------------------------------------------------

def _extract_pitch(y: np.ndarray, sr: int) -> dict:
    """Pitch (F0) statistics via pyin."""
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=sr, hop_length=HOP_LENGTH,
    )
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
    if len(f0_voiced) < 5:
        return dict(pitch_mean=np.nan, pitch_std=np.nan,
                    pitch_range=np.nan, pitch_slope=np.nan)

    # linear trend: slope of F0 over voiced frames
    x = np.arange(len(f0_voiced), dtype=np.float64)
    slope = float(np.polyfit(x, f0_voiced, 1)[0])

    return dict(
        pitch_mean=float(np.mean(f0_voiced)),
        pitch_std=float(np.std(f0_voiced)),
        pitch_range=float(np.ptp(f0_voiced)),
        pitch_slope=slope,
    )


def _extract_speech_rate(y: np.ndarray, sr: int, duration: float) -> dict:
    """Approximate syllable / speech rate using onset detection."""
    onsets = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=HOP_LENGTH, backtrack=False,
    )
    rate = len(onsets) / duration if duration > 0 else 0.0
    return dict(speech_rate=float(rate))


def _extract_pauses(y: np.ndarray, sr: int, duration: float) -> dict:
    """Detect silent gaps and compute pause statistics."""
    # frame-level power in dB
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms) + 1e-10)

    is_silent = rms_db < PAUSE_THRESHOLD_DB
    frame_dur = HOP_LENGTH / sr  # duration of one frame in seconds

    # run-length encoding to find silent stretches
    pauses: list[float] = []
    current_run = 0
    for silent in is_silent:
        if silent:
            current_run += 1
        else:
            if current_run > 0:
                gap = current_run * frame_dur
                if gap >= PAUSE_MIN_DURATION:
                    pauses.append(gap)
            current_run = 0
    # flush last run
    if current_run > 0:
        gap = current_run * frame_dur
        if gap >= PAUSE_MIN_DURATION:
            pauses.append(gap)

    if not pauses:
        return dict(pause_count=0, pause_mean_duration=0.0,
                    pause_max_duration=0.0, pause_rate=0.0)

    minutes = duration / 60.0 if duration > 0 else 1.0
    return dict(
        pause_count=len(pauses),
        pause_mean_duration=float(np.mean(pauses)),
        pause_max_duration=float(np.max(pauses)),
        pause_rate=len(pauses) / minutes,
    )


def _extract_energy(y: np.ndarray, sr: int) -> dict:
    """RMS energy statistics and trend."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    if len(rms) < 2:
        return dict(energy_mean=np.nan, energy_std=np.nan, energy_slope=np.nan)

    x = np.arange(len(rms), dtype=np.float64)
    slope = float(np.polyfit(x, rms, 1)[0])
    return dict(
        energy_mean=float(np.mean(rms)),
        energy_std=float(np.std(rms)),
        energy_slope=slope,
    )


def _extract_mfcc(y: np.ndarray, sr: int) -> dict:
    """Mean of first N_MFCC coefficients across time."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    means = np.mean(mfcc, axis=1)
    return {f"mfcc_{i+1:02d}": float(means[i]) for i in range(N_MFCC)}


def _extract_spectral(y: np.ndarray, sr: int) -> dict:
    """Spectral centroid, rolloff, and zero-crossing rate."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
    return dict(
        spectral_centroid_mean=float(np.mean(centroid)),
        spectral_rolloff_mean=float(np.mean(rolloff)),
        zero_crossing_rate=float(np.mean(zcr)),
    )


# ---------------------------------------------------------------------------
# Main extraction — per-file
# ---------------------------------------------------------------------------

def extract_features_from_file(audio_path: str | Path) -> dict:
    """Load one audio file, extract the full feature vector.

    Long files are loaded in chunks to limit peak memory usage, then
    features are computed on the concatenated waveform.

    Returns a dict of feature-name → value (no metadata).
    """
    audio_path = Path(audio_path)

    # Load full waveform — chunked reading via offset/duration to cap RAM
    duration_total = librosa.get_duration(path=audio_path)
    if duration_total < MIN_AUDIO_SECONDS:
        raise ValueError(
            f"Audio too short ({duration_total:.1f}s < {MIN_AUDIO_SECONDS}s): "
            f"{audio_path.name}"
        )

    chunks: list[np.ndarray] = []
    offset = 0.0
    while offset < duration_total:
        chunk_dur = min(CHUNK_SECONDS, duration_total - offset)
        y_chunk, _ = librosa.load(
            audio_path, sr=SAMPLE_RATE, offset=offset, duration=chunk_dur,
            mono=True,
        )
        chunks.append(y_chunk)
        offset += chunk_dur

    y = np.concatenate(chunks)
    sr = SAMPLE_RATE

    # Extract all feature groups
    feats: dict = {}
    feats.update(_extract_pitch(y, sr))
    feats.update(_extract_speech_rate(y, sr, duration_total))
    feats.update(_extract_pauses(y, sr, duration_total))
    feats.update(_extract_energy(y, sr))
    feats.update(_extract_mfcc(y, sr))
    feats.update(_extract_spectral(y, sr))

    return feats


def _process_row(row: dict, audio_dir: Path) -> Optional[dict]:
    """Worker function: extract features for one manifest row.

    Expects row to have keys: ticker, date, quarter, audio_path.
    Returns a full feature dict with metadata, or None on failure.
    """
    ticker = row["ticker"]
    raw_path = row["audio_path"]
    # audio_path in manifest is relative (e.g. data/raw/audio/AAPL/AAPL_2023-04-28_Q2.wav)
    # Try as-is first, then relative to audio_dir
    audio_path = Path(raw_path)
    if not audio_path.exists():
        audio_path = audio_dir / Path(raw_path).name
    if not audio_path.exists():
        audio_path = audio_dir / ticker / Path(raw_path).name

    if not audio_path.exists():
        logger.warning("Audio file not found: %s", audio_path)
        return None

    try:
        feats = extract_features_from_file(audio_path)
    except Exception as exc:
        logger.error("Failed to extract features from %s: %s", audio_path, exc)
        return None

    feats["ticker"] = ticker
    feats["date"] = row.get("date", "")
    feats["quarter"] = row.get("quarter", "")
    feats["audio_file"] = str(audio_path)
    return feats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_extraction(
    audio_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    n_workers: int = 1,
) -> pd.DataFrame:
    """Run the full extraction pipeline over every file in the manifest.

    Args:
        audio_dir: Root dir containing {ticker}/ sub-folders with WAV files.
        manifest_path: CSV with columns: ticker, date, quarter, audio_path.
        output_dir: Where to write audio_features.csv and .npz.
        n_workers: Number of parallel workers (1 = sequential).

    Returns:
        DataFrame of extracted features.
    """
    manifest = pd.read_csv(manifest_path)
    required = {"ticker", "date", "quarter", "audio_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    rows = manifest.to_dict("records")
    results: list[dict] = []

    if n_workers <= 1:
        # Sequential — simpler debugging
        for row in tqdm(rows, desc="Extracting audio features"):
            result = _process_row(row, audio_dir)
            if result is not None:
                results.append(result)
    else:
        # Parallel with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_row, row, audio_dir): row
                for row in rows
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Extracting audio features",
            ):
                result = future.result()
                if result is not None:
                    results.append(result)

    if not results:
        logger.error("No features extracted — check manifest and audio files.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Reorder: metadata first, then features
    meta_cols = ["ticker", "date", "quarter", "audio_file"]
    feat_cols = AudioFeatures.feature_columns()
    col_order = meta_cols + [c for c in feat_cols if c in df.columns]
    df = df[col_order]

    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "audio_features.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved CSV: %s (%d rows, %d features)", csv_path, len(df), len(feat_cols))

    # .npz: numeric features only, plus metadata arrays for alignment
    numeric = df[feat_cols].values.astype(np.float32)
    npz_path = output_dir / "audio_features.npz"
    np.savez_compressed(
        npz_path,
        features=numeric,
        feature_names=np.array(feat_cols),
        tickers=df["ticker"].values,
        dates=df["date"].values,
        quarters=df["quarter"].values,
    )
    logger.info("Saved NPZ: %s", npz_path)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract audio features from earnings call recordings.",
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=Path("data/raw/audio"),
        help="Root directory containing {ticker}/ sub-folders with WAV files.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/raw/audio/manifest.csv"),
        help="CSV manifest with columns: ticker, date, quarter, filename.",
    )
    parser.add_argument(
        "--output-path", type=Path, default=Path("data/processed"),
        help="Output directory for audio_features.csv and .npz.",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel workers (default: 1 = sequential).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Audio feature extraction starting")
    logger.info("  audio_dir : %s", args.audio_dir)
    logger.info("  manifest  : %s", args.manifest)
    logger.info("  output    : %s", args.output_path)
    logger.info("  workers   : %d", args.n_workers)

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest)
        sys.exit(1)

    df = run_extraction(
        audio_dir=args.audio_dir,
        manifest_path=args.manifest,
        output_dir=args.output_path,
        n_workers=args.n_workers,
    )

    if df.empty:
        logger.error("Pipeline produced no output.")
        sys.exit(1)

    logger.info("Done — %d calls processed successfully.", len(df))


if __name__ == "__main__":
    main()
