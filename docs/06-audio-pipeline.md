# 06 — Audio pipeline (optional, currently unused)

**Files:** `src/data/audio_downloader.py`, `src/data/audio_features.py`

## The idea

Earnings calls are spoken. Vocal cues — pitch variation, speaking rate, pauses, loudness — have been
studied as stress/deception indicators. The plan was to add them as a second input ("modality")
alongside the filing text.

## Why it's off

The available free audio is from speech-recognition benchmarks (Earnings-21: ~44 calls; Earnings-22:
~125 calls, mostly non-US companies). They were chosen for accent/sector variety, not to match our 49
tickers and 2020–2025 quarters, and identify calls by numeric IDs. Almost none would join to our
labels, so the model would learn nothing from them. `dataset.py` therefore defaults to text-only.

## Downloader

```bash
python src/data/audio_downloader.py --source huggingface --max-calls 5 --output-dir data/raw/audio
```

- Sources: Hugging Face datasets (`Revai/earnings21`, `distil-whisper/earnings21`,
  `argmaxinc/earnings22`, …) and optionally scraping investor-relations pages.
- Converts to 16 kHz mono WAV and records each file in `manifest.csv`.
- **Storage:** without `--max-calls`, roughly 20–35 GB (downloads + WAVs). Note `--max-calls` limits
  files kept, not download size — each HF dataset is downloaded in full first.

## Feature extractor

```bash
python -m src.data.audio_features --audio-dir data/raw/audio \
    --manifest data/raw/audio/manifest.csv --output-path data/processed --n-workers 4
```

Per call, using `librosa`, processed in chunks to limit memory:

| Group | Function | Examples |
|---|---|---|
| Pitch | `_extract_pitch` | mean / std / range of fundamental frequency |
| Speech rate | `_extract_speech_rate` | onsets per second |
| Pauses | `_extract_pauses` | pause count, mean length, silence ratio |
| Energy | `_extract_energy` | RMS loudness mean / variation |
| MFCC | `_extract_mfcc` | timbre summary coefficients |
| Spectral | `_extract_spectral` | centroid, bandwidth, rolloff |

Output: `data/processed/audio_features.csv` and `.npz` (`features`, `tickers`, `dates`), which
`dataset.py` loads when `use_audio=True`, keyed by exact `(ticker, date)`.

## When it would become useful

If you obtain call recordings for our own tickers and quarters (e.g. company IR archives), key them by
ticker and announcement date and enable `use_audio=True`.

## Concepts to learn here

Audio signal basics (sample rate, frames, pitch, MFCCs); multimodal learning; why a modality is only
as useful as its overlap with your labelled data.
