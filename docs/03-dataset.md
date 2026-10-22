# 03 — Dataset: pairing text with labels, without cheating

**File:** `src/data/dataset.py` · **Class:** `EarningsCallDataset` · **Helper:** `get_dataloaders`

## What it does

Builds the list of training samples: each one is *(MD&A text, label)*. It's a PyTorch `Dataset`, so
a `DataLoader` can batch it for training.

```bash
python src/data/dataset.py               # sanity check, text-only (default)
python src/data/dataset.py --use-audio   # include audio features
```

```
EarningsCallDataset — 661 samples
  Labels:  met=0 → 569  |  missed=1 → 92  (13.9% positive)
  Transcripts available: 661/661
```

## The key decision: pair each filing with the *next* announcement

Timeline for Apple:

```
2025-07-31  Q3 results announced (EPS 1.57 vs 1.43 est → beat)
2025-08-01  10-Q filed  ── its MD&A already discusses the Q3 results
2025-10-30  Q4 results announced (EPS 1.85 vs 1.77 est → beat)   ◀── paired label
```

If we labelled the 2025-08-01 filing with the 07-31 result, the text would literally contain the
answer — the model would learn to read reported numbers, not to predict. This is **data leakage**.

So `_match_transcript` does the reverse lookup: for every announcement, take the company's **latest
filing strictly before it**, as long as it's within `max_transcript_gap_days` (120). In practice gaps
are 63–120 days (median 90) and every filing is used at most once — 661 of 664 filings get a label.

## Loading transcripts

`_load_transcripts` reads both `{TICKER}.json` (a list of records) and `{TICKER}/*.json`, takes
`text`/`transcript`/`mda_text`, and keys each by `(ticker, filing date)`.

## Text-only by default (`use_audio=False`)

- Audio files aren't loaded.
- Labels with no matching filing are dropped (1,899 → 661); a text model can't use an empty input.
- Batches contain `text`, `label`, `ticker`, `date`, `has_transcript` — no audio keys.

With `use_audio=True` the original behaviour returns: all 1,899 labels, plus `audio_features`
(zeros when missing) and `has_audio_features`.

## Splitting: `get_dataloaders(split="temporal")`

| Part | Announcement dates | Samples | Misses |
|---|---|---|---|
| train | before 2024-01-01 | 370 | 48 |
| val | 2024 | 122 | 21 |
| test | 2025-01-01 on | 169 | 23 |

Why by time: in real use you train on the past and predict the future. A random split
(`split="random"`, still available) mixes years, so the model can learn e.g. "2022 was a bad year for
banks" from training rows and apply it to test rows from the same period — an optimistic estimate.
Cut-offs are adjustable (`val_start`, `test_start`).

## Class weights

`get_class_weights()` returns inverse-frequency weights (≈ 0.58 for beat, 3.6 for miss), so each
miss counts ~6× more in the loss. `train.py` recomputes them from the **training split only**.

## Concepts to learn here

PyTorch `Dataset`/`DataLoader`/`collate_fn`; temporal joins ("as-of" joins); leakage and why it is
the #1 way ML projects fool themselves; temporal vs random validation; class imbalance.
