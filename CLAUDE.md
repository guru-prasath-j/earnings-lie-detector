# Earnings Lie Detector — project context

Predict whether a company will **miss its next earnings EPS estimate** (label 1) or meet/beat it (label 0)
from the MD&A section of its SEC 10-Q filings. Text-only for now; audio is optional and off.

## Keep this file current (standing instruction)

This file is the hand-off between machines and sessions. After any meaningful change — new/changed
scripts, a pipeline run, training results, a design decision, a new known issue — update the relevant
section below and the **Log** at the bottom (date, what changed, results). Keep it short and factual;
replace stale info rather than appending contradictions. Commit it together with the code it describes.

Learning docs live in `docs/` (`00-overview.md` = high-level, `01`–`06` = one per component). The user
reads them to learn — when an implementation or result changes, update the matching doc too.

## Setup on a new machine

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`data/raw/`, `data/processed/`, `*.pt` and `checkpoints/` are gitignored — regenerate them:

```bash
python src/data/scraper.py                                         # → data/raw/transcripts/{TICKER}.json (~664 filings, 49 tickers)
python src/data/labeler.py --transcripts-dir data/raw/transcripts  # → data/processed/labels.csv (~1,899 rows)
python src/data/dataset.py                                         # sanity check: 661 samples, 92 misses
python src/models/train.py --data-root . --freeze-encoder --epochs 10 --batch-size 8   # 5 seeds, ~7 min on RTX A1000
python src/models/baselines.py --data-root .                                          # history / TF-IDF yardsticks, ~30 s
```

## Pipeline

| File | Role |
|---|---|
| `src/data/scraper.py` | SEC EDGAR 10-Q MD&A text → flat `data/raw/transcripts/{TICKER}.json` (list of records: `company`, `date`=filing date, `quarter`, `form`, `mda_text`, …) plus combined `_all_transcripts.json` |
| `src/data/labeler.py` | Default `--method direct`: yfinance `get_earnings_dates` (history back to ~2014, `lookback_quarters=40`) → one row per earnings announcement; `label = 1 if eps_actual < eps_estimate`. `date` = announcement date. Reads both `{T}.json` and `{T}/*.json` layouts, skips `_*` files |
| `src/data/dataset.py` | `EarningsCallDataset`. Pairs each label (announcement date) with the latest filing **strictly before** it within 120 days (`max_transcript_gap_days`). `use_audio=False` by default → text-only, drops labels with no filing, batches have no audio keys. `get_dataloaders(split="temporal")` default: train < 2024-01-01 (370, 48 miss), val 2024 (122, 21), test ≥ 2025-01-01 (169, 23); `split="random"` still available |
| `src/data/audio_downloader.py`, `audio_features.py` | Optional audio pipeline (Earnings-21/22 HF datasets → ~20–28 librosa features). Not run here; won't join to labels (benchmark calls ≠ our tickers/dates). Downloads ~20–35 GB if run without `--max-calls` |
| `src/models/model.py` | `EarningsTextClassifier`: `ProsusAI/finbert` (fallback `distilbert-base-uncased`) → [CLS] → Dropout(0.3) → Linear(768→128) → ReLU → Dropout(0.15) → Linear(128→2). `freeze_encoder` / `freeze_n_layers` |
| `src/models/train.py` | Class-weighted CE (weights from train split), AdamW (head LR = 10× encoder LR), warmup→cosine, AMP on CUDA, early stop on val F1 (patience 3). `--seeds` (default 42–46) → `checkpoints/seed_N/{best_model.pt,training_history.json}` + `checkpoints/summary.json` (mean ± std). `--split temporal|random`, `--val-start`, `--test-start` |
| `src/models/baselines.py` | Logistic-regression yardsticks on the same samples/split: `history` (prior-quarter miss/surprise, 4q rolling stats, `shift(1)` so no leakage), `tfidf` (full MD&A text), `combined`. C picked by val AUC → `checkpoints/baselines.json` |
| `docs/` | Learning docs: `00-overview.md` (high-level) + one per component |
| `notebooks/03_eda.ipynb` | EDA (label distribution, feature tests, correlations, temporal) |

## Key decisions

- **Forward-looking pairing**: a 10-Q is filed right after its own quarter's earnings announcement and already
  states that EPS, so it is paired with the *next* announcement (gap 63–120 days, median 90). Do not revert.
- **Class imbalance**: ~14% misses. Use `ds.get_class_weights()`; judge on F1/AUC, not accuracy.
- **Truncation**: MD&A is 5k–20k tokens; model sees the first 512 only.

## Current results

Temporal split, test = 2025–2026 (169 samples, 23 misses), 2026-10-22:

| Model | Test AUC | Test F1 |
|---|---|---|
| FinBERT frozen, 512 tokens (5 seeds) | 0.515 ± 0.029 | 0.119 ± 0.051 |
| TF-IDF full text (LR, C=10) | 0.672 | 0.227 |
| History only (LR, C=0.01) | **0.714** | **0.400** |
| History + TF-IDF | 0.724 | 0.400 |

Takeaways: text has signal (TF-IDF ≫ 0.5) but FinBERT-512-frozen discards it; company history is the
strongest predictor; text model must beat AUC ≈ 0.71 to matter. (Random split had shown FinBERT test
AUC 0.646 — optimistic.)

## Open issues / next steps

1. 512-token truncation sees ~3% of the MD&A (mostly boilerplate) — implement chunk/sliding-window pooling over the whole filing; try skipping the opening boilerplate.
2. Try partial (`--freeze-n-layers 8`) and full fine-tune (`--lr 2e-5 --epochs 15`) once full-text input works.
3. Check whether TF-IDF signal is just company identity (add per-company feature / company-held-out eval).
4. More/better text: only 661 samples. 8-K earnings press releases (Ex. 99.1) or call transcripts.

## Reference

- Repo: https://github.com/guru-prasath-j/earnings-lie-detector (branch `master`). On the Linux box
  (Meril1) this repo pushes with a repo-local credential helper reading `~/personal/.secrets/github.env`
  (account guru-prasath-j); the global `~/.git-credentials` login (guruprasath37) can't push here.
- Full project plan: https://claude.ai/artifact/9sTZ183A6ZcowvVVU6n6fX

## Log

- **2026-10-22** — Fixed scraper `size` parsing; added `yfinance` to requirements. Labeler: supports flat
  `{TICKER}.json`, uses `get_earnings_dates` first (history to ~2014), lookback 40 → 1,899 labels (258 misses).
  Dataset: forward-looking filing→next-announcement pairing, flat-file loading, text-only default (`use_audio`).
  Added `src/models/model.py` + `train.py`; frozen-encoder baseline test F1 0.20 / AUC 0.65 (random split).
- **2026-10-22** — Temporal split (default) in `get_dataloaders`; `train.py` multi-seed with summary.json and
  train-only class weights; added `baselines.py`. Results table above: FinBERT at chance on temporal split,
  history-only best (AUC 0.714). Wrote `docs/` (overview + 6 component docs).
- **2026-10-22** — Pushed to GitHub (guru-prasath-j/earnings-lie-detector, `master`).
