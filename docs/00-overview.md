# Earnings Lie Detector — Overview

> Start here. This page explains *what* we are building and *why*, end to end.
> Each component has its own deep-dive doc (linked at the bottom).

## The question

Every quarter, a public company reports its earnings per share (EPS). Before that, Wall Street
analysts publish a **consensus estimate** of what EPS will be. If the company reports less than the
estimate, it **missed**; otherwise it **met or beat**.

Companies also file a quarterly report with the SEC (the **10-Q**). One section of it,
**Management's Discussion and Analysis (MD&A)**, is management explaining in their own words how
the business is going. The idea behind this project:

> **Does the way management writes about the business hint that the *next* quarter will disappoint?**

Hedging, vague language, sudden changes in tone, or dropping topics they used to brag about might
precede a miss — a "lie detector" in a loose sense. We test that by training a model that reads an
MD&A and predicts **miss (1)** vs **met/beat (0)** for the company's next earnings announcement.

## The pipeline at a glance

```
 SEC EDGAR ──scraper.py──▶ data/raw/transcripts/{TICKER}.json      (664 MD&A texts, 49 companies, 2020–2025)
 yfinance  ──labeler.py──▶ data/processed/labels.csv               (1,899 announcements: EPS actual vs estimate)
                                   │
                         dataset.py │ pairs each filing with the NEXT announcement after it
                                   ▼
                         661 (text, label) samples, 92 misses (14%)
                                   │
            ┌──────────────────────┼───────────────────────┐
            ▼                      ▼                       ▼
     train.py (FinBERT)    baselines.py (TF-IDF)   baselines.py (history only)
            └──────────────────────┴───────────────────────┘
                                   ▼
                 F1 / AUC on a time-based test set (2025–2026)
```

| Step | File | Doc |
|---|---|---|
| 1. Collect text | `src/data/scraper.py` | [01-scraper.md](01-scraper.md) |
| 2. Create labels | `src/data/labeler.py` | [02-labeler.md](02-labeler.md) |
| 3. Join text + labels, split | `src/data/dataset.py` | [03-dataset.md](03-dataset.md) |
| 4. Neural model + training | `src/models/model.py`, `train.py` | [04-model-and-training.md](04-model-and-training.md) |
| 5. Measuring and comparing | `src/models/baselines.py` | [05-evaluation-and-baselines.md](05-evaluation-and-baselines.md) |
| 6. Text features (embeddings, change scores) | `src/models/text_features.py` | [07-text-features.md](07-text-features.md) |
| 7. Fusion model (history + text, hand-written loop) | `src/models/fusion.py` | [08-fusion-model.md](08-fusion-model.md) |
| 8. Hyperparameter sweep (W&B) | `src/models/sweep.py` | [09-wandb-sweeps.md](09-wandb-sweeps.md) |
| 9. Evaluation deep dive (multi-year, paired bootstrap, unseen companies) | `src/models/evaluate.py` | [10-evaluation-deep-dive.md](10-evaluation-deep-dive.md) |
| 10. Publish to the Hugging Face Hub | `src/models/fusion_net.py`, `push_to_hub.py` | [11-huggingface-hub.md](11-huggingface-hub.md) |
| (optional) Audio | `src/data/audio_*.py` | [06-audio-pipeline.md](06-audio-pipeline.md) |

## Three ideas that matter most

1. **No peeking at the answer (leakage).** A 10-Q is filed *right after* that quarter's results are
   announced, so it already contains the EPS we'd be predicting. We therefore pair each filing with the
   **next** announcement, ~90 days later, which hadn't happened when the text was written.
   → [03-dataset.md](03-dataset.md)
2. **Test on the future, not the past.** We train on 2020–2023, tune on 2024, and test on 2025–2026.
   A random split let the model study a company's later quarters and then "predict" its earlier ones,
   which made an early result look better than it really was (AUC 0.65 → 0.52).
   → [05-evaluation-and-baselines.md](05-evaluation-and-baselines.md)
3. **Imbalanced classes.** Only ~14% of announcements are misses. A model that always says "beat" is
   86% accurate and useless, so we ignore accuracy and use **F1** and **AUC**, and weight the loss so
   misses count more. → [05-evaluation-and-baselines.md](05-evaluation-and-baselines.md)

## Where we are (2026-10-22)

Test set: 169 announcements in 2025–2026, 23 of them misses. AUC 0.5 = coin flip, 1.0 = perfect.

| Model | Val AUC | Test AUC | Test F1 |
|---|---|---|---|
| FinBERT, first 512 tokens, encoder frozen (5 seeds) | 0.59 | 0.515 ± 0.03 | 0.12 |
| Change-from-last-quarter scores only | 0.51 | 0.518 | 0.21 |
| TF-IDF, full text | 0.65 | 0.672 | 0.23 |
| MiniLM embedding of the full text | 0.65 | 0.703 | 0.31 |
| Company's past surprises only (no text) | 0.73 | 0.728 | 0.39 |
| **Past surprises + embedding (PCA-16)** — best | **0.75** | **0.752** | 0.35 |

**What this tells us**

- The filing text **does** carry signal, but only when the model reads the whole thing: a frozen
  embedding of the full text (0.70) beats FinBERT on the first 512 tokens (0.52).
- A company's own track record is the strongest single predictor.
- Text + history together is the best so far (+0.02 AUC on both validation and test) — promising,
  but within the noise of a 23-miss test set. → [07-text-features.md](07-text-features.md)
- "Did they rewrite the report?" scores showed no signal for next-quarter EPS.
- A small neural fusion network (history branch + text branch) reaches val 0.69 / test 0.73 — not
  better than the logistic regression on val, which is how we choose. → [08-fusion-model.md](08-fusion-model.md)
- A 50-run W&B sweep tuned it to val 0.73 / test 0.74 — level with logistic regression on history.
  The sweep's main finding: squeeze the text to 4 features, give history more room. → [09-wandb-sweeps.md](09-wandb-sweeps.md)
- **Deep dive (2023–26 pooled, 58 misses):** best models reach ~0.70 AUC; text adds +0.016 over history
  [−0.02, +0.05], so not clearly anything. On companies never seen in training, the text embedding
  drops to chance (0.48): it mostly recognises *which company* wrote the filing.
  → [10-evaluation-deep-dive.md](10-evaluation-deep-dive.md)

## What's next

1. Make the gain solid: rolling evaluation over several test years, and more labelled data.
2. Better text sources (earnings press releases, call transcripts) where management talks about the
   outlook more directly than in a 10-Q.
3. Optional: LLM-scored features (hedging, outlook tone, new risks) as another input.

## Glossary

- **EPS** — earnings per share, profit ÷ shares outstanding.
- **Consensus estimate** — average analyst forecast of EPS.
- **Surprise %** — (actual − estimate) / |estimate| × 100.
- **10-Q / MD&A** — quarterly SEC report / its narrative section written by management.
- **Leakage** — when training or input data contains information that wouldn't be available at
  prediction time, making results look better than they'd be in real use.
- **Baseline** — a deliberately simple model used as a yardstick.
