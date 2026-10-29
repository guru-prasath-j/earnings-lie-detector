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
 universe.py ─▶ data/universe.csv                                  (49 original + 140 S&P 500 companies by sector)
 SEC EDGAR ──scraper.py──▶ data/raw/transcripts/{TICKER}.json      (3,193 MD&A texts, 2020–2025)
           ──press_releases.py──▶ data/raw/press_releases/          (8,313 earnings releases, 2016–2026)
 yfinance  ──labeler.py──▶ data/processed/labels.csv               (7,089 announcements: EPS actual vs estimate)
                                   │
                         dataset.py │ pairs each filing with the NEXT announcement after it
                                   ▼
              MD&A: 2,983 samples, 545 misses · press: 6,916 samples, 1,161 misses (17%)
                                   │
        ┌──────────────────────────┼─────────────────────────────┐
        ▼                          ▼                             ▼
  train.py: FinBERT         text_features.py:              baselines.py:
  (frozen / LoRA,           MiniLM embedding of the        history features
   512 tokens or chunks)    whole filing, change scores    (past surprises)
        │                          └──────────────┬──────────────┘
        │                                         ▼
        │                    baselines.py (logistic regression) · fusion.py (neural, tuned by sweep.py)
        └───────────────────────────────┬─────────┘
                                        ▼
            evaluate.py: test years 2023 / 2024 / 2025–26, paired bootstrap, unseen companies
                                        ▼
                       push_to_hub.py → huggingface.co/Guruprasath12/lie-detect (history + sector + guidance)
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
| 10. Publish to the Hugging Face Hub | `src/models/fusion_net.py`, `miss_model.py`, `guidance_rules.py`, `push_to_hub.py` | [11-huggingface-hub.md](11-huggingface-hub.md) |
| 11. Earnings press releases (8-K Ex. 99.1) as a second text source; guidance rules and LLM scores | `src/data/press_releases.py`, `evaluate.py --text press`, `llm_guidance.py` | [12-press-releases.md](12-press-releases.md) |
| 12. More companies (49 → 188) and a sector baseline | `src/data/universe.py`, `evaluate.py --companies` | [13-more-companies.md](13-more-companies.md) |
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

## Where we are (2026-10-29)

### The answer so far

**Partly yes, but not because of the language.** Models rank upcoming misses clearly better than chance
(AUC ≈ 0.67 on 188 companies over 2023–26; flagging the riskiest 10% finds misses at ~2× the base rate).
That signal comes from the company's **own earnings track record and its sector**. The text, whether 10-Q
MD&A or earnings press release, read as an embedding, with guidance rules or by an LLM, adds at most
+0.006 AUC on top of history + sector, not significantly different from zero.

With 49 companies, guidance had looked like the exception (+0.033 on unseen companies). Scaling to 188
companies ([13](13-more-companies.md)) showed that most of that gain was luck plus a hidden **sector**
effect: whether a company publishes an outlook at all depends on its industry. Guidance *direction*
still means something (miss rate 11% after a raise, 28% after a cut, 17% base), but history already
carries most of it.

### The journey (each step has its own doc)

| Step | What we tried | What we learned | Doc |
|---|---|---|---|
| 1 | FinBERT on the first 512 tokens | Test AUC 0.52 = chance. The first 512 tokens are boilerplate | [04](04-model-and-training.md) |
| 2 | Baselines: past surprises only; TF-IDF of the full text | History alone 0.73; full text 0.67. The bar any text model must clear | [05](05-evaluation-and-baselines.md) |
| 3 | Frozen MiniLM embedding of the whole filing; "did they rewrite it?" scores | Embedding 0.70, best text representation; rewrite scores: no signal | [07](07-text-features.md) |
| 4 | LoRA fine-tuning; FinBERT reading 16 chunks of the filing | Whole filing lifts FinBERT to 0.60; fine-tuning adds nothing. Frozen features + small model win | [04](04-model-and-training.md) |
| 5 | Neural fusion of history + embedding, hand-written loop | Late fusion > early fusion; still ≈ logistic regression | [08](08-fusion-model.md) |
| 6 | 50-run W&B hyperparameter sweep | Best configs squeeze text to 4 features, give history 16. Winner's curse visible | [09](09-wandb-sweeps.md) |
| 7 | Evaluation over 3 test years, paired bootstrap, unseen companies | ~0.70 pooled; text gain not significant; text = company identity | [10](10-evaluation-deep-dive.md) |
| 8 | Published the fusion model with an honest model card | Round-trip test caught a silent branch-order bug | [11](11-huggingface-hub.md) |
| 9 | Text with company identity removed (vs own past filings) | Chance (0.49–0.51); MD&A has no next-quarter signal | [10](10-evaluation-deep-dive.md) §5 |
| 10 | Earnings press releases (2,115, 2016–26; 1,848 samples) | +0.023 over history (P=0.84, not significant), but text keeps 0.58 on unseen companies (MD&A 0.48) | [12](12-press-releases.md) |
| 11 | Rule-based guidance features (raised / cut / maintained / withdrawn, hedging) | **First significant gain:** +0.033 over history on unseen companies (P=0.99); over time +0.014–0.028, not significant. Works from MD&A too | [12](12-press-releases.md) |
| 12 | LLM-scored guidance (gpt-oss-20b via NVIDIA API: EPS / revenue direction and size, tone) | Same as the rules (+0.012 over time, +0.025 unseen, P 0.94); combining both adds nothing. Raises ≥ 3%: 0 misses in 31 | [12](12-press-releases.md) |
| 13 | 4× the companies (49 → 188, S&P 500 by sector; REITs excluded) and a sector baseline | Intervals ±0.05 → ±0.02. Guidance gain shrinks to +0.012 and is mostly sector; text over history + sector ≤ +0.006, n.s. | [13](13-more-companies.md) |
| 14 | Re-published the Hub model as LR history + sector + guidance (chosen by val), rules shipped with it | Test 2025–26 AUC 0.671; card states text adds +0.006, n.s. | [11](11-huggingface-hub.md) |

### Headline numbers, 188 companies (press releases, rolling-origin test 2023–26: 2,816 samples, 469 misses)

| Model | Pooled AUC | vs history alone | Unseen companies vs history |
|---|---|---|---|
| Past surprises only | 0.655 | — | — |
| + sector | 0.670 | +0.015 [−0.002, +0.033] | +0.015 [+0.004, +0.025] |
| + sector + guidance rules | **0.676** | +0.021 [+0.002, +0.041] | +0.018 [+0.006, +0.031] |
| Guidance on top of history + sector | | +0.006 [−0.004, +0.017] | +0.004 [−0.003, +0.011] |

Base rate 16.7%. Brackets are 95% paired-bootstrap intervals.

### Earlier headline, 49 companies (MD&A, 403 test samples, 58 misses)

| Model | Pooled AUC | vs history alone | Precision in top 10% |
|---|---|---|---|
| Past surprises only (logistic regression) | 0.693 | — | 30% |
| Text embedding only | 0.675 | −0.018 | 38% |
| Past surprises + embedding PCA-16 | **0.708** | +0.016 [−0.02, +0.05] | 40% |
| Neural late fusion (swept) | 0.705 | +0.012 [−0.05, +0.08] | 38% |

Base rate 14.4%. All intervals include 0; with 5× fewer misses they are 2.5× wider than above.

## What's next

1. **Better history features** are where the signal is: analyst-revision trends, estimate dispersion,
   the number of analysts (needs a data source beyond Yahoo's EPS table).
2. Text with a different kind of information: earnings-call Q&A (no free source) rather than more
   ways of reading the same press release.

## Glossary

- **EPS** — earnings per share, profit ÷ shares outstanding.
- **Consensus estimate** — average analyst forecast of EPS.
- **Surprise %** — (actual − estimate) / |estimate| × 100.
- **10-Q / MD&A** — quarterly SEC report / its narrative section written by management.
- **Leakage** — when training or input data contains information that wouldn't be available at
  prediction time, making results look better than they'd be in real use.
- **Baseline** — a deliberately simple model used as a yardstick.
- **AUC** — chance a random miss is scored above a random beat; 0.5 = coin flip, 1.0 = perfect.
- **Rolling-origin evaluation** — repeat train → test with the test year moving forward, always training on the past.
- **Paired bootstrap** — resample the same test rows for two models to get a confidence interval on their *difference*.
- **LoRA** — fine-tuning through small low-rank adapter matrices while the original weights stay frozen.
