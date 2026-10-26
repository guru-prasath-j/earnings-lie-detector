# 10 — Evaluation deep dive: how sure are we?

**File:** `src/models/evaluate.py` (~1 min) → `checkpoints/evaluation.json`

Every result so far came from **one** test window (2025–26) with **23 misses**. With so few
positives, each AUC has a ±0.11 bootstrap range and no model difference was clearly real. This
script asks four questions.

## 1. Rolling-origin evaluation: test on several years

Repeat the train → val → test protocol with the test year moving forward, always training only on
the past:

| Fold | Train | Val (picks C, threshold) | Test |
|---|---|---|---|
| 2023 | 2020–21 (157, 18 miss) | 2022 (101, 16) | 2023 (112, 14) |
| 2024 | 2020–22 (258, 34) | 2023 (112, 14) | 2024 (122, 21) |
| 2025–26 | 2020–23 (370, 48) | 2024 (122, 21) | 2025–26 (169, 23) — the usual split |

Pooling the three test years gives **403 samples / 58 misses**, 2.5× more evidence.

**Test AUC per fold:**

| Model | 2023 | 2024 | 2025–26 | **Pooled** [95%] |
|---|---|---|---|---|
| history (LR) | 0.630 | 0.728 | 0.728 | 0.693 [0.61–0.76] |
| embed (LR) | 0.716 | 0.629 | 0.703 | 0.675 [0.60–0.75] |
| history + PCA-16 (LR) | 0.633 | 0.724 | 0.752 | **0.708** [0.63–0.78] |
| late fusion, swept (3 seeds) | 0.689 | 0.721* | 0.749 | 0.705 [0.62–0.78] |

\* The sweep picked its settings on 2024 data, which is this fold's test year, so 0.721 is optimistic.
(PCA-16 as a design was also chosen by looking at 2024.)

- **The single 2025–26 window was flattering.** history + PCA-16 drops from 0.752 there to 0.708
  pooled.
- **Model rankings flip between years.** Text alone is best in 2023 and worst in 2024. One year is not
  enough to rank models.

## 2. Paired bootstrap: is text really adding anything?

Comparing two separate confidence intervals is weak: they overlap almost always. Instead, resample the
pooled test rows 2,000 times and compute **AUC(model) − AUC(history) on the same resample**. Both models
see the same easy and hard rows, so most of the noise cancels out.

| Model | ΔAUC vs history | 95% interval | P(Δ > 0) |
|---|---|---|---|
| history + PCA-16 | +0.016 | [−0.021, +0.053] | 0.80 |
| late fusion, swept | +0.012 | [−0.054, +0.075] | 0.64 |
| embed only | −0.018 | [−0.106, +0.072] | 0.34 |

Every interval includes 0. **Adding the filing text to the company's track record gives at most a few
hundredths of AUC, and we can't rule out zero.**

## 3. Decisions: thresholds and "flag the top 10%"

AUC only measures ranking. To *act*, you need a cutoff.

| Model (pooled) | Avg. precision | F1 @ 0.5 | F1 @ val-tuned threshold | Precision in top 10% |
|---|---|---|---|---|
| history | 0.270 | 0.394 | 0.318 | 30% (12/40) |
| embed | 0.283 | 0.314 | 0.312 | 38% (15/40) |
| history + PCA-16 | 0.285 | 0.364 | 0.340 | 40% (16/40) |
| late fusion, swept | 0.307 | 0.412 | 0.341 | 38% (15/40) |

Base rate = 14.4% (a random guess gets average precision ≈ 0.144 and top-10% precision ≈ 14%).

- **Tuning the threshold on val made F1 *worse*.** The F1-best cutoff on 100 val rows with 15–20 misses
  is noise and doesn't transfer. Class-balanced training already puts a sensible cutoff at 0.5.
- **Most practical framing:** flag the 10% riskiest announcements each year, and **~40% of them are real
  misses, ~2.7× the base rate**. Useful as a watch-list, not as a yes/no prediction.

## 4. Company-held-out: is the text just recognising the company?

5-fold cross-validation over all years, split two ways (this ignores time, so it is a diagnostic, not a
forecast):

- **random**: rows shuffled into folds, so the same company appears in train and test
- **by company** (`GroupKFold` on ticker): test companies are **never seen** in training

| Model | AUC, random folds | AUC, unseen companies |
|---|---|---|
| history | 0.703 | 0.681 |
| **embed** | 0.673 | **0.481** (chance) |
| history + PCA-16 | 0.715 | 0.674 |

**This is the key finding.** The text embedding's skill vanishes for unseen companies. Each company's
MD&A has a recognisable style, so the embedding mostly learns *"this is company X, and X misses often"*.
That's the same information the history features carry directly (a company's own miss rate). History
transfers to new companies because "missed last quarter" means the same thing everywhere.

It also explains question 2: text adds little on top of history because both mostly encode the
company's track record.

## What this means for the project

1. **The honest headline:** about 0.70 AUC pooled over 2023–26, driven by the company's own earnings history.
   A 10% watch-list is ~40% precise.
2. **MD&A embeddings carry no transferable "lie detector" signal.** What they carry is company identity.
   To find real language signal, features must be *within-company*: how this filing differs from the
   same company's earlier ones (our change features tried that and found nothing), or text where
   management talks about the coming quarter (earnings-call Q&A, guidance in 8-K press releases).
3. **Evaluate with rolling-origin + paired bootstrap from now on.** A single 23-miss window can't
   separate models that differ by 0.02.
