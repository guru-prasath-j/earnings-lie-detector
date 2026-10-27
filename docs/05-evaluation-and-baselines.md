# 05 — Evaluation and baselines

**File:** `src/models/baselines.py` · **Output:** `checkpoints/baselines.json`

## Reading the metrics

Confusion matrix on the test set (positive class = **miss**):

|  | predicted miss | predicted beat |
|---|---|---|
| **actual miss** | TP (caught) | FN (missed the miss) |
| **actual beat** | FP (false alarm) | TN |

| Metric | Formula | Plain meaning |
|---|---|---|
| Precision | TP / (TP + FP) | When we say "miss", how often are we right? |
| Recall | TP / (TP + FN) | Of the real misses, how many did we catch? |
| Specificity | TN / (TN + FP) | Of the beats, how many did we correctly leave alone? |
| **F1** | 2·P·R / (P + R) | One number balancing precision and recall; 0 if we never catch a miss |
| **AUC** | area under ROC curve | Chance a random miss is scored higher than a random beat. 0.5 = coin flip, 1.0 = perfect. Doesn't depend on the 0.5 threshold |

**Why not accuracy?** 86% of samples are beats. "Always beat" scores 86% accuracy, F1 = 0, AUC = 0.5.

## Why baselines

A number like "AUC 0.62" means nothing alone. Baselines are simple models that tell us whether the
fancy one is actually adding anything.

```bash
python src/models/baselines.py --data-root .
```

Same 661 samples and the same temporal split as `train.py`. Every model is logistic regression with
`class_weight="balanced"`; the regularisation strength `C` (0.001–10) is chosen by validation AUC.
Each test AUC comes with a **95% bootstrap range** (resample the test set 2,000 times).

| Feature block | Features | Question it answers |
|---|---|---|
| `history` | prior quarter miss and surprise %, 4-quarter miss rate, mean and std of surprise | How far can we get **without any text**? |
| `tfidf` | TF-IDF of the **full** MD&A (1–2-word phrases, 50k vocab, fit on train only) | Does the text carry signal when nothing is truncated? |
| `change`, `embed`, `embed_pca*` | see [07-text-features.md](07-text-features.md) | Which text representation works best? |

Blocks are tested alone and combined (e.g. `history+embed_pca16`), plus an average of separate
history and embedding models.

- **History features** use only announcements *before* the one being predicted (`shift(1)`), so
  there's no leakage. Data before 2020 (from the labeler) provides history for early samples.
- **TF-IDF** scores each word/phrase by how often it appears in this document vs how common it is
  across all documents; logistic regression then learns which phrases lean "miss".

## Results (2026-10-22, test = 2025–2026, 169 samples, 23 misses)

| Model | Val AUC | Test AUC [95% range] | Test F1 | Precision | Recall |
|---|---|---|---|---|---|
| FinBERT frozen, 512 tokens (5 seeds) | 0.59 | 0.515 ± 0.03 | 0.12 | 0.25 | 0.08 |
| TF-IDF full text | 0.652 | 0.672 [0.55–0.78] | 0.227 | 0.24 | 0.22 |
| Embedding full text | 0.645 | 0.703 [0.58–0.82] | 0.312 | 0.24 | 0.43 |
| History only | 0.734 | 0.728 [0.61–0.83] | 0.386 | 0.32 | 0.48 |
| History + TF-IDF | 0.739 | 0.739 [0.63–0.83] | 0.386 | 0.32 | 0.48 |
| **History + embedding PCA-16** | **0.752** | **0.752** [0.64–0.85] | 0.351 | 0.29 | 0.43 |

(History-only moved from 0.714 to 0.728 when `C=0.001` was added to the search grid.)

Earlier, on a random split, the same FinBERT setup scored test AUC 0.65 — the drop to 0.52 on the
time-based split shows how much a random split flattered it.

## Interpretation and caveats

- **Earnings surprises are persistent**: companies that beat tend to keep beating (analysts set
  beatable estimates for some firms). History alone is a strong baseline; any text model must beat
  **AUC ≈ 0.73** to matter.
- **The text has signal** (0.67), but it turned out to be mostly *which company* wrote it: on companies
  held out of training, a text-only model drops to chance (AUC 0.48). That overlaps with what history
  already captures. → [10-evaluation-deep-dive.md](10-evaluation-deep-dive.md)
- **Small test set**: 23 misses. One extra caught miss moves recall by 4 points. Differences under
  ~0.03–0.05 AUC are not reliable. Over three test years (58 misses) with a paired bootstrap, the
  +0.024 of history + PCA-16 here shrinks to +0.016 [−0.02, +0.05]: not significant.
  → [10-evaluation-deep-dive.md](10-evaluation-deep-dive.md)

## Concepts to learn here

Confusion matrix and threshold-free metrics; baselines as yardsticks; regularisation in linear
models; confounders (company identity); statistical noise in small evaluation sets.
