# 13 — More companies: 49 → 188

**Files:** `src/data/universe.py` → `data/universe.csv`; `evaluate.py --companies all|original|expansion`
and the sector baseline; outputs `checkpoints/evaluation.json`, `evaluation_press.json`,
`evaluation_press_original.json`

## Why

Every result in steps 9–12 ended with "not significant". How wide a paired-bootstrap interval is depends
mostly on the **number of misses**, roughly 1/√(misses). A better feature extractor (the LLM, step 12)
didn't help, so the next lever was more data.

## The universe

`universe.py` keeps the original 50 tickers and adds **150 companies from the current S&P 500**:

- **Stratified by GICS sector:** each sector gets new companies in proportion to its share of the remaining
  pool, sampled with a fixed seed. So the expansion isn't just more mega-cap tech, and Materials, Real
  Estate and Utilities appear for the first time.
- **One ticker per company:** GOOG/GOOGL and similar pairs → one; `BRK.B` → `BRK-B`.
- **Committed:** the list is saved in `data/universe.csv`, so every machine scrapes the same set even after
  the index changes.

The same three pipelines then ran on the new tickers: 10-Q MD&A (2,529 sections, ~1 h), 8-K press
releases (6,198, ~2 h) and Yahoo labels (~2 min). The new labels went into `labels_expansion.csv` and
were then appended, so the original 1,899 rows are byte-identical (`--companies original` reproduces
every earlier number).

### A label problem: REITs

The new **Real Estate** companies "missed" 36% of the time; equity REITs such as O, CPT, EQIX, UDR and
VICI missed 45–62%. EQIX "missed" by 25–49% quarter after quarter. REITs report **FFO** (funds from
operations), and Yahoo's EPS actual and estimate are often on different bases. That is label noise, not a
signal, so **the Real Estate sector is excluded** (`exclude` column in `universe.csv`).

Utilities also miss often (28%), but with normal-sized surprises (median 6%), so they stay.

### Caveats

- **Survivorship:** today's index members did well enough to stay in the index. The original 50 have the
  same bias.
- **Recent listings** (RDDT, SNDK, HONA, …) contribute only a few quarters.
- **Banks have 1 MD&A each** (JPM, BAC, GS, …), in the original scrape too: banks put their MD&A in an
  exhibit, not the 10-Q body. Their press releases are complete.

| | 49 companies | **188 companies** |
|---|---|---|
| Press-release samples / misses | 1,848 / 251 | **6,916 / 1,161** |
| …pooled rolling-origin test misses | 100 | **469** |
| MD&A samples / misses | 661 / 92 | **2,983 / 545** |
| Miss rate | 13.6% | 16.8% |

## A new baseline: sector

Miss rates now range from **8% (IT) to 29% (Materials)**, so "which kind of company" carries
base-rate information. Text can pick that up too: a release from a utility *reads* like a utility. To
check whether text adds anything beyond this, `evaluate.py` adds **one-hot GICS sector** as a block and
compares every text model with **history + sector** as well as with history.

## Results (press releases, 188 companies)

| Model | Pooled AUC | Δ vs history [95%], P | Unseen companies Δ vs history [95%], P |
|---|---|---|---|
| history | 0.655 | — | — (0.653) |
| history + guidance | 0.662 | +0.007 [−0.007, +0.020], 0.84 | +0.012 [+0.003, +0.022], 1.00 |
| history + embed PCA-16 | 0.666 | +0.010 [−0.008, +0.029], 0.86 | +0.024 [+0.012, +0.035], 1.00 |
| **history + sector** | 0.670 | **+0.015 [−0.002, +0.033], 0.96** | **+0.015 [+0.004, +0.025], 1.00** |
| history + sector + guidance | **0.676** | +0.021 [+0.002, +0.041], 0.99 | +0.018 [+0.006, +0.031], 1.00 |
| fusion (swept) | **0.676** | +0.021 [+0.003, +0.040], 0.99 | — |

**Text vs history + sector:**

| Added to history + sector | Pooled Δ [95%], P | Unseen companies Δ [95%], P |
|---|---|---|
| guidance | +0.006 [−0.004, +0.017], 0.86 | +0.004 [−0.003, +0.011], 0.85 |
| embed PCA-16 | +0.001 [−0.009, +0.011], 0.58 | +0.008 [+0.002, +0.013], 1.00 |
| guidance + embed | +0.004 [−0.009, +0.018], 0.72 | +0.001 [−0.008, +0.009], 0.54 |

On MD&A it is the same, only weaker: history + sector 0.664 (+0.020 over history, P 0.96), and every
text block added to it is ≤ 0 pooled and ≤ +0.004 on unseen companies.

## What changed, and why

1. **The intervals narrowed as expected,** from ±0.05 to ±0.02. That is what 4.7× the test misses buys.
2. **The guidance gain shrank:** +0.033 → +0.012 on unseen companies, +0.014 → +0.007 over time. With 50
   companies the first estimate was inflated by luck. This is the *winner's curse* again (see
   [09](09-wandb-sweeps.md)): the result we noticed because it was the biggest was also the luckiest.
3. **Most of it was the sector.** Re-running the original 50 with the sector baseline: history + sector
   alone gives **+0.033 on unseen companies**, exactly the "guidance gain" from step 11, and guidance
   added on top gives −0.010. Whether a company publishes an outlook depends on its industry (96% of
   health-care releases have one, 42% in communication services), so `has_outlook` partly acted as an
   industry flag.
4. **The direction of guidance still means something,** just less than it seemed. On the new companies the
   next-quarter miss rate is 10.7% after a raise (n = 699) and 27.8% after a net cut (54), against a
   17% base. On the original 50 it was 6% and 44%. History already carries most of it: companies that
   raise have usually been beating.
5. **The MD&A guidance "best model" (+0.028) was noise:** −0.002 with 186 companies.

## Bottom line

With 188 companies and 1,161 misses, the answer is clear and well measured: **a company's own earnings
history plus its sector predict next-quarter misses (AUC ≈ 0.67); press-release text, whether as a
generic embedding, guidance rules or LLM scores, adds at most +0.006, not significantly different
from zero.** The large early gains were small-sample luck plus a hidden sector effect. That's why we
scaled up before believing them.
