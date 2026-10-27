# 12 — Earnings press releases (8-K, Exhibit 99.1)

**Files:** `src/data/press_releases.py` (scraper), `evaluate.py --text press` → `checkpoints/evaluation_press.json`

## Why press releases

The MD&A turned out to carry company identity and nothing else ([10](10-evaluation-deep-dive.md) §4–5).
The press release is a different kind of text: it is written for investors on announcement day, and it is
where management frames the quarter and often gives guidance for the next one.

On that day the company files an **8-K** with **Item 2.02 "Results of Operations and Financial
Condition"** and attaches the release as **Exhibit 99.1**. Both are free on EDGAR, the same source as the 10-Qs.

## The scraper

1. **Find the filings.** The submissions API tags each 8-K with its item numbers (`"2.02,9.01"`), so we
   keep 8-Ks containing 2.02. The API's main file holds only the ~1,000 most recent filings. For a bank
   (which files thousands of debt notices a year) that is about one year, so older pages are fetched too.
2. **Find the exhibit.** `index.json` doesn't say which file is which, but the filing's `-index.htm` page
   has a *Type* column: pick `EX-99.1` (fallbacks: a filename like `ex991`, then any `EX-99.*`).
3. **Extract the text.** Strip iXBRL and HTML, drop `<table>`s (the financial statements are numbers, not
   language), join with spaces so footnote superscripts don't break sentences, and cut the SGML header.

Result: **2,115 releases from 50 companies, 2016–2026**, median ~16k characters. Two companies are thin:
XOM (1 release; it rarely tags results 8-Ks as Item 2.02) and BLK (9).

## Pairing, and a leakage guard that mattered

The output uses the same `date` / `text` record format as the MD&A, so `EarningsCallDataset` pairs each
announcement with the latest release before it, exactly as before. But a release is filed **on**
announcement day and states that day's EPS. "Strictly before" is only safe if the label dates are exact,
and they come from Yahoo Finance.

`min_transcript_gap_days=30` (in `baselines.TEXT_SOURCES`) requires at least 30 days between release and
label. Without it:

- **3 samples** were paired with an announcement 0–3 days after the release, which was almost certainly
  its own result. That is leakage.
- **82 more** were paired with a mid-quarter 8-K (a pre-announcement, a bank's January update,
  Costco's monthly sales report).

With the guard, gaps are 83–98 days (median 91), the same shape as the 10-Q pairing.

**1,848 samples, 251 misses**, 2.8× the MD&A dataset, because labels start in 2017 and releases do too.

## Results

Same protocol as [10](10-evaluation-deep-dive.md): rolling-origin folds testing 2023, 2024 and 2025–26,
pooled into **738 test samples / 100 misses** (vs 403 / 58 for MD&A). Each model is compared with history
*on the same rows*.

| Model | Pooled AUC | ΔAUC vs history [95%] | P(Δ>0) | Unseen companies |
|---|---|---|---|---|
| history | 0.653 | — | — | 0.646 |
| embed (press release) | 0.603 | −0.051 [−0.133, +0.027] | 0.09 | **0.580** |
| history + embed PCA-16 | 0.676 | +0.023 [−0.023, +0.066] | 0.84 | 0.661 |
| late fusion, swept | 0.679 | +0.026 [−0.020, +0.071] | 0.86 | — |
| resid (within-company) | 0.542 | −0.111 [−0.200, −0.028] | 0.01 | 0.523 |
| history + resid PCA-16 | 0.610 | −0.043 [−0.092, +0.001] | 0.03 | 0.642 |
| delta (vs previous release) | 0.502 | −0.151 | 0.00 | 0.513 |
| history + delta PCA-16 | 0.656 | +0.003 [−0.044, +0.047] | 0.52 | 0.647 |

Per fold, history + embed is ahead of history in all three years: 2023 0.688 vs 0.653 (+0.035),
2024 0.646 vs 0.634 (+0.012), 2025–26 0.709 vs 0.673 (+0.036). The direction is consistent, but each
gain is small next to the noise.

### Against the bar

The bar was: **beat history in the paired bootstrap *and* hold up on unseen companies.**

- **Paired bootstrap: not passed.** +0.023 with P(Δ>0) = 0.84. The same size of gain as MD&A (+0.016),
  still with an interval that includes 0, despite almost twice the test misses.
- **Unseen companies: partly passed.** This is the real difference from MD&A. A text-only model on press
  releases keeps **0.58** on companies it never saw (MD&A: 0.48, chance), and history + text keeps
  +0.015 over history there. So *some* of what the release embedding knows is not just "which company".
- **Within-company versions don't help.** resid and delta are near chance and hurt when added. Whatever
  signal there is lives in the release's *absolute* content, not in how it changed from last quarter.

## What this means

Press releases are a better text source than the MD&A: more data, and a little signal that transfers
across companies. But at this sample size, the gain over the company's own track record is still not
distinguishable from zero. History remains the backbone.

The history baseline is lower on these rows (0.653) than on the MD&A rows (0.693): the press-release
sample covers more announcements per company and year. Comparisons are only valid *within* one table.

Possible next steps: score the release for specific things (guidance raised / cut / withdrawn, hedging)
instead of a generic embedding, or combine MD&A and press release on their shared rows.

## Step 3: targeted guidance features (2026-10-27)

A generic embedding squeezes 16,000 characters into one vector, and one sentence like *"we are raising
our full-year EPS guidance"* gets diluted. So we read that sentence directly, with rules
(`text_features.guidance_features`):

1. Split the release into sentences. Keep those mentioning **guidance / outlook / forecast**, minus
   boilerplate ("forward-looking", "reconciliation… without unreasonable effort", "accounting guidance").
2. In each, find the **direction verb closest** to that noun (within 40 characters): raise, cut,
   maintain (reaffirm, reiterate), narrow, initiate (reinstate, introduce), withdraw (suspend, "not providing").
   Verb forms only: bare "increase" and "lower" are usually a noun or an adjective ("an increase of 2%",
   "lower than expected").
3. Features per release: `has_outlook`, `n_outlook`, counts `g_raise`, `g_cut`, `g_maintain`, `g_narrow`,
   `g_initiate`, `g_withdraw`, `g_net` = sign(raise − cut), and `outlook_hedge_rate` (words like
   *uncertain, headwinds, cautious, tariffs* per 1,000 words in outlook sentences).

The rules were tuned by **reading sampled matches for precision, without looking at labels**. The first
version's "cut" was ~50% right (it caught "lower than expected"); after tightening, ~85%.

### Simple tables first

| Next-quarter miss rate | With | Without |
|---|---|---|
| Guidance raised (233 releases) | **6.0%** | 14.7% |
| Guidance cut, net (16) | **43.8%** | 14.3% |
| Guidance maintained (145) | 8.3% | 14.0% |
| Any outlook section (1,206) | 10.8% | 18.8% |
| Heavy hedging in outlook (122) | 20.5% | 13.1% |

Striking, but a company that raises guidance has usually been beating already, which history captures.
The models answer whether guidance adds anything *beyond* that.

### Results

| Model | Rolling-origin pooled AUC | Δ vs history [95%], P(Δ>0) | Unseen companies | Δ vs history [95%], P(Δ>0) |
|---|---|---|---|---|
| history | 0.653 | — | 0.646 | — |
| guidance alone | 0.581 | | 0.606 | |
| history + guidance | 0.667 | +0.014 [−0.038, +0.064], 0.68 | 0.679 | **+0.033 [+0.004, +0.062], 0.99** |
| history + guidance + embed PCA-16 | 0.678 | +0.025 [−0.033, +0.083], 0.80 | 0.684 | **+0.039 [+0.006, +0.072], 0.99** |

The unseen-company column now has its own paired bootstrap: every one of the 1,848 rows is a test row
once, so it has 2.5× the evidence of the rolling-origin test.

- **Guidance transfers across companies.** Alone it scores 0.612 with random folds and 0.606 on unseen
  companies, almost no drop. The embedding fell from 0.717 to 0.580. Guidance means the same thing for
  every company.
- **The first interval in this project that excludes zero:** history + guidance on unseen companies.
- **Over time, the gain is smaller and not significant** (+0.014 to +0.025).
- **The same features from the 10-Q MD&A agree.** history + guidance 0.720 vs history 0.693 (+0.028,
  P = 0.92), the best MD&A model; on unseen companies +0.022 (P = 0.87).

### Caveats

- Company-held-out CV ignores time order: models train on later years too. It is a diagnostic.
- About ten models are compared, so one P = 0.99 is less rare than it looks. But two related models,
  and a second document (MD&A), point the same way.
- n is small for the strongest classes: 21 releases with a cut, 26 with a withdrawal.

**Bottom line:** the first text feature that carries real, transferable signal is *what management
says about guidance*, not how the text "feels" overall. It adds about +0.02–0.04 AUC on top of history,
significant on unseen companies, suggestive over time.
