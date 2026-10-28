# 12 — Earnings press releases (8-K, Exhibit 99.1)

**Files:** `src/data/press_releases.py` (scraper), `src/models/llm_guidance.py` (step 4), `evaluate.py --text press` → `checkpoints/evaluation_press.json`

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

## Step 4: LLM-scored guidance (2026-10-28)

The rules know the *direction* of a guidance change, but not its *size*, whether it is about EPS or
revenue, or the tone around it. Step 4 asks an LLM instead (`src/models/llm_guidance.py`).

### How it works

1. **Only the outlook passage is sent.** Sentences mentioning guidance / outlook / forecast (minus
   boilerplate) plus two neighbours on each side, at most 6,000 characters. 1,376 of 2,115 releases have
   one; the rest are cached as "no outlook" without a call.
2. **A fixed JSON schema**, temperature 0: `eps_guidance` and `revenue_guidance` (raised / lowered /
   maintained / initiated / given / withdrawn / not_mentioned), `eps_change_pct` and
   `revenue_change_pct` (change of the midpoint), `horizon`, `outlook_tone` (−2 to 2), `headwinds`,
   `specificity` (0 none, 1 qualitative, 2 numbers). The prompt spells out edge cases: "raised" only if
   the text compares with a previous figure, "given" when figures are stated without a comparison,
   segment / margin / tax-rate guidance doesn't count.
3. **Nine numeric features** per release (`LLM_COLS`): has guidance, EPS direction (+1 / −1), EPS
   withdrawn, EPS change % (clipped to ±20), revenue direction and change, tone, headwinds,
   specificity. `build_table(text="press")` adds them when the cache exists.

**The API:** NVIDIA's free, OpenAI-compatible endpoint (`integrate.api.nvidia.com`) with
`openai/gpt-oss-20b`. Llama 3.3 70B had been retired (HTTP 410), and several other listed models
returned 404 or unparseable output. The key is read from `~/personal/.secrets/nvidia.env`, never printed.

**Engineering lessons:**

- **Cache every answer** (`data/processed/llm_guidance.jsonl`, one JSON line per release) so a crash or
  restart resumes. `--retry-failed` re-scores the failures.
- **Test on 20 releases first** and read the answers. The first prompt overused "initiated" and gave a
  revenue change of 33% for a growth rate moving from 3% to 4%. Adding "given" and the percentage-point
  rule fixed both.
- **Free tiers queue.** Calls took 16 s at first and 90–250 s under load. A 120 s client timeout threw
  away a quarter of the calls; 300 s and 16 parallel workers fixed it. The whole run took ~4 hours.
- **Don't retry permanent errors** (4xx other than 429).

Result: 1,367 releases scored, 6 failed after retries (unparseable answers, treated as "no guidance"),
738 with no outlook passage.

### Simple tables first

| Next-quarter miss rate (base 13.6%) | n | Miss rate |
|---|---|---|
| EPS guidance raised | 217 | 6.9% |
| EPS raised by ≥ 3% | 31 | **0%** |
| EPS raised by < 3% | 186 | 8.1% |
| EPS guidance lowered | 27 | 11.1% |
| Revenue guidance raised | 169 | 5.3% |
| Tone confident (+1 / +2) | 399 / 211 | 8.3% / 6.6% |
| Tone neutral or cautious (0 / −1 / −2) | 1,082 / 111 / 45 | 16.5% / 17.1% / 13.3% |
| Numeric guidance | 722 | 8.6% |
| No guidance | 976 | 17.7% |

The size of a raise matters (no misses after a raise of 3% or more). But "lowered" is *weaker* than the
rules' "cut" (11% vs 44%): the LLM finds more cuts (27 vs 16), and the extra ones are milder. The rules
only fire on blunt sentences, which turn out to be the alarming ones.

Rules and LLM mostly agree on direction: 190 releases both call a raise, 8 both call a cut. The LLM adds
82 raises and 23 cuts the rules missed, and sees no EPS or revenue change in 35 of the rule raises
(likely segment, margin or cash-flow guidance, which the schema excludes; not checked one by one).

### Results

| Model | Rolling-origin pooled AUC | Δ vs history [95%], P(Δ>0) | Unseen companies | Δ vs history [95%], P(Δ>0) |
|---|---|---|---|---|
| history | 0.653 | — | 0.646 | — |
| guidance (rules) | 0.581 | | 0.606 | |
| llm | 0.585 | | 0.597 | |
| history + guidance (rules) | 0.667 | +0.014 [−0.038, +0.064], 0.68 | 0.679 | **+0.033 [+0.004, +0.062], 0.99** |
| history + llm | 0.665 | +0.012 [−0.039, +0.063], 0.67 | 0.671 | +0.025 [−0.006, +0.055], 0.94 |
| history + guidance + llm | 0.669 | +0.016 [−0.039, +0.070], 0.69 | 0.665 | +0.020 [−0.013, +0.051], 0.88 |
| history + llm + embed PCA-16 | 0.679 | +0.026 [−0.028, +0.078], 0.84 | 0.682 | **+0.037 [+0.001, +0.071], 0.98** |

- **The LLM matches the rules; it doesn't beat them.** Alone 0.585 vs 0.581; with history 0.665 vs 0.667.
- **Combining both adds nothing**: the two read the same sentences, so they carry the same information.
- **The LLM's extra detail (size, EPS vs revenue, tone) doesn't show up in AUC.** Large raises are rare
  (31), and logistic regression on nine features with ~250 misses can't exploit them.

### What this means

Guidance is the signal, and simple rules already capture it. An LLM costs ~4 hours of API time (free
here; a few dollars on a paid API) for the same result. That is a useful finding: the bottleneck is the
number of misses to learn from, not the quality of the extraction.
