# 07 — Text features: change scores and document embeddings

**Files:** `src/models/text_features.py` (feature code), `src/models/baselines.py` (uses them)
**Cache:** `data/processed/embeddings_minilm.npz`

## Why not just FinBERT?

FinBERT reads 512 tokens (~3% of an MD&A) and has 110M parameters to fit on 370 training examples.
With so little data, a better recipe is: turn each filing into a few informative numbers using
models that **don't need training**, then fit a small model (logistic regression) on top.

```
company history (5 numbers) ─────────────┐
document embedding (384 → PCA 16 numbers) ┼──▶ logistic regression ──▶ P(miss)
change-from-last-quarter (12 numbers)  ───┘   (dropped: no signal)
```

## 1. Change features (`change_features`)

Idea from finance research ("Lazy Prices", Cohen, Malloy & Nguyen 2020): companies mostly copy-paste
last quarter's report; when they substantially *rewrite* it, something has often changed — and that
has tended to precede bad news.

For every filing, compared with (a) the company's **previous** filing and (b) the filing **~1 year
earlier** (same fiscal quarter, so Q2-vs-Q2 structure matches):

| Feature | Meaning |
|---|---|
| `sim_*` | cosine similarity of word-count vectors (1 = identical wording) |
| `jaccard_*` | share of vocabulary in common |
| `log_len_ratio_*` | did the section get longer or shorter? |
| `neg_rate`, `unc_rate` | negative / uncertainty words per 1,000 words (small hand-made lists) |
| `d_neg_rate_prev`, `d_unc_rate_prev` | change in those rates vs previous filing |
| `has_prev`, `has_yoy` | 0 if there was no earlier filing to compare (values then filled with medians) |

**Result:** no signal here (test AUC 0.52). Possible reasons: the paper used multi-year horizons and
stock returns rather than next-quarter EPS; our word lists are tiny; 10-K years are missing, so
"previous filing" is sometimes 6 months back. A proper finance dictionary (Loughran-McDonald) is a
possible upgrade, but this is low priority.

## 2. Document embeddings (`document_embeddings`)

An **embedding** is a list of numbers representing a text's meaning; similar texts get nearby
vectors. We use `sentence-transformers/all-MiniLM-L6-v2` (22M params, 384-dim), frozen — never
trained by us.

1. Tokenise the whole MD&A and cut it into 256-token chunks (`return_overflowing_tokens=True`).
2. Embed each chunk: mean of the token vectors (ignoring padding), then L2-normalise.
3. Average all chunk vectors → one 384-dim vector per filing. **Reads 100% of the text.**
4. Cache to `embeddings_minilm.npz` (first run ~1.5 min on GPU for 661 filings; then instant).

**Result:** on its own, test AUC 0.70 — the best text-only model so far (FinBERT 0.52, TF-IDF 0.67).

## 3. Combining with history: why PCA

Putting 384 embedding numbers next to 5 history numbers in one logistic regression gave *no*
improvement: one regularisation strength `C` must suit both blocks, and the big block drowns the small
one. **PCA** (principal component analysis) compresses the 384 dims to the 16 directions along which
filings vary most (fit on training filings only). Now the blocks are similar in size.

| Model | Val AUC | Test AUC | Test 95% range |
|---|---|---|---|
| history | 0.734 | 0.728 | 0.61–0.83 |
| embed (384) | 0.645 | 0.703 | 0.58–0.82 |
| history + embed (384) | 0.645 | 0.743 | 0.62–0.85 |
| **history + embed PCA-16** | **0.752** | **0.752** | 0.64–0.85 |
| history + embed PCA-8 / PCA-32 | 0.749 / 0.699 | 0.749 / 0.727 | — |
| average of history & embed models | 0.660 | 0.718 | 0.60–0.82 |

The configuration is chosen by **validation** AUC (PCA-16), and the test score is just reported.
Picking whichever row looks best on test would be fitting to the test set.

## Honest reading

- Text embeddings and company history make fairly **different** predictions (correlation ≈ 0.39),
  which is why combining them helps.
- The gain (+0.02 AUC on both val and test) is consistent in direction but **smaller than the
  noise**: the 95% ranges overlap heavily because the test set has only 23 misses. Treat it as
  "promising", not "proven".
- To make it solid: more labelled data, or rolling evaluation over several test years.

## Concepts to learn here

Embeddings and mean pooling; chunking long documents; frozen feature extractors vs fine-tuning;
dimensionality reduction (PCA); regularisation across feature blocks; ensembles; bootstrap
confidence intervals; not selecting models on the test set.

## Within-company embeddings (added 2026-10-27)

`within_company_embeddings(df, emb_cols)` returns two versions of each embedding with the company's
"accent" removed: `resid` (minus the mean of the company's earlier filings) and `delta` (minus its previous
filing). Only earlier rows are used. Running mean via `cumsum` minus the current row, divided by
`cumcount`. Result: no signal, see [10-evaluation-deep-dive.md](10-evaluation-deep-dive.md) section 5.

## Guidance features (rule-based)

`guidance_features(docs)` reads only the sentences that mention *guidance / outlook / forecast*, skips
boilerplate, and labels each by the direction verb closest to that noun (within 40 characters): raise,
cut, maintain, narrow, initiate, withdraw. Per document it returns those counts, `g_net` =
sign(raise − cut), `has_outlook`, `n_outlook` and a hedging-word rate in the outlook sentences. About 3 s
for 2,000 press releases. This is the first text feature that transfers to unseen companies; results in
[12-press-releases.md](12-press-releases.md).
