# 14 — Analyst actions and price run-up

**Files:** `src/data/market_data.py` → `data/raw/market/`; `src/models/market_features.py`;
`build_table` joins the features; `evaluate.py` adds the `analyst` and `price` blocks.

## Why

After step 13, the signal was clearly in the company's track record and sector, not in its text. The
finance literature suggests looking at what happens *before* the announcement: analysts revising
their estimates, how much they disagree, and how the stock moves.

## The point-in-time problem

For a 2021 announcement we may only use what was known in 2021. Yahoo Finance offers:

| Data | History? | Used |
|---|---|---|
| EPS estimate, low / high (dispersion), number of analysts | today's snapshot only | no: it would leak the future |
| EPS trend 7 / 30 / 60 / 90 days ago, revisions up / down | relative to today only | no |
| **Analyst rating actions** (`upgrades_downgrades`) | timestamped since ~2012 | **yes** |
| **Daily prices** | full history | **yes** |

Real estimate-revision history isn't free. Price-target changes are the closest proxy: an analyst who
cuts a target usually cuts the estimate too.

## Features

From the 90 days before the announcement (`ANALYST_COLS`):
- `an_n`: log(1 + number of rating actions).
- `an_net_grade`: upgrades − downgrades.
- `an_pt_net`: (targets raised − lowered) / target changes.
- `an_pt_chg`: mean % change of the price target.

From prices (`PRICE_COLS`):
- `px_rel_63` and `px_rel_21`: ~3-month and 1-month return minus SPY's, clipped.
- `px_vol_63`: volatility of daily returns.

**Leakage guard:** every window ends **2 days before** Yahoo's announcement date. Step 10 showed that
date can be off by a day or two, and an analyst cutting a target the evening of the real release must
not count.

**Simple tables first** (base rate 16.8%):

| Next-quarter miss rate | Miss rate |
|---|---|
| Net price-target **cuts** in the last 90 days | **22.3%** |
| No target changes | 17.4% |
| Net price-target **raises** | **13.7%** |
| Stock in the bottom fifth by 3-month return vs SPY | **22.2%** |
| Other four fifths | 14.5–17.9% |

Upgrades vs downgrades show no clear pattern (18.6% / 15.6% / 17.5%).

## Results (press-release rows, 188 companies)

| Added to history + sector | Over time (rolling-origin) | Unseen companies |
|---|---|---|
| guidance rules (text) | +0.006 [−0.004, +0.017] | +0.004 [−0.003, +0.011] |
| analyst actions | +0.005 [−0.006, +0.016] | **+0.011 [+0.004, +0.017]** |
| price run-up | +0.001 [−0.011, +0.013] | **+0.013 [+0.006, +0.020]** |
| analyst + price | +0.005 [−0.009, +0.020] | **+0.019 [+0.011, +0.028]** |
| analyst + price + guidance | +0.010 [−0.006, +0.027] | **+0.020 [+0.010, +0.030]** |

Best pooled model: history + sector + analyst + price + guidance, **AUC 0.680** (history alone 0.655).

### Checking the unseen-company gain

Company-held-out CV ignores time, so a model could learn from *other companies in the same quarter*: a
market-wide shock (early 2020) moves prices and causes misses everywhere at once. To test this, each
test row was predicted by a model trained **without any rows from its own year**. The gain didn't
move (+0.019 [+0.011, +0.028]), so it isn't a same-quarter artefact. The market features carry signal
that transfers across companies, about 5× what the text added.

### Over time, it depends on the year

| Test year | history + sector | + analyst + price |
|---|---|---|
| 2023 | 0.670 | 0.689 (+0.019) |
| 2024 | 0.658 | 0.670 (+0.012) |
| 2025–26 | 0.673 | 0.670 (−0.003) |

The features help in 2023 and 2024 and not in 2025–26, the largest test period, so the pooled gain
is +0.005 and not significant. With three test years we can't tell a real change in how markets
behave from noise.

### Same check on the MD&A rows

Over history + sector: analyst + price **+0.005** over time (P 0.66) and **+0.021 [+0.007, +0.033]** on
unseen companies; price alone +0.018 [+0.008, +0.028]. The same pattern, but not independent evidence:
the market features describe the *announcement*, and the MD&A rows are largely the same announcements
paired with a different document.

## What this means

- **The strongest add-on found so far,** and it is not text. Analyst actions and price moves are
  about 5× more useful than anything read from the press release, on companies the model hasn't seen.
- **Not yet proven going forward:** the most recent period shows no gain. The bar (beat the baseline
  over time *and* on unseen companies) is half met, like guidance on 49 companies, but here the
  cross-company result is robust to the same-year check and rests on 1,161 misses.
- **Published anyway, with the caveat up front:** it won clearly on validation (mean 0.692 vs 0.677),
  so the Hub model now includes these features ([11](11-huggingface-hub.md), version 3), and its card
  shows the year-by-year table.
- **What would settle it:** real estimate-revision history (paid data: I/B/E/S, FactSet, Zacks), or
  another year of test data.
