# Earnings Lie Detector

**Can the way management writes about the business predict that the company will miss its next
earnings estimate?**

This project reads US companies' quarterly SEC filings (the 10-Q MD&A) and earnings press releases and
predicts whether the company will **miss the analyst consensus EPS estimate** at its *next* earnings
announcement, about 90 days later.

**Answer (188 companies, 2,816 test announcements, 469 misses):** past earnings surprises plus sector
reach **AUC ≈ 0.67**; the text adds at most +0.006 on top, not significantly different from zero.
The findings below are in the order we found them, including early gains that didn't survive more data.

## Findings

- **The track record predicts, the language mostly doesn't.** Over test years 2023–26 on the first 49
  companies (403 announcements, 58 misses), the best models reach **AUC ≈ 0.70**. Flagging the riskiest 10% of announcements finds
  misses at **~2.7× the 14% base rate**. Almost all of that comes from the company's own recent earnings
  surprises.
- **Adding the filing text** to that history improves AUC by +0.016, with a 95% paired-bootstrap interval
  of [−0.02, +0.05], so we can't rule out zero.
- **The text signal is company identity.** With companies held out of training, a text-only model
  drops from AUC 0.67 to 0.48 (chance). It had learned *which company* wrote the filing, not *how* it
  was written.
- **Earnings press releases do a little better than the 10-Q.** 2,115 releases (8-K, Exhibit 99.1) give
  1,848 samples. Text alone keeps AUC 0.58 on unseen companies (10-Q: 0.48), but the gain over history
  (+0.023, P(Δ>0)=0.84) is still not significant.
- **What management says about guidance carries real signal.** Simple rules detect raised / cut /
  maintained / withdrawn guidance. Added to history, they give the project's first significant gain:
  +0.033 AUC on unseen companies (95% interval [+0.004, +0.062]). Over time the gain is +0.01–0.03 and
  not significant. Releases that cut guidance precede a miss 44% of the time, raises only 6%.
- **An LLM reading the guidance does no better than the rules.** `gpt-oss-20b` (free NVIDIA API) scores
  EPS / revenue direction, size and tone for 1,367 releases: +0.025 on unseen companies (P 0.94) vs the
  rules' +0.033. The bottleneck is the number of misses, not the extraction.
- **With 4× the companies (188, 1,161 misses), the guidance gain mostly disappears.** It shrinks to
  +0.012 on unseen companies, and most of it was a hidden **sector** effect: whether a company publishes
  an outlook depends on its industry. On top of history + sector, text adds ≤ +0.006 AUC (not significant).
  Best model: history + sector + guidance, AUC 0.676. Early gains on 49 companies were small-sample luck.
- **The market says more than the text.** Analyst price-target changes and the stock's run-up in the 90
  days before the announcement (point-in-time from Yahoo) add +0.019 [+0.011, +0.028] AUC over history +
  sector on unseen companies, about 5× the text. Over time the gain is +0.005: it helps in 2023–24 but not
  in 2025–26. Best pooled model: history + sector + analyst + price + guidance, AUC 0.680.
- **Reading the whole document matters more than fine-tuning.** FinBERT on the first 512 tokens scores
  at chance. Reading 16 chunks of the filing lifts it to 0.60, and LoRA fine-tuning adds nothing on top.
  A frozen MiniLM embedding of the full text with logistic regression does better (0.70 on 2025–26).
- **A single small test window can mislead.** The 2025–26 window alone (23 misses) suggested 0.75; a
  sweep's best run looked better than it was on rerun (winner's curse); a seed ensemble's 0.78 turned
  out to be luck.

## Approach

| Stage | Details |
|---|---|
| Data | 3,193 10-Q MD&A sections and 8,313 earnings press releases (8-K Ex. 99.1) from SEC EDGAR (188 S&P 500 companies across 10 sectors, REITs excluded, 2016–2026); 7,089 EPS actual vs estimate from Yahoo Finance. Each filing is paired with the **next** announcement, so the text never contains the answer |
| Split | By time: train 2020–23, validate 2024, test 2025–26. Plus rolling-origin folds testing 2023, 2024 and 2025–26 |
| Models | FinBERT (frozen, LoRA, whole-filing chunking) · TF-IDF and MiniLM embeddings with logistic regression · history features · a PyTorch late-fusion network tuned with a W&B Bayesian sweep |
| Evaluation | AUC and F1 (14% positives), bootstrap intervals, paired bootstrap for model differences, company-held-out cross-validation |

The best model (logistic regression on past surprises, sector, analyst actions, price run-up and guidance
rules; 188 companies; pooled AUC 0.68) is published on the Hugging Face Hub:
[Guruprasath12/lie-detect](https://huggingface.co/Guruprasath12/lie-detect). It takes a company's past EPS
surprises, its sector, its latest press release text, and fetches analyst and price data from Yahoo
Finance. The model card covers its limitations, including no gain over history + sector in 2025–26.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/data/universe.py                                        # 49 original + 150 S&P 500 companies → data/universe.csv
python src/data/scraper.py --tickers $(python src/data/universe.py --list all)  # SEC EDGAR → data/raw/transcripts/
python src/data/labeler.py --tickers $(python src/data/universe.py --list all)  # yfinance → data/processed/labels.csv
python src/models/baselines.py --data-root .                       # logistic-regression models, ~1 min
python src/models/fusion.py --data-root .                          # neural fusion, 5 seeds, ~5 s
python src/models/evaluate.py --data-root .                        # multi-year evaluation, ~1 min
python src/data/press_releases.py --tickers $(python src/data/universe.py --list all)  # 8-K press releases, ~3 h
python src/models/llm_guidance.py                                  # optional: LLM guidance scores (NVIDIA key), hours
python src/models/evaluate.py --data-root . --text press           # same evaluation on press releases
python src/models/train.py --data-root . --freeze-encoder          # FinBERT, ~7 min on a laptop GPU
```

## Docs

Step-by-step learning notes are in [`docs/`](docs/). Start with
[`docs/00-overview.md`](docs/00-overview.md), then one doc per component, from scraping to publishing
on the Hub.

Educational project. Not investment advice.
