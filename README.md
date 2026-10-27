# Earnings Lie Detector

**Can the way management writes about the business predict that the company will miss its next
earnings estimate?**

This project reads the MD&A section of US companies' quarterly SEC filings (10-Q) and predicts whether
the company will **miss the analyst consensus EPS estimate** at its *next* earnings announcement,
about 90 days later.

## Findings

- **The track record predicts, the language mostly doesn't.** Over test years 2023–26 (403 announcements,
  58 misses), the best models reach **AUC ≈ 0.70**. Flagging the riskiest 10% of announcements finds
  misses at **~2.7× the 14% base rate**. Almost all of that comes from the company's own recent earnings
  surprises.
- **Adding the filing text** to that history improves AUC by +0.016, with a 95% paired-bootstrap interval
  of [−0.02, +0.05], so we can't rule out zero.
- **The text signal is company identity.** With companies held out of training, a text-only model
  drops from AUC 0.67 to 0.48 (chance). It had learned *which company* wrote the filing, not *how* it
  was written.
- **Reading the whole document matters more than fine-tuning.** FinBERT on the first 512 tokens scores
  at chance. Reading 16 chunks of the filing lifts it to 0.60, and LoRA fine-tuning adds nothing on top.
  A frozen MiniLM embedding of the full text with logistic regression does better (0.70 on 2025–26).
- **A single small test window can mislead.** The 2025–26 window alone (23 misses) suggested 0.75; a
  sweep's best run looked better than it was on rerun (winner's curse); a seed ensemble's 0.78 turned
  out to be luck.

## Approach

| Stage | Details |
|---|---|
| Data | 664 10-Q MD&A sections from SEC EDGAR (49 large US companies, 2020–2026); EPS actual vs estimate from Yahoo Finance. Each filing is paired with the **next** announcement, so the text never contains the answer |
| Split | By time: train 2020–23, validate 2024, test 2025–26. Plus rolling-origin folds testing 2023, 2024 and 2025–26 |
| Models | FinBERT (frozen, LoRA, whole-filing chunking) · TF-IDF and MiniLM embeddings with logistic regression · history features · a PyTorch late-fusion network tuned with a W&B Bayesian sweep |
| Evaluation | AUC and F1 (14% positives), bootstrap intervals, paired bootstrap for model differences, company-held-out cross-validation |

The tuned fusion model is published on the Hugging Face Hub as `Guruprasath12/lie-detect`, with a
model card covering its limitations. The repo is currently private.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/data/scraper.py                                         # SEC EDGAR → data/raw/transcripts/
python src/data/labeler.py --transcripts-dir data/raw/transcripts  # yfinance → data/processed/labels.csv
python src/models/baselines.py --data-root .                       # logistic-regression models, ~1 min
python src/models/fusion.py --data-root .                          # neural fusion, 5 seeds, ~5 s
python src/models/evaluate.py --data-root .                        # multi-year evaluation, ~1 min
python src/models/train.py --data-root . --freeze-encoder          # FinBERT, ~7 min on a laptop GPU
```

## Docs

Step-by-step learning notes are in [`docs/`](docs/). Start with
[`docs/00-overview.md`](docs/00-overview.md), then one doc per component, from scraping to publishing
on the Hub.

Educational project. Not investment advice.
