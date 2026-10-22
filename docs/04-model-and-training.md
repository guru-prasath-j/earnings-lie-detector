# 04 — Model and training

**Files:** `src/models/model.py` (`EarningsTextClassifier`), `src/models/train.py`

## The model

```
MD&A text
  → tokenizer (first 512 tokens)
  → FinBERT encoder (12 transformer layers, 768-dim)
  → [CLS] vector (one 768-dim summary of the text)
  → Dropout 0.3 → Linear 768→128 → ReLU → Dropout 0.15 → Linear 128→2
  → logits for [beat, miss]
```

- **FinBERT** (`ProsusAI/finbert`) is BERT further trained on financial text, so it already
  "knows" finance vocabulary. If it can't be downloaded, `distilbert-base-uncased` is used.
- **[CLS] pooling**: BERT prepends a special `[CLS]` token; its output vector is the standard
  whole-text representation for classification.
- **The head** is a small 2-layer network that turns that vector into a decision. Dropout randomly
  zeroes values during training to reduce overfitting — important with only 370 training examples.

### How much of the encoder to train

| Option | What trains | When |
|---|---|---|
| `--freeze-encoder` | head only (~100k params) | fast baseline, least overfitting |
| `--freeze-n-layers 8` | head + top 4 layers | middle ground |
| (neither) | everything (~110M params) | most flexible, needs a low LR, overfits easily |

## Training loop (`train.py`)

```bash
python src/models/train.py --data-root . --freeze-encoder --epochs 10 --batch-size 8
python src/models/train.py --data-root . --epochs 15 --lr 2e-5
python src/models/train.py --data-root . --freeze-n-layers 8 --epochs 15 --seeds 42 43 44
```

Per epoch: tokenise a batch → forward → loss → backward → optimizer step, then evaluate on val.

| Piece | What / why |
|---|---|
| **Weighted cross-entropy** | Misses weighted ~6× (from train split) so the model can't win by always saying "beat" |
| **AdamW**, head LR = 10 × encoder LR | The new head starts random and must learn fast; pre-trained layers need only small nudges |
| **Warmup → cosine decay** | LR ramps up over the first 10% of steps (avoids wrecking pre-trained weights early), then smoothly decays |
| **Mixed precision (AMP)** | On GPU, computes in 16-bit where safe: faster, less memory |
| **Gradient clipping (1.0)** | Caps update size to avoid unstable steps |
| **Early stopping on val F1** (patience 3) | Stops when val F1 hasn't improved for 3 epochs; F1, not loss, because loss can look fine while recall is zero |
| **Multiple seeds** (default 5) | Each seed changes head initialisation, shuffling and dropout; we report mean ± std |

## Outputs

```
checkpoints/
  summary.json              # mean ± std of val and test metrics across seeds, plus per-run results
  seed_42/best_model.pt     # weights from the best-val-F1 epoch
  seed_42/training_history.json   # per-epoch train loss and val metrics
```

## Current result and why

Frozen FinBERT, temporal split, 5 seeds: **test AUC 0.515 ± 0.03, F1 0.12** — no better than chance.
A full-text TF-IDF model gets AUC 0.67 on the same data, so the signal exists. Likely causes:

1. **Truncation** — an MD&A is 5k–20k tokens; we read the first 512 (~3%), which is mostly the
   section heading and boilerplate ("This Item should be read in conjunction with…").
2. **Frozen encoder** — FinBERT was trained for sentiment of single sentences; its [CLS] vector for
   a long document isn't tuned for this task unless we fine-tune.

Fixes to try: chunk the document into 512-token windows and average/max-pool chunk outputs; skip the
boilerplate opening; fine-tune the top layers.

## Concepts to learn here

Transfer learning; tokenisation and context-length limits; fine-tuning vs feature extraction;
learning-rate schedules; regularisation (dropout, weight decay, early stopping); seed variance.
