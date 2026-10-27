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

## Fine-tuning cheaply: LoRA

Full fine-tuning updates all ~110M FinBERT weights; with 370 training examples that overfits fast.
**LoRA** (Low-Rank Adaptation) freezes the original weights and adds a small trainable "adapter"
next to chosen weight matrices:

```
W x  →  W x + (B A) x        W: 768×768 frozen     A: r×768, B: 768×r trainable, r = 8
```

`B A` is a rank-8 correction to `W`, at 2 × 768 × 8 ≈ 12k parameters instead of 590k per matrix. `B` starts at
zero, so training begins from exactly the pre-trained model. We put adapters on the attention
`query` and `value` matrices in all 12 layers (`peft` library): **393,602 trainable parameters, 0.36%
of the model**.

```bash
python src/models/train.py --data-root . --lora-r 8 --lr 2e-4 --head-lr 2e-4 --epochs 10
```

LoRA needs a higher LR (2e-4) than full fine-tuning (2e-5) because the adapters start at zero and are
small. Checkpoints store only trainable weights (~1.6 MB instead of ~440 MB).

## Reading the whole filing: chunking + gradient accumulation

A 512-token window sees ~3% of an MD&A. With `--max-chunks N` the tokenizer cuts each document into
consecutive 512-token chunks (`return_overflowing_tokens=True`) and keeps up to N of them, evenly spaced
from start to end. Each chunk is encoded separately, and the chunk [CLS] vectors are **mean-pooled per
document** (`index_add_` by `doc_ids`) before the head.

```bash
python src/models/train.py --data-root . --lora-r 8 --lr 2e-4 --head-lr 2e-4 \
    --max-chunks 16 --batch-size 1 --grad-accum 8 --seeds 42 43 44
```

16 chunks × 512 tokens through FinBERT with gradients only fits one document at a time on our RTX A1000.
**Gradient accumulation** fakes a bigger batch: `.backward()` *adds* to each parameter's `.grad`, so
we call it on 8 one-document batches and only then do `optimizer.step()` + `zero_grad()`.
Effective batch = `batch_size × grad_accum` = 8.

**A subtle bug this exposed:** `CrossEntropyLoss(weight=..., reduction="mean")` divides by the *sum
of the batch's class weights*. With one document per batch that cancels the weight completely: a
miss and a beat contribute equally, and class weighting silently does nothing. The fix: use
`reduction="sum"` and divide by the fixed effective batch size.

## Results (temporal split, test = 2025–26, 23 misses)

| Setup | Val AUC | Test AUC | Test F1 |
|---|---|---|---|
| Frozen FinBERT, first 512 tokens (5 seeds) | 0.59 | 0.515 ± 0.03 | 0.12 |
| LoRA r=8, first 512 tokens (5 seeds) | 0.598 ± 0.034 | 0.525 ± 0.029 | 0.17 |
| Frozen FinBERT, 16 chunks (3 seeds) | 0.667 ± 0.003 | 0.603 ± 0.011 | 0.07 |
| LoRA r=8, 16 chunks (3 seeds) | 0.639 ± 0.004 | 0.611 ± 0.020 | 0.12 |
| *Frozen MiniLM, whole filing + logistic regression* | *0.645* | *0.703* | *0.31* |

- **Truncation was the bottleneck.** Reading the whole filing lifts FinBERT from chance (0.52) to 0.60.
  LoRA on 512 tokens didn't help: the first 512 tokens are mostly boilerplate, and val AUC peaked at
  epoch 1 and then fell as it memorised them.
- **Fine-tuning adds nothing** over a frozen encoder (0.611 vs 0.603), and both trail a frozen MiniLM
  embedding + logistic regression (0.70). With ~370 training documents, end-to-end fine-tuning of a
  110M-parameter model can't beat good frozen features plus a small model. The project continued in
  that direction: [07-text-features.md](07-text-features.md), [08-fusion-model.md](08-fusion-model.md).
- Test F1 is low because these models rarely score a document above 0.5. See
  [10-evaluation-deep-dive.md](10-evaluation-deep-dive.md) for thresholds.

## Concepts to learn here

Transfer learning; tokenisation and context-length limits; fine-tuning vs feature extraction;
learning-rate schedules; regularisation (dropout, weight decay, early stopping); seed variance;
parameter-efficient fine-tuning (LoRA); long-document chunking and pooling; gradient accumulation;
how loss reduction interacts with class weights.
