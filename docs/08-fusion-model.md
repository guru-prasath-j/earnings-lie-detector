# 08 — Fusion model and a hand-written training loop

**File:** `src/models/fusion.py` (features and split shared with `baselines.py` via `build_table` /
`temporal_split`)

## The idea

We have two kinds of evidence about each upcoming announcement:

| Block | Size | What it is |
|---|---|---|
| `history` | 5 numbers | The company's own track record: missed last quarter? last surprise %, 4-quarter miss rate / mean / std of surprise |
| `embed` | 384 numbers | MiniLM embedding of the whole MD&A (see [07-text-features.md](07-text-features.md)) |

**Fusion** = one model that uses both. The logistic-regression baselines already tried this by gluing
the columns together; here we do it with a small neural network, which can learn its own compressed
features from each block.

### Early vs late fusion

```
early:  [history ; embed] (389) ──> MLP ──> logit

late:   history (5)   ──> Linear ──> ReLU ──> 8   ┐
                                                  ├── concat (24) ──> MLP ──> logit
        embed   (384) ──> Dropout ──> Linear ──> ReLU ──> 16 ┘
```

- **Early fusion** concatenates the raw inputs. Simple, but 384 text numbers vs 5 history numbers:
  the network's first layer is dominated by text, and the strong-but-small history signal gets lost.
- **Late fusion** gives each block its own small *branch* that squeezes it to a few learned features
  first (text 384 → 16, history 5 → 8). Both arrive at the head at a similar size, so neither can
  drown the other. This is the neural version of "PCA the embedding to 16 dims" from the baselines.
- The output is **one logit**; `sigmoid(logit)` = P(miss). Loss is `BCEWithLogitsLoss` with
  `pos_weight = #beats / #misses` (~6.7), the binary equivalent of the class-weighted cross-entropy
  used in `train.py`.

## The manual training loop

No `DataLoader`, no Trainer. 661 samples × 389 floats fit on the GPU as plain tensors, so an epoch is:

```python
perm = torch.randperm(n)                    # shuffle row order
for start in range(0, n, batch_size):
    idx = perm[start:start + batch_size]
    logits = model(x[idx])                  # 1. forward
    loss = criterion(logits, y[idx])        # 2. loss
    optimizer.zero_grad()                   # 3. clear old gradients (.backward ADDS to .grad)
    loss.backward()                         # 4. compute gradients
    optimizer.step()                        # 5. update weights
```

Then, every epoch:

| Step | Why |
|---|---|
| `model.eval()` + `torch.no_grad()` for prediction | Turns dropout off and skips gradient bookkeeping |
| Early stopping on **val AUC**, patience 20 | AUC is smooth and threshold-free; val F1 flips 0 ↔ 0.35 between epochs |
| `copy.deepcopy(model.state_dict())` at the best epoch | `state_dict()` returns *references* to live tensors — without the copy, the "snapshot" keeps changing as training continues |
| `load_state_dict(best_state)` at the end | Evaluate the best epoch, not the last one |
| 5 seeds + **seed ensemble** | Each seed changes init, shuffling and dropout. Averaging the 5 models' probabilities is a cheap ensemble |

```bash
python src/models/fusion.py --data-root .                    # late fusion (default), 5 seeds, ~5 s
python src/models/fusion.py --data-root . --fusion early
python src/models/fusion.py --data-root . --blocks history   # single branch
python src/models/fusion.py --data-root . --blocks embed
```

Output: `checkpoints/fusion/<config>/summary.json` (per-seed metrics + training curves, mean ± std,
seed ensemble with 95% bootstrap AUC range).

## Results (2026-10-26)

Temporal split, test = 2025–2026 (169 samples, 23 misses). Per-seed = mean ± std over 5 seeds.

| Model | Val AUC | Test AUC (per seed) | Test AUC (seed ensemble) | Test F1 (per seed) |
|---|---|---|---|---|
| MLP, history only | 0.722 ± 0.011 | 0.669 ± 0.011 | 0.670 | 0.353 |
| MLP, embed only | 0.655 ± 0.017 | 0.695 ± 0.015 | 0.707 | 0.279 |
| Early fusion | 0.654 ± 0.014 | 0.732 ± 0.020 | 0.730 | 0.391 |
| **Late fusion** | **0.686 ± 0.027** | 0.734 ± 0.019 | 0.784 [0.68–0.87] | 0.337 |
| *LR: history only* | *0.734* | *0.728* | — | *0.386* |
| *LR: history + embed PCA-16* | *0.752* | *0.752* | — | *0.351* |

## What this tells us

- **Late beats early fusion on val** (0.686 vs 0.654). Early fusion's val AUC equals the embed-only
  model's: the history signal really is drowned when raw inputs are concatenated.
- **The MLP is worse than logistic regression on history alone** (val 0.722 vs 0.734, test 0.669 vs
  0.728). Five features and 370 rows don't need hidden layers: LR with very strong regularisation
  (C = 0.001) is nearly a fixed weighted sum, and the MLP mostly adds variance.
- **By the rule we use (pick on val), LR history + PCA-16 is still the best model** (val 0.752 vs 0.686).
  The neural fusion model's test numbers look comparable or better, but we can't choose on test.
- **The seed ensemble jumps on test (0.734 → 0.784) but not on val (0.686 → 0.686).** An improvement
  that shows up on only one of the two held-out sets is most likely luck with 23 test misses. The
  bootstrap range [0.68–0.87] overlaps every model above.
- Late fusion's best epoch is usually 1–5: with so little data, it overfits within a few passes.
  Learning rate, branch sizes, dropout and weight decay are untuned — that's what the W&B sweep
  (next task) is for.
