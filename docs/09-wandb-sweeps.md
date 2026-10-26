# 09 — Hyperparameter sweeps with Weights & Biases

**File:** `src/models/sweep.py` (trains `fusion.py`'s late-fusion model)

## What a sweep is

Hyperparameters are the settings the model doesn't learn by itself: learning rate, layer sizes,
dropout, weight decay, batch size. A **sweep** tries many combinations and keeps the best.

With W&B:

1. `wandb.sweep(config)` registers the search space on the W&B server → returns a sweep id.
2. `wandb.agent(sweep_id, fn, count=50)` loops 50 times: ask the server for the next config, run
   `fn` (which reads it from `wandb.config` and trains), report the metric back.
3. `method: bayes` — the server fits a model of "config → score" to the runs so far and proposes
   configs likely to score well. It *exploits*: once a region looks good, most later runs go there.

The dashboard shows every run's curves, a parallel-coordinates plot and parameter importance.

## Two rules that matter more than the tool

| Rule | Why |
|---|---|
| **Score = mean val AUC over 3 seeds** | One seed's val AUC moves ±0.03 from the random seed alone — as much as most hyperparameter changes. A one-seed score would mostly reward luck |
| **The sweep never touches test** | The best of 50 configs on a 21-miss val set is chosen *because* it scored high, so its val score is optimistic ("winner's curse"). Test is run **once**, on the winner only |

## Search space (50 runs, Bayesian)

| Parameter | Range |
|---|---|
| `lr` | 1e-4 – 1e-2 (log scale) |
| `weight_decay` | 1e-4 – 0.3 (log scale) |
| `history_dim` (history branch output) | 4, 8, 16 |
| `embed_dim` (text branch output) | 4, 8, 16, 32 |
| `hidden` | 8, 16, 32, 64 |
| `dropout`, `input_dropout` | 0 – 0.6, 0 – 0.5 |
| `batch_size` | 16, 32, 64 |

```bash
wandb login                                               # once per machine
python src/models/sweep.py --data-root . --count 50       # ~5 min
python src/models/sweep.py --best ENTITY/earnings-lie-detector/SWEEP_ID   # top 5 + fusion.py command
python src/models/sweep.py --data-root . --dry-run        # one run, W&B off
```

## Results (2026-10-26, sweep `0zsaf6qz`)

Val AUC over the 50 runs: min 0.635, median 0.695, max 0.747.

**What the sweep learned** (Spearman correlation of each setting with val AUC):

| Setting | Corr. | Reading |
|---|---|---|
| `embed_dim` | **−0.75** | Squeeze the text to *few* features. Mean val: 4 dims 0.717, 8 dims 0.685, 16–32 dims ~0.66 |
| `history_dim` | +0.53 | Give history *more* room: 16 dims 0.710 vs 4 dims 0.664 |
| `lr` | +0.52 | Higher LR (~2e-3 – 8e-3) did better than the 1e-3 default |
| `input_dropout` | +0.46 | Heavier dropout on the 384 text inputs helps |
| `weight_decay` | +0.29 | Stronger regularisation helps a little |
| `batch_size`, `hidden` | ~0 | Don't matter |

All top-5 configs have text → **4**, history → **16**: *lean on history, use text lightly and heavily
regularised.* That's the same lesson as PCA-16 in the baselines, found automatically.

**Winner tested once** (5 seeds, `checkpoints/fusion/late_swept`):

| Model | Val AUC | Test AUC | Test F1 |
|---|---|---|---|
| Late fusion, defaults (Task 2) | 0.686 ± 0.027 | 0.734 ± 0.019 | 0.337 |
| **Late fusion, swept** | **0.734 ± 0.028** | **0.742 ± 0.032** | 0.361 |
| *LR history only* | *0.734* | *0.728* | *0.386* |
| *LR history + embed PCA-16* | *0.752* | *0.752* | *0.351* |

## What this tells us

- **Winner's curse, visible:** the winner scored val 0.747 in the sweep (3 seeds) but 0.734 on a
  fresh 5-seed rerun — the extra seed 46 got 0.682. The sweep's top score was partly luck.
- **Tuning helped val a lot (+0.05) but test only a little (+0.01).** Part of the val gain is the
  sweep fitting the 2024 val set; the real improvement is small.
- **The tuned network ≈ logistic regression on history**, and still just below LR history + PCA-16.
  On 370 training rows, a neural model doesn't add anything a well-regularised linear model can't do.
- **Task 2's seed-ensemble test AUC of 0.784 was luck:** with tuned settings the ensemble scores 0.745.
