# Earnings Lie Detector — project context

Predict whether a company will **miss its next earnings EPS estimate** (label 1) or meet/beat it (label 0)
from the MD&A section of its SEC 10-Q filings. Text-only for now; audio is optional and off.

## Keep this file current (standing instruction)

This file is the hand-off between machines and sessions. After any meaningful change — new/changed
scripts, a pipeline run, training results, a design decision, a new known issue — update the relevant
section below and the **Log** at the bottom (date, what changed, results). Keep it short and factual;
replace stale info rather than appending contradictions. Commit it together with the code it describes.

Learning docs live in `docs/` (`00-overview.md` = high-level + project journey, `01`–`11` = one per component). `README.md` = GitHub-facing summary. The user
reads them to learn — when an implementation or result changes, update the matching doc too.

## Setup on a new machine

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`data/raw/`, `data/processed/`, `*.pt` and `checkpoints/` are gitignored — regenerate them:

```bash
python src/data/scraper.py                                         # → data/raw/transcripts/{TICKER}.json (~664 filings, 49 tickers)
python src/data/labeler.py --transcripts-dir data/raw/transcripts  # → data/processed/labels.csv (~1,899 rows)
python src/data/dataset.py                                         # sanity check: 661 samples, 92 misses
python src/models/train.py --data-root . --freeze-encoder --epochs 10 --batch-size 8   # 5 seeds, ~7 min on RTX A1000
python src/models/baselines.py --data-root .                                          # LR yardsticks; first run builds MiniLM cache (~1.5 min)
python src/models/fusion.py --data-root .                                             # neural late fusion, 5 seeds, ~5 s
python src/models/evaluate.py --data-root .                                           # rolling-origin deep dive → checkpoints/evaluation.json, ~1 min
python src/data/press_releases.py                                                     # → data/raw/press_releases/{TICKER}.json (2,115 releases, 50 tickers), ~50 min
python src/models/evaluate.py --data-root . --text press                              # → checkpoints/evaluation_press.json (first run embeds, ~1.5 min)
```

## Pipeline

| File | Role |
|---|---|
| `src/data/scraper.py` | SEC EDGAR 10-Q MD&A text → flat `data/raw/transcripts/{TICKER}.json` (list of records: `company`, `date`=filing date, `quarter`, `form`, `mda_text`, …) plus combined `_all_transcripts.json` |
| `src/data/press_releases.py` | 8-K Item 2.02 earnings press releases: submissions API (all pages — banks' "recent" covers ~1 yr) → `-index.htm` Type `EX-99.1` (fallback filename `ex991`, any `EX-99.*`) → text without tables (kept if narrative < 1,500 chars), SGML header cut. Reuses `scraper.py` helpers. Records `date`/`text` like MD&A → `data/raw/press_releases/{T}.json`. Default 2016-07-01 → today. XOM 1 release, BLK 9 |
| `src/data/labeler.py` | Default `--method direct`: yfinance `get_earnings_dates` (history back to ~2014, `lookback_quarters=40`) → one row per earnings announcement; `label = 1 if eps_actual < eps_estimate`. `date` = announcement date. Reads both `{T}.json` and `{T}/*.json` layouts, skips `_*` files |
| `src/data/dataset.py` | `EarningsCallDataset`. Pairs each label (announcement date) with the latest filing **strictly before** it within 120 days (`max_transcript_gap_days`) and at least `min_transcript_gap_days` (default 0; 30 for press releases) before it. `transcripts_dir` picks the text source. `use_audio=False` by default → text-only, drops labels with no filing, batches have no audio keys. `get_dataloaders(split="temporal")` default: train < 2024-01-01 (370, 48 miss), val 2024 (122, 21), test ≥ 2025-01-01 (169, 23); `split="random"` still available |
| `src/data/audio_downloader.py`, `audio_features.py` | Optional audio pipeline (Earnings-21/22 HF datasets → ~20–28 librosa features). Not run here; won't join to labels (benchmark calls ≠ our tickers/dates). Downloads ~20–35 GB if run without `--max-calls` |
| `src/models/model.py` | `EarningsTextClassifier`: `ProsusAI/finbert` (fallback `distilbert-base-uncased`) → [CLS] → Dropout(0.3) → Linear(768→128) → ReLU → Dropout(0.15) → Linear(128→2). `freeze_encoder` / `freeze_n_layers` / LoRA (`lora_r>0`: peft adapters on query+value, r=8 → 393,602 trainable = 0.36%). `max_chunks>1`: whole filing → 512-token chunks, up to N evenly spaced, [CLS] mean-pooled per doc (`doc_ids`) |
| `src/models/train.py` | Class-weighted CE (weights from train split), AdamW (head LR = 10× encoder LR), warmup→cosine, AMP on CUDA, early stop on val F1 (patience 3). `--seeds` (default 42–46) → `checkpoints/seed_N/{best_model.pt,training_history.json}` + `checkpoints/summary.json` (mean ± std). `--split temporal|random`, `--val-start`, `--test-start`, `--lora-r/--lora-alpha/--lora-dropout`, `--head-lr`, `--max-chunks`, `--grad-accum` (loss uses `reduction="sum"` / fixed effective batch so class weights survive batch-size 1). Checkpoints hold trainable weights only (~1.6 MB) |
| `src/models/text_features.py` | `change_features` (cosine/Jaccard/length vs previous and ~1-year-earlier filing, neg/uncertainty word rates) and `within_company_embeddings` (`resid` = emb − mean of company's earlier rows, `delta` = emb − previous row; first row 0) and `guidance_features` (outlook sentences → closest direction verb within 40 chars: raise/cut/maintain/narrow/initiate/withdraw counts, `g_net`, `has_outlook`, `n_outlook`, `outlook_hedge_rate`; `GUIDANCE_COLS`; ~85% precision on spot checks, tuned without labels) and `document_embeddings` (whole MD&A → 256-token chunks → frozen `all-MiniLM-L6-v2` mean-pooled → avg, 384-dim; cached in `data/processed/embeddings_minilm.npz`, ~1.5 min on GPU) |
| `src/models/fusion_net.py` | Self-contained (torch + huggingface_hub) `FusionNet` and `HubFusionNet` (`PyTorchModelHubMixin`; explicit `blocks` list because config.json sorts keys; train scaler mean/std as buffers → takes raw features; `predict_proba(history, embedding)`) |
| `src/models/push_to_hub.py` | Trains swept fusion (seed 42, CPU, `evaluate.FUSION_CFG`), wraps in `HubFusionNet`, `save_pretrained` → `checkpoints/hub/earnings-miss-fusion/` + model card + `fusion_net.py`, asserts identical preds after `from_pretrained`. Uploads only with `--push --repo-id` (private unless `--public`); needs `hf auth login` |
| `src/models/fusion.py` | `FusionNet` (imported from `fusion_net.py`): history (5) and MiniLM embed (384) blocks; `--fusion late` (per-block Linear+ReLU branches 5→8, 384→16, concat → MLP → 1 logit) or `early` (raw concat → MLP); `--blocks` for single-branch. Hand-written loop (no DataLoader): BCEWithLogits + `pos_weight`, AdamW 1e-3, early stop on val AUC (patience 20), best `state_dict` deepcopy, 5 seeds + seed-ensemble → `checkpoints/fusion/<config>/summary.json`. ~5 s per config |
| `src/models/sweep.py` | W&B Bayesian sweep over late-fusion lr, weight_decay, branch dims, hidden, dropouts, batch size. Objective = mean val AUC over 3 seeds; never evaluates test. `--best ENTITY/PROJECT/ID` prints the `fusion.py` command for the winner (test once, `checkpoints/fusion/late_swept`). `--dry-run` works without login. Needs `wandb login` |
| `src/models/evaluate.py` | Deep dive → `checkpoints/evaluation.json`, ~1 min: rolling-origin folds (test 2023 / 2024 / 2025–26, val = preceding year, train = all earlier) for LR history / embed / history+embed_pca16 / resid / history+resid_pca16 / delta / history+delta_pca16 / guidance / history+guidance / history+guidance+embed_pca16 + swept fusion (blocks via generic `make_blocks`, `_pcaN` suffix); `--text press` → `checkpoints/evaluation_press.json`; pooled AUC + paired-bootstrap ΔAUC vs history; F1 @0.5 vs val-tuned threshold, avg precision, precision in top 10%; company-held-out (GroupKFold by ticker vs random 5-fold, C from the 2025-26 fold; paired bootstrap ΔAUC vs history on the by-company predictions) |
| `src/models/baselines.py` | Logistic regression (balanced, C ∈ 0.001–10 picked by val AUC) on feature blocks alone and combined: `history` (prior-quarter miss/surprise, 4q rolling stats, `shift(1)` so no leakage), `change`, `embed`, `embed_pca{8,16,32}` (PCA fit on train), `tfidf`; plus `avg(history,embed)`. Reports 95% bootstrap AUC range → `checkpoints/baselines.json`. ~1 min. `build_table(data_root, text="mda"|"press")` (incl. `ticker`; `TEXT_SOURCES` = dir, embedding cache, min gap) / `temporal_split` shared with `fusion.py`, `evaluate.py` |
| `docs/` | Learning docs: `00-overview.md` (high-level) + one per component (`07-text-features.md` = embeddings/change features, `08-fusion-model.md` = fusion + manual loop, `09-wandb-sweeps.md`, `10-evaluation-deep-dive.md`, `11-huggingface-hub.md`, `12-press-releases.md`) |
| `notebooks/03_eda.ipynb` | EDA (label distribution, feature tests, correlations, temporal) |

## Key decisions

- **Forward-looking pairing**: a 10-Q is filed right after its own quarter's earnings announcement and already
  states that EPS, so it is paired with the *next* announcement (gap 63–120 days, median 90). Do not revert.
- **Class imbalance**: ~14% misses. Use `ds.get_class_weights()`; judge on F1/AUC, not accuracy.
- **Truncation**: MD&A is 5k–20k tokens; FinBERT sees the first 512 unless `--max-chunks` > 1. MiniLM embeddings always cover the whole filing.
- **Press-release pairing**: releases are filed *on* announcement day and state its EPS; `min_transcript_gap_days=30` stops a release pairing with its own result (without it 3 samples leaked). Do not remove.
- **Model selection**: by val AUC (never test); compare models with `evaluate.py` (rolling-origin + paired bootstrap).

## Current results

Temporal split, test = 2025–2026 (169 samples, 23 misses), 2026-10-22:

| Model | Val AUC | Test AUC [95% bootstrap] | Test F1 |
|---|---|---|---|
| FinBERT frozen, 512 tokens (5 seeds) | 0.59 | 0.515 ± 0.029 | 0.119 |
| FinBERT + LoRA r=8, 512 tokens (5 seeds) | 0.598 ± 0.034 | 0.525 ± 0.029 | 0.170 |
| FinBERT frozen, 16 chunks mean-pooled (3 seeds) | 0.667 ± 0.003 | 0.603 ± 0.011 | 0.065 |
| FinBERT + LoRA r=8, 16 chunks (3 seeds) | 0.639 ± 0.004 | 0.611 ± 0.020 | 0.116 |
| change features only | 0.508 | 0.518 [0.37–0.66] | 0.208 |
| TF-IDF full text | 0.652 | 0.672 [0.55–0.78] | 0.227 |
| MiniLM embedding full text | 0.645 | 0.703 [0.58–0.82] | 0.312 |
| History only (C=0.001) | 0.734 | 0.728 [0.61–0.83] | 0.386 |
| **History + embed PCA-16** (best by val) | **0.752** | **0.752** [0.64–0.85] | 0.351 |
| avg(history, embed) prob ensemble | 0.660 | 0.718 | 0.328 |
| MLP history only (5 seeds) | 0.722 ± 0.011 | 0.669 ± 0.011 | 0.353 |
| MLP embed only (5 seeds) | 0.655 ± 0.017 | 0.695 ± 0.015 | 0.279 |
| Fusion early (5 seeds) | 0.654 ± 0.014 | 0.732 ± 0.020 | 0.391 |
| Fusion late (5 seeds; seed-ensemble test 0.784 [0.68–0.87]) | 0.686 ± 0.027 | 0.734 ± 0.019 | 0.337 |
| Fusion late, W&B-swept (5 seeds; ensemble test 0.745 [0.62–0.84]) | 0.734 ± 0.028 | 0.742 ± 0.032 | 0.361 |

**Rolling-origin pooled (test 2023–26, 403 samples / 58 misses, `evaluate.py`):** history 0.693, embed 0.675,
history+PCA-16 0.708 (ΔAUC vs history +0.016 [−0.021, +0.053], P(Δ>0)=0.80), swept fusion 0.705. Per-year AUC
swings ±0.05 and rankings flip. Top-10% watch-list precision ~38–40% vs 14% base. Val-tuned threshold hurts F1
vs 0.5. **Company-held-out: embed AUC 0.673 → 0.481 (chance)**, history 0.703 → 0.681 — the text embedding
mostly encodes company identity; no transferable text signal. **Within-company text (2026-10-27):** resid 0.494 /
delta 0.506 pooled (chance), adding them to history −0.014 / −0.019, unseen companies 0.51 / 0.44 → MD&A has no
next-quarter signal once identity is removed.

**Press releases (8-K Ex. 99.1, 2026-10-27, `evaluate.py --text press`):** 1,848 samples / 251 misses
(2017–26); pooled test 738 / 100 misses. history 0.653, embed 0.603, history+embed_pca16 0.676 (Δ +0.023
[−0.023, +0.066], P=0.84; ahead in all 3 folds), swept fusion 0.679 (Δ +0.026, P=0.86), resid 0.542, delta
0.502 (within-company: no signal). Unseen companies: embed 0.717 → **0.580** (MD&A 0.481), history+embed
0.661 vs history 0.646. → Some transferable signal, but not significant over history. Fails the bar.
History is lower on these rows than on MD&A rows (0.653 vs 0.693); compare within one table only.

**Guidance features (rule-based, 2026-10-27):** press: history+guidance 0.667 pooled (Δ +0.014, P 0.68), +embed
0.678 (+0.025, P 0.80); **unseen companies history+guidance 0.679 vs 0.646, Δ +0.033 [+0.004, +0.062] P 0.99**
(+embed +0.039 [+0.006, +0.072]) — first interval excluding 0. Guidance alone 0.612 → 0.606 (transfers; embed
0.717 → 0.580). MD&A: history+guidance 0.720 pooled (+0.028, P 0.92, best MD&A model), +embed 0.726; unseen
+0.022 (P 0.87). Miss rate after raise 6%, after net cut 44% (n=16), base 14%. Caveats: CV ignores time; ~10
models compared.

Takeaways: frozen text features + small model beat end-to-end FinBERT (too little data). Full-text
embedding is the best text representation, but its signal is company identity; history drives the
results, and text over history is not significant. Change features: no
signal. Neural late fusion > early fusion on val but < LR history+PCA-16; MLP underperforms LR on history alone. Sweep: text branch → 4 dims, history → 16, high input dropout; tuned net ≈ LR history, still < LR history+PCA-16. Select configs by val AUC, never test. (Random split had shown FinBERT test AUC 0.646 — optimistic.)

## Open issues / next steps

0. Day-2 learning plan in progress (user wants concept → code → explanation, one task at a time, wait for "ready"): Task 1 LoRA done (A: 512 tokens, B: whole-filing chunks); Task 2 fusion model + manual loop done; Task 3 W&B sweep, Task 4 evaluation deep dive done; Task 5 HF Hub done (public); Task 6 docs done. Day-2 plan complete. Day-3 (same style): step 1 within-company MD&A check done (null); step 2 = 8-K press releases done (some transferable signal, not significant over history; docs/12). Step 3 rule-based guidance features done (first significant gain, on unseen companies). Next: user decides (LLM-scored guidance with size/EPS-vs-revenue; MD&A+press combined; refresh HF model with guidance).
1. Evaluate new models with `evaluate.py` (rolling-origin + paired bootstrap), not the single 23-miss window.
2. More/better text: press releases done (1,848 samples). Call transcripts would be next but have no free source.
3. Optional LLM-scored features (hedging, outlook tone, new risks, guidance specificity) — needs Anthropic API key, a few $ per run.
4. Done: text signal is company identity (embed → chance on unseen companies). Text features need to be within-company (vs own past filings) or forward-looking (call Q&A, 8-K guidance) to add anything.
5. Possible upgrades: stronger embedder (e.g. bge-base), Loughran-McDonald dictionary for tone features, gradient boosting.

## Reference

- Repo: https://github.com/guru-prasath-j/earnings-lie-detector (branch `master`). On the Linux box
  (Meril1) this repo pushes with a repo-local credential helper reading `~/personal/.secrets/github.env`
  (account guru-prasath-j); the global `~/.git-credentials` login (guruprasath37) can't push here.
- W&B: Meril1 is logged in to the CoreWeave-hosted server `api.forge.coreweave.com`, entity `guruprasath-nuvo`, project `earnings-lie-detector`. Late-fusion sweep id `0zsaf6qz`.
- HF Hub: https://huggingface.co/Guruprasath12/lie-detect (**public** since 2026-10-27, MIT in card). Meril1 `hf auth` user Guruprasath12. Re-push: `python src/models/push_to_hub.py --data-root . --repo-id Guruprasath12/lie-detect --push` (visibility is unchanged by re-push).
- Full project plan: https://claude.ai/artifact/9sTZ183A6ZcowvVVU6n6fX

## Log

- **2026-10-22** — Fixed scraper `size` parsing; added `yfinance` to requirements. Labeler: supports flat
  `{TICKER}.json`, uses `get_earnings_dates` first (history to ~2014), lookback 40 → 1,899 labels (258 misses).
  Dataset: forward-looking filing→next-announcement pairing, flat-file loading, text-only default (`use_audio`).
  Added `src/models/model.py` + `train.py`; frozen-encoder baseline test F1 0.20 / AUC 0.65 (random split).
- **2026-10-22** — Temporal split (default) in `get_dataloaders`; `train.py` multi-seed with summary.json and
  train-only class weights; added `baselines.py`. Results table above: FinBERT at chance on temporal split,
  history-only best (AUC 0.714). Wrote `docs/` (overview + 6 component docs).
- **2026-10-22** — Pushed to GitHub (guru-prasath-j/earnings-lie-detector, `master`).
- **2026-10-22** — Added `text_features.py` (change features + full-text MiniLM embeddings), extended
  `baselines.py` (PCA blocks, prob-average ensemble, bootstrap CIs, C grid incl. 0.001). Best: history +
  embed PCA-16, val/test AUC 0.752 vs history 0.734/0.728. Change features no signal. Added `docs/07-text-features.md`.
- **2026-10-23** — Task 1A: LoRA (peft, r=8, α=16, q/v) in `model.py`; `train.py` `--lora-*`, `--head-lr`, trainable-only checkpoints. LoRA on 512 tokens: test AUC 0.525 ± 0.029 vs frozen 0.515 — no gain; val AUC usually peaks at epoch 1 then falls (overfits boilerplate). Confirms truncation is the bottleneck. Val F1 is a jumpy early-stopping metric (flips 0↔0.35).
- **2026-10-23** — Task 1B: `max_chunks` whole-filing chunking (mean-pooled [CLS]) in `model.py`; `train.py` `--max-chunks`, `--grad-accum`, sum-reduced loss (mean reduction cancelled class weights at batch 1). 16 chunks lifts FinBERT test AUC 0.515 → 0.603 (frozen) / 0.611 (LoRA) — confirms truncation was the bottleneck, but still below MiniLM full-text LR (0.703); LoRA adds nothing over frozen. Test F1 low (threshold 0.5 too conservative). Runs in `checkpoints/{frozen,lora_r8}_chunks16/`.
- **2026-10-26** — Task 2: `fusion.py` (late/early fusion of history + MiniLM embed, hand-written loop, early stop on val AUC, seed ensemble); `baselines.py` refactored to expose `build_table`/`temporal_split` (results unchanged). Late fusion val 0.686 / test 0.734 (seed-ens 0.784, not mirrored on val → likely noise); early fusion val 0.654 (history drowned); MLP history-only worse than LR. LR history+PCA-16 still best by val. Added `docs/08-fusion-model.md`.
- **2026-10-26** — Task 3: `sweep.py` (W&B Bayesian sweep, 50 runs, objective mean val AUC over 3 seeds, no test); `fusion.py` gained `load_tensors`, `train_one(eval_test, on_epoch)`. Winner (lr 2.3e-3, wd 0.2, history 16, embed 4, hidden 16, dropout 0.53, input-dropout 0.50, batch 64): val 0.734 ± 0.028 / test 0.742 ± 0.032 vs defaults 0.686 / 0.734. Sweep val 0.747 shrank to 0.734 on 5-seed rerun (winner's curse). embed_dim corr −0.75 with val AUC. Still ≤ LR history+PCA-16. Added `docs/09-wandb-sweeps.md`.
- **2026-10-26** — Task 4: `evaluate.py` (rolling-origin 3 folds, pooled + paired bootstrap, thresholds/top-10%, company-held-out); `build_table` adds `ticker`. Pooled AUC ~0.70; text over history +0.016 (CI incl. 0); embed → 0.481 on unseen companies (company identity). Top-10% precision ~40% (2.7× base). Fusion fold-2024 optimistic (sweep tuned on 2024). Added `docs/10-evaluation-deep-dive.md`.
- **2026-10-27** — Task 5: `fusion_net.py` (FusionNet moved here + `HubFusionNet` with mixin and scaler buffers), `push_to_hub.py` (train → wrap → save → model card → round-trip check → optional push). `fusion.py`: `load_tensors(return_scalers)`, `train_one(return_model)`. Round-trip caught a silent bug: config.json sorted `in_dims` keys → branch order swapped → fixed with explicit `blocks`. Packaged checkpoint (seed 42, CPU): val 0.723 / test 0.719, 1,989 params. Fusion/sweep outputs unchanged. Added `docs/11-huggingface-hub.md`.
- **2026-10-27** — Pushed to HF Hub `Guruprasath12/lie-detect` (private); verified `from_pretrained` from the Hub with the downloaded `fusion_net.py`.
- **2026-10-27** — Task 6 docs: `04` gained LoRA, chunking, grad-accum + class-weight bug, FinBERT results; `00` new pipeline diagram, "where we are" (answer, 8-step journey, pooled headline table), next steps, glossary; `05` caveats point to Task 4 findings; new GitHub `README.md`; `requirements.txt` adds huggingface_hub, safetensors.
- **2026-10-27** — Made HF repo `Guruprasath12/lie-detect` public (anonymous download verified); README links it.
- **2026-10-27** — Day-3 step 1: `within_company_embeddings` in `text_features.py`; `evaluate.py` generic `make_blocks` (blocks `history|embed|resid|delta` + `_pcaN`; renamed `history+pca16` → `history+embed_pca16`), company-held-out C now from fold 2025-26 (same values as before). resid/delta at chance (0.494/0.506), hurt when added to history. Old models unchanged. Docs 07, 10 §5, 00 updated.
- **2026-10-27** — Day-3 step 2: `src/data/press_releases.py` (8-K Item 2.02 EX-99.1, 2,115 releases, ~50 min); `dataset.py` `min_transcript_gap_days` (30 for press: without it 3 samples paired a release with its own result); `baselines.TEXT_SOURCES` + `build_table(text=)`; `evaluate.py --text press` → `evaluation_press.json`. 1,848 samples. history+embed +0.023 (P 0.84, n.s.), embed 0.58 on unseen companies (MD&A 0.48); resid/delta no signal. MD&A results unchanged. Added `docs/12-press-releases.md`; updated 00, 10, README.
- **2026-10-27** — Day-3 step 3: `text_features.guidance_features` (rule-based outlook direction + hedging; rules tightened for precision by reading samples, not labels); `build_table` adds guidance cols; `evaluate.py` models guidance / history+guidance / history+guidance+embed_pca16, company-held-out paired bootstrap. Press: unseen-company Δ +0.033 [+0.004, +0.062] (first significant gain); rolling-origin +0.014 n.s. MD&A: history+guidance 0.720 (+0.028, P 0.92). Old models unchanged. Docs 12 (step 3 section), 07, 00, README.
