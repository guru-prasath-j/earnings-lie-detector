# Earnings Lie Detector — project context

Predict whether a company will **miss its next earnings EPS estimate** (label 1) or meet/beat it (label 0)
from the MD&A section of its SEC 10-Q filings and its 8-K earnings press releases. 188 companies since
2026-10-29 (49 original + 139 S&P 500 by sector, REITs excluded). Text-only; audio is optional and off.

## Keep this file current (standing instruction)

This file is the hand-off between machines and sessions. After any meaningful change — new/changed
scripts, a pipeline run, training results, a design decision, a new known issue — update the relevant
section below and the **Log** at the bottom (date, what changed, results). Keep it short and factual;
replace stale info rather than appending contradictions. Commit it together with the code it describes.

Learning docs live in `docs/` (`00-overview.md` = high-level + project journey, `01`–`14` = one per component). `README.md` = GitHub-facing summary. The user
reads them to learn — when an implementation or result changes, update the matching doc too.

## Setup on a new machine

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`data/raw/`, `data/processed/`, `*.pt` and `checkpoints/` are gitignored — regenerate them:

```bash
T=$(python src/data/universe.py --list all)                        # data/universe.csv is committed (200 rows, 10 REITs excluded → 190)
python src/data/scraper.py --tickers $T                            # → data/raw/transcripts/{TICKER}.json (3,193 filings), ~1.5 h
python src/data/labeler.py --tickers $T                            # → data/processed/labels.csv (~7,100 rows; here: 1,899 original + 5,190 expansion, appended)
python src/data/dataset.py                                         # sanity check (old 49-company numbers: 661 samples, 92 misses)
python src/models/train.py --data-root . --freeze-encoder --epochs 10 --batch-size 8   # 5 seeds, ~7 min on RTX A1000
python src/models/baselines.py --data-root .                                          # LR yardsticks; first run builds MiniLM cache (~1.5 min)
python src/models/fusion.py --data-root .                                             # neural late fusion, 5 seeds, ~5 s
python src/models/evaluate.py --data-root .                                           # rolling-origin deep dive → checkpoints/evaluation.json, ~1 min
python src/data/press_releases.py --tickers $T                                        # → data/raw/press_releases/{TICKER}.json (8,313 releases), ~3 h
python src/models/llm_guidance.py                                                    # optional, original 50 only so far (build_table drops LLM cols if >1% unscored), needs ~/personal/.secrets/nvidia.env → data/processed/llm_guidance.jsonl (~4 h on free tier; resumes)
python src/data/market_data.py                                                        # → data/raw/market/{T}_analyst.csv, {T}_prices.csv (+ SPY), ~3 min
python src/models/evaluate.py --data-root . --text press                              # → checkpoints/evaluation_press.json (first run embeds, ~1.5 min)
```

## Pipeline

| File | Role |
|---|---|
| `src/data/universe.py` | Original 50 tickers (`scraper.get_default_tickers`) + 150 from current S&P 500 (GitHub datasets/s-and-p-500-companies), one per CIK, `.`→`-`, largest-remainder allocation by GICS sector, seed 0 → committed `data/universe.csv` (`ticker,name,sector,cik,batch,exclude`). `EXCLUDED_SECTORS` = Real Estate (REIT FFO vs GAAP EPS → 45–62% "misses"). `load_universe(batch)` (skips excluded), `load_sectors()`. `--list all|original|expansion` prints tickers |
| `src/data/market_data.py` | Yahoo point-in-time data per universe ticker + SPY: `upgrades_downgrades` (rating + price-target actions, timestamped since ~2012) and daily Close from 2015 → `data/raw/market/`. Estimate tables (eps_trend/revisions/dispersion/analyst count) are today's snapshot only → not used (leak) |
| `src/data/scraper.py` | SEC EDGAR 10-Q MD&A text → flat `data/raw/transcripts/{TICKER}.json` (list of records: `company`, `date`=filing date, `quarter`, `form`, `mda_text`, …) plus combined `_all_transcripts.json` |
| `src/data/press_releases.py` | 8-K Item 2.02 earnings press releases: submissions API (all pages — banks' "recent" covers ~1 yr) → `-index.htm` Type `EX-99.1` (fallback filename `ex991`, any `EX-99.*`) → text without tables (kept if narrative < 1,500 chars), SGML header cut. Reuses `scraper.py` helpers. Records `date`/`text` like MD&A → `data/raw/press_releases/{T}.json`. Default 2016-07-01 → today. XOM 1 release, BLK 9 |
| `src/data/labeler.py` | Default `--method direct`: yfinance `get_earnings_dates` (history back to ~2014, `lookback_quarters=40`) → one row per earnings announcement; `label = 1 if eps_actual < eps_estimate`. `date` = announcement date. Reads both `{T}.json` and `{T}/*.json` layouts, skips `_*` files |
| `src/data/dataset.py` | `EarningsCallDataset`. Pairs each label (announcement date) with the latest filing **strictly before** it within 120 days (`max_transcript_gap_days`) and at least `min_transcript_gap_days` (default 0; 30 for press releases) before it. `transcripts_dir` picks the text source. `use_audio=False` by default → text-only, drops labels with no filing, batches have no audio keys. `get_dataloaders(split="temporal")` default: train < 2024-01-01 (370, 48 miss), val 2024 (122, 21), test ≥ 2025-01-01 (169, 23); `split="random"` still available |
| `src/data/audio_downloader.py`, `audio_features.py` | Optional audio pipeline (Earnings-21/22 HF datasets → ~20–28 librosa features). Not run here; won't join to labels (benchmark calls ≠ our tickers/dates). Downloads ~20–35 GB if run without `--max-calls` |
| `src/models/model.py` | `EarningsTextClassifier`: `ProsusAI/finbert` (fallback `distilbert-base-uncased`) → [CLS] → Dropout(0.3) → Linear(768→128) → ReLU → Dropout(0.15) → Linear(128→2). `freeze_encoder` / `freeze_n_layers` / LoRA (`lora_r>0`: peft adapters on query+value, r=8 → 393,602 trainable = 0.36%). `max_chunks>1`: whole filing → 512-token chunks, up to N evenly spaced, [CLS] mean-pooled per doc (`doc_ids`) |
| `src/models/train.py` | Class-weighted CE (weights from train split), AdamW (head LR = 10× encoder LR), warmup→cosine, AMP on CUDA, early stop on val F1 (patience 3). `--seeds` (default 42–46) → `checkpoints/seed_N/{best_model.pt,training_history.json}` + `checkpoints/summary.json` (mean ± std). `--split temporal|random`, `--val-start`, `--test-start`, `--lora-r/--lora-alpha/--lora-dropout`, `--head-lr`, `--max-chunks`, `--grad-accum` (loss uses `reduction="sum"` / fixed effective batch so class weights survive batch-size 1). Checkpoints hold trainable weights only (~1.6 MB) |
| `src/models/text_features.py` | `change_features` (cosine/Jaccard/length vs previous and ~1-year-earlier filing, neg/uncertainty word rates) and `within_company_embeddings` (`resid` = emb − mean of company's earlier rows, `delta` = emb − previous row; first row 0) and `guidance_features` (outlook sentences → closest direction verb within 40 chars: raise/cut/maintain/narrow/initiate/withdraw counts, `g_net`, `has_outlook`, `n_outlook`, `outlook_hedge_rate`; `GUIDANCE_COLS`; ~85% precision on spot checks, tuned without labels) and `document_embeddings` (whole MD&A → 256-token chunks → frozen `all-MiniLM-L6-v2` mean-pooled → avg, 384-dim; cached in `data/processed/embeddings_minilm.npz`, ~1.5 min on GPU) |
| `src/models/llm_guidance.py` | LLM-scored guidance: outlook passage (outlook sentences ±2, ≤6k chars) → NVIDIA OpenAI-compatible API, `openai/gpt-oss-20b` (llama-3.3-70b retired, 410), temp 0, fixed JSON schema (EPS/revenue direction incl. `given`, change %, horizon, tone, headwinds, specificity). JSONL cache with resume, `--retry-failed`, `--timeout 300` (free tier queues 90–250 s), 16 workers, no retry on 4xx≠429. Key from `~/personal/.secrets/nvidia.env`, never printed. `llm_features()` → 9 `LLM_COLS`; `build_table(text="press")` joins them when the cache exists; `evaluate.py` then adds llm / history+llm / history+guidance+llm / history+llm+embed_pca16 |
| `src/models/guidance_rules.py` | Self-contained (`re` only) guidance rules: `GUIDANCE_COLS`, regexes, `guidance_vector(text)`. `text_features.guidance_features` imports it (refactor verified identical on 8,268 releases); shipped to the Hub |
| `src/models/miss_model.py` | Self-contained `HubMissModel` (mixin; `Linear(d→1)` + mean/std buffers; config `features`, `sectors`, `C`, `blocks`, `fills`; v2 configs without blocks still load), `history_vector(past_surprises)` (matches `baselines.history_features`), `feature_vector`, `predict(past_surprises, sector, release_text, ticker=, announcement_date= | market_inputs=)`, `predict_proba(X)`. Shipped with `guidance_rules.py`, `market_features.py` |
| `src/models/market_features.py` | `market_features(keys, market_dir)`: windows end `SAFETY_DAYS`=2 before the announcement (Yahoo dates can be late). `ANALYST_COLS` (90 days: `an_n` log count, `an_net_grade` up−down, `an_pt_net` PT raised−lowered share, `an_pt_chg` mean PT % change ±50%), `PRICE_COLS` (`px_rel_63`/`px_rel_21` return − SPY clipped ±1/±0.5, `px_vol_63`). `build_table` joins them when `data/raw/market` exists; `evaluate.py` `MARKET_MODELS`. Per-announcement `market_vector(an, px, spy, date)` + `prepare_analyst`, `yahoo_inputs(ticker)` (live fetch); shipped to the Hub (refactor verified identical on 7,089 announcements) |
| `src/models/fusion_net.py` | Self-contained (torch + huggingface_hub) `FusionNet` and `HubFusionNet` (`PyTorchModelHubMixin`; explicit `blocks` list because config.json sorts keys; train scaler mean/std as buffers → takes raw features; `predict_proba(history, embedding)`) |
| `src/models/push_to_hub.py` | `--model miss` (default): LR `MISS_MODEL = history+sector+analyst+price+guidance` on press releases (C=10 on 2024 val → test 2025–26 AUC 0.671, then refit on all 6,916 rows), wraps in `HubMissModel`, checks sklearn = wrapper, raw path (past surprises + sector + text + market inputs) = table on 300 rows, live Yahoo vs cached on 5 rows, round trip; card from `evaluation_press.json` → `checkpoints/hub/earnings-miss-model/`; push deletes `fusion_net.py` from the repo. `--model fusion`: trains swept fusion (seed 42, CPU, `evaluate.FUSION_CFG`), wraps in `HubFusionNet`, `save_pretrained` → `checkpoints/hub/earnings-miss-fusion/` + model card + `fusion_net.py`, asserts identical preds after `from_pretrained`. Uploads only with `--push --repo-id` (private unless `--public`); needs `hf auth login` |
| `src/models/fusion.py` | `FusionNet` (imported from `fusion_net.py`): history (5) and MiniLM embed (384) blocks; `--fusion late` (per-block Linear+ReLU branches 5→8, 384→16, concat → MLP → 1 logit) or `early` (raw concat → MLP); `--blocks` for single-branch. Hand-written loop (no DataLoader): BCEWithLogits + `pos_weight`, AdamW 1e-3, early stop on val AUC (patience 20), best `state_dict` deepcopy, 5 seeds + seed-ensemble → `checkpoints/fusion/<config>/summary.json`. ~5 s per config |
| `src/models/sweep.py` | W&B Bayesian sweep over late-fusion lr, weight_decay, branch dims, hidden, dropouts, batch size. Objective = mean val AUC over 3 seeds; never evaluates test. `--best ENTITY/PROJECT/ID` prints the `fusion.py` command for the winner (test once, `checkpoints/fusion/late_swept`). `--dry-run` works without login. Needs `wandb login` |
| `src/models/evaluate.py` | Deep dive → `checkpoints/evaluation.json`, ~1 min: rolling-origin folds (test 2023 / 2024 / 2025–26, val = preceding year, train = all earlier) for LR history / embed / history+embed_pca16 / resid / history+resid_pca16 / delta / history+delta_pca16 / guidance / history+guidance / history+guidance+embed_pca16 + swept fusion (blocks via generic `make_blocks`, `_pcaN` suffix); `--text press` → `checkpoints/evaluation_press.json`; `--companies all|original|expansion` (suffix `_original`, filter after `build_table`); sector one-hot block + `SECTOR_MODELS` with a second paired bootstrap vs `history+sector`; pooled AUC + paired-bootstrap ΔAUC vs history; F1 @0.5 vs val-tuned threshold, avg precision, precision in top 10%; company-held-out (GroupKFold by ticker vs random 5-fold, C from the 2025-26 fold; paired bootstrap ΔAUC vs history on the by-company predictions) |
| `src/models/baselines.py` | Logistic regression (balanced, C ∈ 0.001–10 picked by val AUC) on feature blocks alone and combined: `history` (prior-quarter miss/surprise, 4q rolling stats, `shift(1)` so no leakage), `change`, `embed`, `embed_pca{8,16,32}` (PCA fit on train), `tfidf`; plus `avg(history,embed)`. Reports 95% bootstrap AUC range → `checkpoints/baselines.json`. ~1 min. `build_table(data_root, text="mda"|"press")` (incl. `ticker`; `TEXT_SOURCES` = dir, embedding cache, min gap) / `temporal_split` shared with `fusion.py`, `evaluate.py` |
| `docs/` | Learning docs: `00-overview.md` (high-level) + one per component (`07-text-features.md` = embeddings/change features, `08-fusion-model.md` = fusion + manual loop, `09-wandb-sweeps.md`, `10-evaluation-deep-dive.md`, `11-huggingface-hub.md`, `12-press-releases.md`, `13-more-companies.md`, `14-market-features.md`) |
| `notebooks/03_eda.ipynb` | EDA (label distribution, feature tests, correlations, temporal) |

## Key decisions

- **Forward-looking pairing**: a 10-Q is filed right after its own quarter's earnings announcement and already
  states that EPS, so it is paired with the *next* announcement (gap 63–120 days, median 90). Do not revert.
- **Class imbalance**: ~14% misses. Use `ds.get_class_weights()`; judge on F1/AUC, not accuracy.
- **Truncation**: MD&A is 5k–20k tokens; FinBERT sees the first 512 unless `--max-chunks` > 1. MiniLM embeddings always cover the whole filing.
- **Press-release pairing**: releases are filed *on* announcement day and state its EPS; `min_transcript_gap_days=30` stops a release pairing with its own result (without it 3 samples leaked). Do not remove.
- **Model selection**: by val AUC (never test); compare models with `evaluate.py` (rolling-origin + paired bootstrap).
- **Text must beat history + sector**, not just history: with 10 sectors, sector carries base-rate signal that text picks up (it was most of the 49-company "guidance gain").
- **REITs excluded** (Real Estate sector): Yahoo EPS labels are on mixed FFO/GAAP bases. Labels for new companies were fetched separately and appended, so original rows are unchanged.

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

**LLM guidance (gpt-oss-20b, 2026-10-28):** 1,367 releases scored, 6 failed. llm alone 0.585 pooled / 0.597 unseen;
history+llm 0.665 (+0.012, P 0.67) / unseen 0.671 (+0.025 [−0.006, +0.055], P 0.94); history+guidance+llm
0.669 / 0.665 (no gain from combining); history+llm+embed 0.679 / unseen +0.037 [+0.001, +0.071]. = rules, not
better. Miss rate after EPS raise ≥3% 0/31; LLM "lowered" 11% (n 27) vs rules' cut 44%; confident tone 7–8% vs
neutral 16.5%. Bottleneck = number of misses, not extraction.

**188 companies (2026-10-29, `docs/13`):** press 6,916 samples / 1,161 misses, pooled test 2,816 / 469. history 0.655,
history+sector 0.670 (+0.015, P 0.96), history+sector+guidance **0.676** (+0.021 [+0.002, +0.041]), fusion 0.676,
history+guidance 0.662 (+0.007). Unseen companies: history+guidance +0.012 [+0.003, +0.022] (was +0.033),
history+sector +0.015. **Over history+sector: guidance +0.006 [−0.004, +0.017] pooled, +0.004 unseen; embed
+0.001 / +0.008** → text adds nothing meaningful. On the original 50, history+sector alone = +0.033 unseen (= old
guidance gain); guidance on top −0.010. MD&A (2,983 / 545; test 1,731 / 325): history 0.644, history+sector 0.664,
history+guidance −0.002 (49-company +0.028 was noise), text over history+sector ≤ 0 pooled. Intervals ±0.05 →
±0.02. Miss rate new companies: after raise 10.7% (699), net cut 27.8% (54), base 17%. Sector miss rates 8% (IT)
– 29% (Materials). `--companies original` reproduces all earlier numbers. LLM models not run (new companies unscored).
The 49-company tables above are kept for history.

**Analyst actions + price run-up (2026-10-29, `docs/14`):** press rows, over history+sector: analyst +0.005 pooled /
+0.011 [+0.004, +0.017] unseen; price +0.001 / +0.013; analyst+price +0.005 [−0.009, +0.020] / **+0.019 [+0.011,
+0.028]**; +guidance +0.010 / +0.020. Unseen gain unchanged when the test row's own year is excluded from training
(not a same-quarter shock). Per fold analyst+price vs h+s: 2023 +0.019, 2024 +0.012, 2025–26 −0.003. Best pooled
h+s+analyst+price+guidance 0.680. MD&A rows: +0.005 pooled / +0.021 unseen (same announcements, not independent).
Miss rate after net PT cuts 22.3% vs raises 13.7%; bottom-quintile 3-month relative return 22.2%.

Takeaways: frozen text features + small model beat end-to-end FinBERT (too little data). Full-text
embedding is the best text representation, but its signal is company identity; history drives the
results, and text over history is not significant. Change features: no
signal. Neural late fusion > early fusion on val but < LR history+PCA-16; MLP underperforms LR on history alone. Sweep: text branch → 4 dims, history → 16, high input dropout; tuned net ≈ LR history, still < LR history+PCA-16. Select configs by val AUC, never test. (Random split had shown FinBERT test AUC 0.646 — optimistic.)

## Open issues / next steps

0. Day-2 learning plan in progress (user wants concept → code → explanation, one task at a time, wait for "ready"): Task 1 LoRA done (A: 512 tokens, B: whole-filing chunks); Task 2 fusion model + manual loop done; Task 3 W&B sweep, Task 4 evaluation deep dive done; Task 5 HF Hub done (public); Task 6 docs done. Day-2 plan complete. Day-3 (same style): step 1 within-company MD&A check done (null); step 2 = 8-K press releases done (some transferable signal, not significant over history; docs/12). Step 3 rule-based guidance features done (first significant gain, on unseen companies). Step 4 LLM-scored guidance done (= rules, no extra gain). Step 5 more companies done (49 → 188; guidance gain mostly sector, text over history+sector n.s.). Step 6 HF model refreshed (LR history+sector+guidance, public, verified anonymously). Step 7 one-page summary artifact done. Step 8 analyst actions + price run-up done (strongest add-on on unseen companies, not proven over time). Step 9 HF model v3 with market features done (won on val 0.692 vs 0.677; card shows per-year table). Next: user decides (paid estimate-revision data; refresh the one-page summary; stop).
1. Evaluate new models with `evaluate.py` (rolling-origin + paired bootstrap), not the single 23-miss window.
2. More/better text: press releases done (1,848 samples). Call transcripts would be next but have no free source.
3. Done: LLM-scored guidance via free NVIDIA API (`llm_guidance.py`) — matches the rules, no extra gain.
4. Done: text signal is company identity (embed → chance on unseen companies). Text features need to be within-company (vs own past filings) or forward-looking (call Q&A, 8-K guidance) to add anything.
5. MD&A for the 7 big banks has 1 filing each (MD&A lives in an exhibit, scraper reads the 10-Q body only). Press releases are complete.
6. Possible upgrades: stronger embedder (e.g. bge-base), Loughran-McDonald dictionary for tone features, gradient boosting.

## Reference

- Repo: https://github.com/guru-prasath-j/earnings-lie-detector (branch `master`). On the Linux box
  (Meril1) this repo pushes with a repo-local credential helper reading `~/personal/.secrets/github.env`
  (account guru-prasath-j); the global `~/.git-credentials` login (guruprasath37) can't push here.
- W&B: Meril1 is logged in to the CoreWeave-hosted server `api.forge.coreweave.com`, entity `guruprasath-nuvo`, project `earnings-lie-detector`. Late-fusion sweep id `0zsaf6qz`.
- HF Hub: https://huggingface.co/Guruprasath12/lie-detect (**public** since 2026-10-27, MIT in card). Since 2026-10-29 holds `HubMissModel` v3 (LR history+sector+analyst+price+guidance, 32 features, needs yfinance at inference); v2 (h+s+g) and the fusion net are in the repo history. Meril1 `hf auth` user Guruprasath12. Re-push: `python src/models/push_to_hub.py --data-root . --repo-id Guruprasath12/lie-detect --push` (visibility is unchanged by re-push).
- Full project plan: https://claude.ai/artifact/9sTZ183A6ZcowvVVU6n6fX
- One-page project summary (2026-10-29, private until shared): https://claude.ai/artifact/5MRWJRsSH5QH1z3LtVMfD4

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
- **2026-10-28** — Day-3 step 4: `src/models/llm_guidance.py` (NVIDIA API, gpt-oss-20b, JSON schema, cached JSONL, ~4 h; timeout raised 120→300 s after 25% of calls timed out); `baselines.TEXT_SOURCES["press"]["llm"]` joins 9 LLM cols; `evaluate.py` 4 LLM models when present; `openai` in requirements. LLM ≈ rules: history+llm +0.012 pooled, +0.025 unseen (P 0.94) vs rules +0.014 / +0.033; combining adds nothing. Old models unchanged. Docs 12 (step 4), 00, README.
- **2026-10-29** — Day-3 step 5: `src/data/universe.py` + committed `data/universe.csv` (50 + 150 S&P 500 by sector; Real Estate excluded for REIT label noise → 188 labeled companies). Scraped MD&A (2,529 new) and press releases (6,198 new); labels fetched to `labels_expansion.csv` and appended (originals byte-identical, backup `labels_original.csv`). `evaluate.py` `--companies`, sector block + `SECTOR_MODELS` + bootstrap vs history+sector; `build_table` drops LLM cols if >1% unscored. Press: history+sector+guidance 0.676; guidance over history+sector +0.006 (n.s.); unseen guidance gain +0.033 → +0.012, mostly sector. MD&A guidance gain was noise. Added `docs/13-more-companies.md`; updated 00, 12, README.
- **2026-10-29** — Day-3 step 6: guidance rules moved to self-contained `guidance_rules.py` (features identical); new `miss_model.py` (`HubMissModel`, `history_vector`); `push_to_hub.py --model miss|fusion` (default miss). Chosen by mean val AUC 2023/24 (h+s+g 0.678 vs +embed 0.681, simpler wins). C=10, test 2025–26 0.671, released refit on all 6,916 rows / 188 companies. Pushed to `Guruprasath12/lie-detect` (fusion_net.py deleted), anonymous load verified. `--model fusion` now trains on 186-company MD&A: val 0.673 / test 0.637. Docs 11 (version 2), 00 (row 14), README.
- **2026-10-29** — Day-3 step 8: `src/data/market_data.py` (Yahoo analyst actions + prices, point-in-time; estimate snapshots rejected as leaky), `src/models/market_features.py` (90-day analyst block, price run-up vs SPY, 2-day safety margin); `build_table` joins them; `evaluate.py` `MARKET_MODELS`. Over history+sector: unseen +0.019 [+0.011, +0.028] (same with own year excluded), over time +0.005 n.s. (2023 +0.019, 2024 +0.012, 2025–26 −0.003). Best pooled 0.680. Old models unchanged. Added `docs/14-market-features.md`; updated 00, README. Also: one-page summary artifact linked under Reference.
- **2026-10-29** — Day-3 step 9: Hub model v3 = LR history+sector+analyst+price+guidance (mean val 2023/24 0.692 vs 0.677 v2). `market_features.py` refactored to `market_vector`/`prepare_analyst`/`yahoo_inputs` (features identical), shipped; `HubMissModel` config `blocks`/`fills`, `predict(..., ticker, announcement_date)` fetches Yahoo. `push_to_hub.py` checks raw path incl. market (300 rows) and live Yahoo = cache (5 rows, ~1e-7). Pooled 0.680, per year vs h+s +0.034/+0.021/−0.002, unseen +0.020. Pushed, anonymous load + live CAT prediction verified. Docs 11 (v3), 14, 00 (row 16), README.
