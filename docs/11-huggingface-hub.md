# 11 — Publishing the model on the Hugging Face Hub

**Files:** `src/models/fusion_net.py` (`FusionNet`, `HubFusionNet`), `src/models/push_to_hub.py`;
since 2026-10-29 also `src/models/miss_model.py` (`HubMissModel`) and `src/models/guidance_rules.py` (see the last section)

## What the Hub is

The Hugging Face Hub is git for models. Each model repo holds:

| File | What |
|---|---|
| `model.safetensors` | The weights. *safetensors* is a plain tensor format, safe to load, unlike pickle-based `.pt`, which can run arbitrary code |
| `config.json` | The arguments needed to rebuild the model object |
| `README.md` | The **model card**: what it predicts, inputs, training data, evaluation, limitations |
| `fusion_net.py` | Our class code. Custom PyTorch models need it to be loaded |

## `PyTorchModelHubMixin`

```python
class HubFusionNet(nn.Module, PyTorchModelHubMixin, library_name="pytorch", tags=[...]):
    def __init__(self, blocks, in_dims, branch_dims, hidden, dropout, input_dropout, ...): ...
```

Adding the mixin gives the class three methods:

- `save_pretrained(dir)` records the `__init__` arguments in `config.json` and the weights in `model.safetensors`
- `push_to_hub(repo_id)` does the same, straight to the Hub
- `HubFusionNet.from_pretrained(repo_id_or_dir)` reads `config.json`, calls `__init__(**config)`, then loads the weights

So every `__init__` argument must be JSON-serialisable (ints, floats, strings, lists, dicts).

## Ship the preprocessing with the weights

The network was trained on *standardised* inputs (minus the train mean, divided by the train std).
Upload only the weights and every user must reproduce those exact means and stds. Instead,
`HubFusionNet` stores them as **buffers**:

```python
self.register_buffer("mean_history", torch.zeros(5))   # saved with the weights, moved by .to(),
self.register_buffer("std_history",  torch.ones(5))     # but not a Parameter, so never trained
```

`forward` standardises first, so users pass **raw** features: `model.predict_proba(history, embedding)`.

## The bug the round-trip test caught

`push_to_hub.py` reloads the saved model and asserts its predictions are identical. On the first run
they weren't. The cause: the mixin writes `config.json` with **sorted keys**, so
`in_dims={"history": 5, "embed": 384}` came back as `{"embed": 384, "history": 5}`. `FusionNet` builds
and concatenates its branches in dict order, so after reloading the head received [text, history] while
its weights expected [history, text]. No error was raised; it just gave wrong probabilities.

Fix: an explicit `blocks=["history", "embed"]` argument. Lists keep their order in JSON.
**Lesson: always test save → load → same predictions before publishing.**

## Running it

```bash
python src/models/push_to_hub.py --data-root . --model fusion   # train, wrap, save, verify → checkpoints/hub/earnings-miss-fusion/
hf auth login                                         # once per machine (token with write access)
python src/models/push_to_hub.py --data-root . --repo-id Guruprasath12/lie-detect --push   # private
python src/models/push_to_hub.py --data-root . --repo-id Guruprasath12/lie-detect --push --public  # public
```

The script trains the model exactly as evaluated: the sweep-winner config, train < 2024, early stopping
on 2024, seed 42, on CPU. That gives val 0.723 / test 0.719 — not Task 3's seed-42 0.728 / 0.747,
because CPU and GPU random generators shuffle differently. Single-seed numbers move that much.

The model card reports this checkpoint's split metrics, the rolling-origin pooled numbers from
`checkpoints/evaluation.json`, and the limitations, above all that the text part mostly encodes
company identity. A model card's job is to stop people from over-trusting the model.

(Since step 13 the MD&A table has 186 companies, so `--model fusion` now trains on more, harder data:
val 0.673 / test 0.637. The numbers above are from the 49-company run.)

## Version 2: the model the evidence supports (2026-10-29)

After scaling to 188 companies ([13](13-more-companies.md)), the MD&A embedding added nothing, so the
Hub repo now holds a different model: **logistic regression on history + sector + guidance rules**.

**Choosing it.** By validation AUC (never test), averaged over the 2023 and 2024 validation years:
history 0.656, history + sector 0.662, **history + sector + guidance 0.678**, + embedding 0.681. The
embedding version is 0.003 higher, far inside the noise, and would need a 90 MB sentence-embedding model
to run. The simpler model reads plain text with regex rules, so it was chosen, and the card says so.

**A logistic regression as a Hub model.** LR is `Linear(25 → 1)` + sigmoid. `HubMissModel` copies
sklearn's `coef_` / `intercept_` into an `nn.Linear` and the scaler's mean and std into buffers, then
uses the same mixin as before. The checks in `push_to_hub.py`:

1. The wrapper gives sklearn's probabilities on all 6,916 rows.
2. **Raw-input path:** for 300 sampled rows, the features rebuilt from *past surprises + sector name +
   release text* (what a user has) equal the training table's features. This catches any drift between
   training code and shipped code.
3. Round trip through `save_pretrained` / `from_pretrained`.

**One copy of the rules.** The guidance regexes moved from `text_features.py` into `guidance_rules.py`
(standard library only), which training imports and the Hub repo ships. Before moving them, the features
for all 8,268 releases were saved, and after the move they were checked to be identical.

**Evaluate, then refit.** C = 10 is chosen on 2024 validation (train < 2024), which gives test 2025–26
AUC 0.671. The released weights are then **refit on all years** with that C: more data for the model
people use, while the reported numbers still come from a model that never saw the test years.

```bash
python src/models/push_to_hub.py --data-root .                                          # miss model (default)
python src/models/push_to_hub.py --data-root . --repo-id Guruprasath12/lie-detect --push
```

The upload deletes `fusion_net.py` from the repo (`delete_patterns`), so the repo never mixes two
models' code. The fusion version is still in the Hub repo's commit history.

```python
from miss_model import HubMissModel        # with miss_model.py and guidance_rules.py downloaded
model = HubMissModel.from_pretrained("Guruprasath12/lie-detect")
model.predict(past_surprises=[3.0, 2.5, 4.1, 1.0], sector="Industrials", release_text=text)
```

## Version 3: adding analyst and price data (2026-10-29)

After step 15 ([14](14-market-features.md)) the Hub model gained the analyst and price blocks:
**history + sector + analyst + price + guidance**, 32 features.

**Choosing it:** mean validation AUC (2023, 2024) 0.692, against 0.682 without guidance and 0.677 for
version 2. A clear win on validation, so it replaced version 2, even though the forward-in-time test
shows no gain in 2025–26. The card leads with that.

**What changed in the package:**
- `market_features.py` is shipped too (numpy + pandas). The per-announcement calculation became
  `market_vector(analyst, prices, spy, date)`, used by training and inference alike. Before the refactor
  the features for all 7,089 announcements were saved and afterwards checked identical.
- `HubMissModel` gained two config fields: `blocks` (which feature groups it uses) and `fills`
  (training values for missing inputs, such as a new listing's volatility). Version 2 configs still load.
- `predict(..., ticker=, announcement_date=)` fetches analyst actions and prices from Yahoo with
  `yahoo_inputs`. `market_inputs=` takes your own data instead.

**Checks added to `push_to_hub.py`:** the raw-input path now includes the market blocks (300 rows), and a
**live Yahoo fetch** is compared with the cached training files on 5 recent announcements (max
difference ~1e-7). That catches the risk that what users fetch differs from what the model was trained on.

```python
from miss_model import HubMissModel   # with miss_model.py, guidance_rules.py, market_features.py downloaded
model = HubMissModel.from_pretrained("Guruprasath12/lie-detect")
model.predict(past_surprises=[4.2, -1.5, 2.8, 6.0], sector="Industrials", release_text=text,
              ticker="CAT", announcement_date="2026-11-04")
```

