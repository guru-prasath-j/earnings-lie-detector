# 11 — Publishing the model on the Hugging Face Hub

**Files:** `src/models/fusion_net.py` (`FusionNet`, `HubFusionNet`), `src/models/push_to_hub.py`

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
python src/models/push_to_hub.py --data-root .        # train, wrap, save, verify → checkpoints/hub/earnings-miss-fusion/
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
