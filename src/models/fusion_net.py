"""
Fusion network definitions — self-contained (torch + huggingface_hub only), so this
file can be shipped next to the weights on the Hugging Face Hub.

- ``FusionNet``: the trainable network used by ``fusion.py`` / ``sweep.py``.  Expects
  inputs that are *already* standardised.
- ``HubFusionNet``: ``FusionNet`` + the standardisation statistics, with Hub
  save/load.  Takes *raw* features and returns P(miss).

Load from the Hub:

    from fusion_net import HubFusionNet
    model = HubFusionNet.from_pretrained("<user>/earnings-miss-fusion")
    p_miss = model.predict_proba(history, embedding)   # (n, 5), (n, 384) → (n,)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class FusionNet(nn.Module):
    """One branch per input block, concatenated, then a small MLP head → 1 logit.

    ``branch_dims`` maps block name → branch output size.  ``early=True`` skips
    the branches and feeds the raw concatenated inputs straight to the head.
    """

    def __init__(self, in_dims: Dict[str, int], branch_dims: Dict[str, int], hidden: int = 32,
                 dropout: float = 0.3, input_dropout: float = 0.2, early: bool = False) -> None:
        super().__init__()
        self.blocks = list(in_dims)
        self.early = early
        if early:
            self.branches = nn.ModuleDict()
            head_in = sum(in_dims.values())
        else:
            # Input dropout on the wide text block: randomly hides some of the 384 dims
            # each step so the branch can't memorise a few of them.
            self.branches = nn.ModuleDict({
                b: nn.Sequential(nn.Dropout(input_dropout if in_dims[b] > 32 else 0.0),
                                 nn.Linear(in_dims[b], branch_dims[b]), nn.ReLU())
                for b in self.blocks
            })
            head_in = sum(branch_dims[b] for b in self.blocks)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.early:
            z = torch.cat([x[b] for b in self.blocks], dim=1)
        else:
            z = torch.cat([self.branches[b](x[b]) for b in self.blocks], dim=1)
        return self.head(z).squeeze(1)  # (batch,) logits; sigmoid(logit) = P(miss)


class HubFusionNet(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="pytorch",
    tags=["tabular-classification", "finance", "earnings", "sec-filings"],
    repo_url="https://github.com/guru-prasath-j/earnings-lie-detector",
):
    """``FusionNet`` that standardises its own inputs.

    Every ``__init__`` argument must be JSON-serialisable: the mixin writes them to
    ``config.json`` and calls ``__init__(**config)`` again in ``from_pretrained``.
    ``blocks`` fixes the branch order explicitly: config.json is written with sorted
    keys, so the order of ``in_dims`` does not survive a save/load round trip — and the
    head's weights depend on which branch comes first in the concatenation.
    The scaler statistics are **buffers** — saved in ``model.safetensors`` with the
    weights, moved by ``.to(device)``, but never updated by the optimiser.
    """

    def __init__(self, blocks: List[str], in_dims: Dict[str, int], branch_dims: Dict[str, int],
                 hidden: int = 32, dropout: float = 0.3, input_dropout: float = 0.2,
                 history_features: Optional[List[str]] = None,
                 embedder: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        super().__init__()
        self.net = FusionNet({b: in_dims[b] for b in blocks}, branch_dims, hidden, dropout, input_dropout)
        self.history_features = history_features or []
        self.embedder = embedder
        for b, d in in_dims.items():
            # Placeholders; real values come from set_scaler() or the saved weights
            self.register_buffer(f"mean_{b}", torch.zeros(d))
            self.register_buffer(f"std_{b}", torch.ones(d))

    def set_scaler(self, block: str, mean: np.ndarray, std: np.ndarray) -> None:
        getattr(self, f"mean_{block}").copy_(torch.as_tensor(mean, dtype=torch.float32))
        getattr(self, f"std_{block}").copy_(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Raw features in, logits out."""
        z = {b: (x[b] - getattr(self, f"mean_{b}")) / getattr(self, f"std_{b}") for b in self.net.blocks}
        return self.net(z)

    @torch.no_grad()
    def predict_proba(self, history, embedding) -> np.ndarray:
        """P(miss) for raw history features (n, 5) and MD&A embeddings (n, 384)."""
        self.eval()
        dev = getattr(self, "mean_history").device
        x = {"history": torch.as_tensor(np.asarray(history), dtype=torch.float32, device=dev),
             "embed": torch.as_tensor(np.asarray(embedding), dtype=torch.float32, device=dev)}
        return torch.sigmoid(self(x)).cpu().numpy()
