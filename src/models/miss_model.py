"""
Earnings-miss model for the Hugging Face Hub — self-contained (numpy, torch, huggingface_hub,
plus ``guidance_rules.py`` from the same folder), so both files can be shipped next to the weights.

A logistic regression on 25 features is just ``Linear(25 → 1)`` + sigmoid.  Wrapping it in an
``nn.Module`` with ``PyTorchModelHubMixin`` gives the same ``save_pretrained`` / ``from_pretrained``
round trip (config.json + model.safetensors) as the earlier fusion model.

Features (raw, in ``features`` order; the model standardises them with training-set statistics
stored as buffers):

- 5 history features from the company's earlier EPS surprises (``history_vector``)
- one-hot GICS sector (``sector_vector``)
- 10 rule-based guidance features of the latest earnings press release (``guidance_rules``)

    from miss_model import HubMissModel
    model = HubMissModel.from_pretrained("<user>/lie-detect")
    p = model.predict(past_surprises=[3.1, -2.0, 5.4, 1.2], sector="Industrials", release_text=text)
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from guidance_rules import GUIDANCE_COLS, guidance_vector

HISTORY_COLS = ["prev_miss", "prev_surprise", "miss_rate_4q", "mean_surprise_4q", "std_surprise_4q"]


def history_vector(past_surprises: Sequence[float]) -> List[float]:
    """History features from earlier EPS surprises in % (oldest → newest), as in training.

    Surprise % = (actual − estimate) / |estimate| × 100; a miss is a negative surprise.
    Uses the last 4 announcements; missing values (a company's first quarters) become 0.
    """
    s = np.clip(np.asarray(past_surprises, dtype=float), -100, 100)[-4:]
    if len(s) == 0:
        return [0.0] * 5
    miss = (s < 0).astype(float)
    return [float(miss[-1]), float(s[-1]), float(miss.mean()), float(s.mean()),
            float(s.std(ddof=1)) if len(s) >= 2 else 0.0]


class HubMissModel(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="pytorch",
    tags=["tabular-classification", "finance", "earnings", "sec-filings"],
    repo_url="https://github.com/guru-prasath-j/earnings-lie-detector",
):
    """Standardise → Linear → logit.  ``sigmoid(logit)`` = P(miss) (class-weighted, so not calibrated).

    Every ``__init__`` argument is JSON-serialisable: the mixin writes them to config.json.
    """

    def __init__(self, features: List[str], sectors: List[str], C: float = 1.0) -> None:
        super().__init__()
        self.features, self.sectors, self.C = list(features), list(sectors), C
        self.linear = nn.Linear(len(features), 1)
        self.register_buffer("mean", torch.zeros(len(features)))
        self.register_buffer("std", torch.ones(len(features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw features (n, d) in, logits (n,) out."""
        return self.linear((x - self.mean) / self.std).squeeze(1)

    def sector_vector(self, sector: str) -> List[float]:
        if sector not in self.sectors:
            raise ValueError(f"unknown sector {sector!r}; expected one of {self.sectors}")
        return [float(sector == s) for s in self.sectors]

    def feature_vector(self, past_surprises: Sequence[float], sector: str, release_text: str) -> List[float]:
        g = guidance_vector(release_text)
        return history_vector(past_surprises) + self.sector_vector(sector) + [g[c] for c in GUIDANCE_COLS]

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        """P(miss) for a raw feature matrix (n, d) in ``features`` order."""
        self.eval()
        x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.mean.device)
        return torch.sigmoid(self(x)).cpu().numpy()

    def predict(self, past_surprises: Sequence[float], sector: str, release_text: str) -> float:
        """P(miss at the next announcement) for one company, from its history and latest press release."""
        return float(self.predict_proba([self.feature_vector(past_surprises, sector, release_text)])[0])
