"""
Earnings-miss model for the Hugging Face Hub — self-contained (numpy, torch, huggingface_hub, plus
``guidance_rules.py`` and, for the market blocks, ``market_features.py`` with pandas from the same
folder), so these files can be shipped next to the weights.

A logistic regression is just ``Linear(d → 1)`` + sigmoid.  Wrapping it in an ``nn.Module`` with
``PyTorchModelHubMixin`` gives the same ``save_pretrained`` / ``from_pretrained`` round trip
(config.json + model.safetensors) as the earlier fusion model.

Feature blocks (``blocks`` in the config, in this order; raw values, standardised inside the model):

- ``history``  5 features from the company's earlier EPS surprises (``history_vector``)
- ``sector``   one-hot GICS sector
- ``guidance`` 10 rule-based features of the latest earnings press release (``guidance_rules``)
- ``analyst``  analyst rating / price-target actions in the 90 days before (``market_features``)
- ``price``    the stock's run-up vs SPY and its volatility (``market_features``)

    from miss_model import HubMissModel
    model = HubMissModel.from_pretrained("<user>/lie-detect")
    p = model.predict(past_surprises=[3.1, -2.0, 5.4, 1.2], sector="Industrials", release_text=text,
                      ticker="CAT", announcement_date="2026-11-04")   # market data fetched from Yahoo
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from guidance_rules import GUIDANCE_COLS, guidance_vector

HISTORY_COLS = ["prev_miss", "prev_surprise", "miss_rate_4q", "mean_surprise_4q", "std_surprise_4q"]
MARKET_BLOCKS = ("analyst", "price")


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
    ``fills`` holds the training-time values for features that can be missing (a new listing's price history).
    """

    def __init__(self, features: List[str], sectors: List[str], C: float = 1.0,
                 blocks: Optional[List[str]] = None, fills: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        self.features, self.sectors, self.C = list(features), list(sectors), C
        self.blocks = list(blocks or ["history", "sector", "guidance"])
        self.fills = dict(fills or {})
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

    def feature_vector(self, past_surprises: Sequence[float], sector: str, release_text: str,
                       ticker: Optional[str] = None, announcement_date: Optional[str] = None,
                       market_inputs=None) -> List[float]:
        """Raw features in ``features`` order.

        Market blocks need ``announcement_date`` and either ``market_inputs`` = (analyst actions, prices,
        SPY prices) as ``market_features.market_vector`` takes them, or a ``ticker`` to fetch them from Yahoo.
        """
        vals: Dict[str, float] = {}
        if "history" in self.blocks:
            vals.update(zip(HISTORY_COLS, history_vector(past_surprises)))
        if "sector" in self.blocks:
            vals.update({f"sec_{s}": v for s, v in zip(self.sectors, self.sector_vector(sector))})
        if "guidance" in self.blocks:
            vals.update(guidance_vector(release_text))
        if any(b in self.blocks for b in MARKET_BLOCKS):
            from market_features import market_vector, yahoo_inputs
            if announcement_date is None:
                raise ValueError("this model uses market data: pass announcement_date (YYYY-MM-DD)")
            if market_inputs is None:
                if ticker is None:
                    raise ValueError("pass ticker (fetched from Yahoo Finance) or market_inputs")
                market_inputs = yahoo_inputs(ticker)
            m = market_vector(*market_inputs, announcement_date)
            vals.update({k: (self.fills.get(k, 0.0) if np.isnan(v) else v) for k, v in m.items()})
        return [float(vals[f]) for f in self.features]

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        """P(miss) for a raw feature matrix (n, d) in ``features`` order."""
        self.eval()
        x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.mean.device)
        return torch.sigmoid(self(x)).cpu().numpy()

    def predict(self, past_surprises: Sequence[float], sector: str, release_text: str, **market) -> float:
        """P(miss at the next announcement) for one company.  ``market``: ticker / announcement_date / market_inputs."""
        return float(self.predict_proba([self.feature_vector(past_surprises, sector, release_text, **market)])[0])
