"""
Text classifier for the Earnings Call Lie Detector project.

FinBERT encoder → [CLS] pooling → small MLP head → 2 logits (beat=0, miss=1).
Falls back to ``distilbert-base-uncased`` if FinBERT cannot be loaded.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODELS: Sequence[str] = ("ProsusAI/finbert", "distilbert-base-uncased")


class EarningsTextClassifier(nn.Module):
    """Transformer encoder with a two-layer classification head.

    Parameters
    ----------
    model_names : sequence of str
        Hugging Face model ids to try in order; the first that loads is used.
    max_length : int
        Token limit per document (MD&A text beyond this is truncated).
    freeze_encoder : bool
        Freeze the whole encoder and train only the head.
    freeze_n_layers : int
        Freeze the embeddings and the first *n* transformer layers.
        Ignored when ``freeze_encoder`` is set.
    """

    def __init__(
        self,
        model_names: Sequence[str] = DEFAULT_MODELS,
        max_length: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        head_dropout: float = 0.15,
        num_classes: int = 2,
        freeze_encoder: bool = False,
        freeze_n_layers: int = 0,
    ) -> None:
        super().__init__()
        self.max_length = max_length

        last_exc: Optional[Exception] = None
        for name in model_names:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name)
                self.encoder = AutoModel.from_pretrained(name)
                self.model_name = name
                break
            except Exception as exc:  # network / missing weights
                logger.warning("Could not load %s: %s", name, exc)
                last_exc = exc
        else:
            raise RuntimeError(f"No encoder could be loaded from {list(model_names)}") from last_exc
        logger.info("Loaded encoder %s", self.model_name)

        enc_dim = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif freeze_n_layers > 0:
            self._freeze_bottom_layers(freeze_n_layers)

    def _freeze_bottom_layers(self, n: int) -> None:
        """Freeze embeddings + the first *n* transformer blocks."""
        enc = self.encoder
        # BERT-style: enc.embeddings / enc.encoder.layer; DistilBERT: enc.embeddings / enc.transformer.layer
        layers = getattr(getattr(enc, "encoder", None), "layer", None)
        if layers is None:
            layers = getattr(getattr(enc, "transformer", None), "layer", [])
        for p in enc.embeddings.parameters():
            p.requires_grad = False
        for layer in list(layers)[:n]:
            for p in layer.parameters():
                p.requires_grad = False
        logger.info("Froze embeddings + %d/%d encoder layers", min(n, len(layers)), len(layers))

    def tokenize(self, texts: List[str], device: torch.device) -> dict:
        batch = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in batch.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [CLS] pooling
        return self.head(cls)

    def encoder_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def head_parameters(self):
        return list(self.head.parameters())
