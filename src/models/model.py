"""
Text classifier for the Earnings Call Lie Detector project.

FinBERT encoder → [CLS] pooling → small MLP head → 2 logits (beat=0, miss=1).
Falls back to ``distilbert-base-uncased`` if FinBERT cannot be loaded.

The encoder can be used frozen, partially/fully fine-tuned, or adapted with
LoRA (low-rank adapters on the attention query/value projections).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODELS: Sequence[str] = ("ProsusAI/finbert", "distilbert-base-uncased")

# Attention query/value projections are named differently per architecture
LORA_TARGETS: Dict[str, List[str]] = {
    "bert": ["query", "value"],
    "distilbert": ["q_lin", "v_lin"],
}


class EarningsTextClassifier(nn.Module):
    """Transformer encoder with a two-layer classification head.

    Parameters
    ----------
    model_names : sequence of str
        Hugging Face model ids to try in order; the first that loads is used.
    max_length : int
        Tokens per chunk (the encoder's input limit).
    max_chunks : int
        1 (default): read only the first ``max_length`` tokens of each document.
        > 1: split the whole document into ``max_length``-token chunks, keep up
        to ``max_chunks`` of them evenly spaced from start to end, encode each,
        and mean-pool the chunk [CLS] vectors into one document vector.
    freeze_encoder : bool
        Freeze the whole encoder and train only the head.
    freeze_n_layers : int
        Freeze the embeddings and the first *n* transformer layers.
        Ignored when ``freeze_encoder`` is set.
    lora_r : int
        LoRA rank.  0 (default) disables LoRA.  When > 0 the encoder is frozen
        and only the low-rank adapters (plus the head) are trained;
        ``freeze_encoder`` / ``freeze_n_layers`` are then ignored.
    lora_alpha, lora_dropout : float
        LoRA scaling (effective strength ``alpha / r``) and adapter dropout.
    """

    def __init__(
        self,
        model_names: Sequence[str] = DEFAULT_MODELS,
        max_length: int = 512,
        max_chunks: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        head_dropout: float = 0.15,
        num_classes: int = 2,
        freeze_encoder: bool = False,
        freeze_n_layers: int = 0,
        lora_r: int = 0,
        lora_alpha: float = 16,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.max_chunks = max_chunks

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

        if lora_r > 0:
            if freeze_encoder or freeze_n_layers:
                logger.warning("LoRA enabled: ignoring freeze_encoder / freeze_n_layers")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=LORA_TARGETS[self.encoder.config.model_type],
                bias="none",
            )
            # Freezes every original encoder weight and injects trainable A/B matrices
            self.encoder = get_peft_model(self.encoder, lora_config)
        elif freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif freeze_n_layers > 0:
            self._freeze_bottom_layers(freeze_n_layers)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info("Trainable parameters: %s / %s (%.2f%%)", f"{n_train:,}", f"{n_total:,}", 100 * n_train / n_total)

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
        """Tokenise a batch of documents into chunk rows.

        Returns ``input_ids`` / ``attention_mask`` of shape (total_chunks, L) and
        ``doc_ids`` (total_chunks,) saying which document each chunk belongs to.
        """
        if self.max_chunks == 1:
            batch = self.tokenizer(texts, truncation=True, padding=True,
                                   max_length=self.max_length, return_tensors="pt")
            doc_ids = torch.arange(len(texts))
        else:
            # Cut every document into consecutive max_length-token chunks
            batch = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length,
                                   return_overflowing_tokens=True, return_tensors="pt")
            chunk_doc = batch.pop("overflow_to_sample_mapping")  # chunk row -> document index
            # Keep at most max_chunks per document, evenly spaced from start to end
            keep = []
            for d in range(len(texts)):
                rows = (chunk_doc == d).nonzero().flatten()
                if len(rows) > self.max_chunks:
                    rows = rows[np.unique(np.linspace(0, len(rows) - 1, self.max_chunks).round().astype(int))]
                keep.append(rows)
            keep = torch.cat(keep)
            batch = {k: v[keep] for k, v in batch.items()}
            doc_ids = chunk_doc[keep]
        return {"input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "doc_ids": doc_ids.to(device)}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                doc_ids: Optional[torch.Tensor] = None, **_: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # one [CLS] vector per chunk
        if doc_ids is None:
            doc_ids = torch.arange(len(cls), device=cls.device)
        # Mean-pool chunk vectors per document: sum them by doc id, divide by chunk count
        n_docs = int(doc_ids.max()) + 1
        summed = torch.zeros(n_docs, cls.shape[1], dtype=cls.dtype, device=cls.device).index_add_(0, doc_ids, cls)
        counts = torch.bincount(doc_ids, minlength=n_docs).clamp(min=1).unsqueeze(1).to(cls.dtype)
        return self.head(summed / counts)

    def encoder_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def head_parameters(self):
        return list(self.head.parameters())
