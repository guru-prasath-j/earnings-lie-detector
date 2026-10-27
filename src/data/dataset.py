"""
PyTorch Dataset for the Earnings Call Lie Detector project.

Brings together transcripts, audio features, and ground truth labels
into a single Dataset that can be consumed by DataLoaders for training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EarningsCallDataset(Dataset):
    """Dataset that joins transcripts, audio feature vectors, and binary labels.

    Each sample is keyed by ``(ticker, date)`` where *date* is the earnings
    announcement date from labels.csv (e.g. ``"2023-07-27"``).  The transcript
    attached to a sample is the most recent filing made *before* that
    announcement (within ``max_transcript_gap_days``), so the text predates the
    result it is labelled with — a 10-Q describing the quarter that was just
    announced would leak the answer.  Missing modalities are allowed: the
    returned dict carries boolean flags ``has_transcript`` and
    ``has_audio_features`` so downstream code can mask or skip as needed.

    Parameters
    ----------
    data_root : str | Path
        Project root that contains ``data/``.
    labels_path : str | Path | None
        Override for the labels CSV.  Defaults to
        ``<data_root>/data/processed/labels.csv``.
    audio_features_path : str | Path | None
        Override for the audio-features file (.npz preferred, .csv fallback).
        Defaults to ``<data_root>/data/processed/audio_features``.
    transcripts_dir : str | Path | None
        Override for the transcripts directory.  Defaults to
        ``<data_root>/data/raw/transcripts``.
    max_transcript_gap_days : int
        Maximum days between a filing and the next earnings announcement for
        the filing to be attached to that announcement's label.
    min_transcript_gap_days : int
        Minimum days between them.  0 for 10-Qs (filed weeks after their own
        announcement).  Use ~30 for earnings press releases, which are filed *on*
        announcement day and state that day's EPS: if the label date is off by a
        day, a 0-day minimum could pair a release with its own result.
    use_audio : bool
        Load audio features and return them in each sample.  Off by default:
        the dataset is then text-only and samples without a transcript are
        dropped.
    """

    def __init__(
        self,
        data_root: str | Path = ".",
        labels_path: Optional[str | Path] = None,
        audio_features_path: Optional[str | Path] = None,
        transcripts_dir: Optional[str | Path] = None,
        max_transcript_gap_days: int = 120,
        min_transcript_gap_days: int = 0,
        use_audio: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.max_transcript_gap_days = max_transcript_gap_days
        self.min_transcript_gap_days = min_transcript_gap_days
        self.use_audio = use_audio

        # --- Labels ----------------------------------------------------------
        labels_path = Path(labels_path) if labels_path else self.data_root / "data" / "processed" / "labels.csv"
        self.labels_df = pd.read_csv(labels_path)
        # Expect columns: ticker, date, label  (label ∈ {0, 1})
        required_cols = {"ticker", "date", "label"}
        if not required_cols.issubset(self.labels_df.columns):
            raise ValueError(f"labels.csv must contain columns {required_cols}, got {set(self.labels_df.columns)}")
        self.labels_df["date"] = self.labels_df["date"].astype(str)

        # --- Audio features --------------------------------------------------
        af_base = Path(audio_features_path) if audio_features_path else self.data_root / "data" / "processed" / "audio_features"
        self.audio_features: Dict[Tuple[str, str], np.ndarray] = {}
        self._audio_dim: int = 0
        if self.use_audio:
            self._load_audio_features(af_base)

        # --- Transcripts -----------------------------------------------------
        transcripts_dir = Path(transcripts_dir) if transcripts_dir else self.data_root / "data" / "raw" / "transcripts"
        self.transcripts: Dict[Tuple[str, str], str] = {}
        self._load_transcripts(transcripts_dir)

        # --- Build unified index ---------------------------------------------
        # The index is the union of all keys present in labels.
        self.samples: List[Dict[str, Any]] = []
        self._build_index()

        # --- Stats -----------------------------------------------------------
        self._print_stats()

    # -- loaders --------------------------------------------------------------

    def _load_audio_features(self, base: Path) -> None:
        """Load audio features from .npz (preferred) or .csv fallback."""
        npz_path = base.with_suffix(".npz")
        csv_path = base.with_suffix(".csv")

        if npz_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            # Expect arrays: 'features' (N x D), 'tickers' (N,), 'dates' (N,)
            features = data["features"]
            tickers = data["tickers"]
            dates = data["dates"]
            self._audio_dim = features.shape[1] if features.ndim == 2 else 0
            for feat, t, d in zip(features, tickers, dates):
                self.audio_features[(str(t), str(d))] = feat.astype(np.float32)
            logger.info("Loaded %d audio feature vectors from %s", len(self.audio_features), npz_path)

        elif csv_path.exists():
            df = pd.read_csv(csv_path)
            feature_cols = [c for c in df.columns if c not in ("ticker", "date")]
            self._audio_dim = len(feature_cols)
            for _, row in df.iterrows():
                key = (str(row["ticker"]), str(row["date"]))
                self.audio_features[key] = row[feature_cols].values.astype(np.float32)
            logger.info("Loaded %d audio feature vectors from %s", len(self.audio_features), csv_path)

        else:
            logger.warning("No audio features found at %s (.npz or .csv)", base)

    def _load_transcripts(self, transcripts_dir: Path) -> None:
        """Load transcript texts keyed by ``(ticker, filing_date)``.

        Supports ``{ticker}/*.json`` and flat ``{ticker}.json`` files, each
        holding a single record or a list of records (scraper.py output).
        Entries starting with ``_`` (e.g. ``_all_transcripts.json``) are skipped.
        """
        if not transcripts_dir.exists():
            logger.warning("Transcripts directory not found: %s", transcripts_dir)
            return

        files: List[Tuple[str, Path]] = []
        for entry in sorted(transcripts_dir.iterdir()):
            if entry.name.startswith("_"):
                continue
            if entry.is_dir():
                files.extend((entry.name.upper(), p) for p in sorted(entry.glob("*.json")))
            elif entry.suffix == ".json":
                files.append((entry.stem.upper(), entry))

        for ticker, json_file in files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("Skipping %s: %s", json_file, exc)
                continue
            for doc in data if isinstance(data, list) else [data]:
                if not isinstance(doc, dict):
                    continue
                text = doc.get("text") or doc.get("transcript") or doc.get("mda_text") or ""
                date = str(doc.get("date", json_file.stem))[:10]
                if text:
                    self.transcripts[(ticker, date)] = text

        # Per-ticker sorted filing dates, for matching to announcement dates
        self._transcript_dates: Dict[str, List[pd.Timestamp]] = {}
        for ticker, date in self.transcripts:
            ts = pd.to_datetime(date, errors="coerce")
            if pd.notna(ts):
                self._transcript_dates.setdefault(ticker, []).append(ts)
        for dates in self._transcript_dates.values():
            dates.sort()

        logger.info("Loaded %d transcripts", len(self.transcripts))

    def _match_transcript(self, ticker: str, date: str) -> Optional[Tuple[str, str]]:
        """Return the transcript key for the latest filing at least ``min_transcript_gap_days``
        (and strictly) before *date*."""
        ann = pd.to_datetime(date[:10], errors="coerce")
        if pd.isna(ann):
            return None
        best = None
        for ts in self._transcript_dates.get(ticker, []):
            if ts >= ann or (ann - ts).days < self.min_transcript_gap_days:
                break
            best = ts
        if best is None or (ann - best).days > self.max_transcript_gap_days:
            return None
        return (ticker, best.strftime("%Y-%m-%d"))

    def _build_index(self) -> None:
        """Create the unified sample list from labels, attaching available modalities."""
        skipped_nan = 0
        skipped_no_text = 0
        for _, row in self.labels_df.iterrows():
            # Skip rows with missing labels
            if pd.isna(row.get("label")):
                skipped_nan += 1
                continue
            key = (str(row["ticker"]).upper(), str(row["date"]))
            transcript_key = self._match_transcript(*key)
            # Text-only mode: a sample without a transcript has no input
            if not self.use_audio and transcript_key is None:
                skipped_no_text += 1
                continue
            sample: Dict[str, Any] = {
                "ticker": key[0],
                "date": key[1],
                "label": int(row["label"]),
                "transcript_key": transcript_key,
                "has_transcript": transcript_key is not None,
                "has_audio_features": key in self.audio_features,
            }
            self.samples.append(sample)
        if skipped_nan:
            logger.info("Skipped %d rows with missing labels", skipped_nan)
        if skipped_no_text:
            logger.info("Skipped %d rows with no matching transcript (text-only mode)", skipped_no_text)

    def _print_stats(self) -> None:
        n = len(self.samples)
        if n == 0:
            logger.warning("Dataset is empty!")
            return

        labels = [s["label"] for s in self.samples]
        n_pos = sum(labels)
        n_neg = n - n_pos
        n_has_text = sum(s["has_transcript"] for s in self.samples)
        n_has_audio = sum(s["has_audio_features"] for s in self.samples)
        n_both = sum(s["has_transcript"] and s["has_audio_features"] for s in self.samples)

        audio_stats = (
            f"  Audio features available: {n_has_audio}/{n}\n"
            f"  Both modalities: {n_both}/{n}\n"
            f"  Audio feature dim: {self._audio_dim}\n"
            if self.use_audio else "  Audio: disabled (text-only)\n"
        )
        stats = (
            f"\n{'=' * 50}\n"
            f"EarningsCallDataset — {n} samples\n"
            f"  Labels:  met=0 → {n_neg}  |  missed=1 → {n_pos}  "
            f"({n_pos / n * 100:.1f}% positive)\n"
            f"  Transcripts available: {n_has_text}/{n}\n"
            f"{audio_stats}"
            f"{'=' * 50}"
        )
        print(stats)
        logger.info(stats)

    # -- Dataset interface ----------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        key = (sample["ticker"], sample["date"])

        # Transcript
        text = self.transcripts.get(sample["transcript_key"], "") if sample["transcript_key"] else ""

        item: Dict[str, Any] = {
            "text": text,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "ticker": sample["ticker"],
            "date": sample["date"],
            "has_transcript": sample["has_transcript"],
        }

        if self.use_audio:
            # Audio features — zero vector when missing
            if key in self.audio_features:
                item["audio_features"] = torch.from_numpy(self.audio_features[key])
            else:
                item["audio_features"] = torch.zeros(max(self._audio_dim, 1), dtype=torch.float32)
            item["has_audio_features"] = sample["has_audio_features"]

        return item

    # -- Convenience ----------------------------------------------------------

    @property
    def audio_dim(self) -> int:
        """Dimensionality of the audio feature vector."""
        return self._audio_dim

    def get_labels(self) -> np.ndarray:
        """Return all labels as a numpy array (useful for stratification)."""
        return np.array([s["label"] for s in self.samples], dtype=np.int64)

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for cross-entropy loss.

        Returns
        -------
        torch.Tensor
            Shape ``(2,)`` — weights for class 0 (met) and class 1 (missed).
        """
        labels = self.get_labels()
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        # Inverse frequency, normalised so they sum to num_classes
        weights = len(labels) / (len(counts) * counts + 1e-8)
        return torch.tensor(weights, dtype=torch.float32)

    def get_feature_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std of audio features across the dataset.

        Returns
        -------
        mean : torch.Tensor, shape ``(audio_dim,)``
        std  : torch.Tensor, shape ``(audio_dim,)``
        """
        if not self.audio_features:
            d = max(self._audio_dim, 1)
            return torch.zeros(d), torch.ones(d)

        all_feats = np.stack(list(self.audio_features.values()))
        mean = torch.from_numpy(all_feats.mean(axis=0).astype(np.float32))
        std = torch.from_numpy(all_feats.std(axis=0).astype(np.float32))
        std = torch.clamp(std, min=1e-8)  # avoid division by zero
        return mean, std


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that keeps text as a list of strings."""
    out = {
        "text": [item["text"] for item in batch],
        "label": torch.stack([item["label"] for item in batch]),
        "ticker": [item["ticker"] for item in batch],
        "date": [item["date"] for item in batch],
        "has_transcript": [item["has_transcript"] for item in batch],
    }
    if "audio_features" in batch[0]:
        out["audio_features"] = torch.stack([item["audio_features"] for item in batch])
        out["has_audio_features"] = [item["has_audio_features"] for item in batch]
    return out


# ---------------------------------------------------------------------------
# Dataloader helper
# ---------------------------------------------------------------------------

def get_dataloaders(
    dataset: EarningsCallDataset,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    split: str = "temporal",
    val_start: str = "2024-01-01",
    test_start: str = "2025-01-01",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders.

    Parameters
    ----------
    dataset : EarningsCallDataset
    batch_size : int
    val_split : float
        Fraction of data for validation (``split="random"`` only).
    test_split : float
        Fraction of data for test (``split="random"`` only).
    seed : int
    num_workers : int
    split : {"temporal", "random"}
        ``"temporal"`` (default): by announcement date — train before
        *val_start*, val in [*val_start*, *test_start*), test from
        *test_start* on.  ``"random"``: stratified random split, which lets a
        company's later quarters train a model tested on its earlier ones.
    val_start, test_start : str
        ``YYYY-MM-DD`` cut-offs for the temporal split.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    n = len(dataset)
    labels = dataset.get_labels()

    if split == "temporal":
        if not val_start < test_start:
            raise ValueError(f"val_start ({val_start}) must be before test_start ({test_start})")
        train_idx, val_idx, test_idx = [], [], []
        for i, s in enumerate(dataset.samples):
            d = s["date"][:10]
            if d < val_start:
                train_idx.append(i)
            elif d < test_start:
                val_idx.append(i)
            else:
                test_idx.append(i)
    elif split == "random":
        # Stratified split via sklearn-style logic (no sklearn dependency)
        rng = np.random.RandomState(seed)
        indices = np.arange(n)

        # Separate by class, shuffle within each, then interleave
        class_indices: Dict[int, np.ndarray] = {}
        for cls in np.unique(labels):
            cls_idx = indices[labels == cls]
            rng.shuffle(cls_idx)
            class_indices[cls] = cls_idx

        train_idx, val_idx, test_idx = [], [], []
        for cls, cls_idx in class_indices.items():
            nc = len(cls_idx)
            n_test = max(1, int(nc * test_split))
            n_val = max(1, int(nc * val_split))

            test_idx.extend(cls_idx[:n_test])
            val_idx.extend(cls_idx[n_test : n_test + n_val])
            train_idx.extend(cls_idx[n_test + n_val :])
    else:
        raise ValueError(f"Unknown split '{split}' (expected 'temporal' or 'random')")

    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if not idx:
            raise ValueError(f"{split} split produced an empty {name} set")

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    print(
        f"Split ({split}) — train: {len(train_set)} ({int(labels[train_idx].sum())} miss), "
        f"val: {len(val_set)} ({int(labels[val_idx].sum())} miss), "
        f"test: {len(test_set)} ({int(labels[test_idx].sum())} miss)"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Sanity-check the EarningsCallDataset")
    parser.add_argument("--data-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for test loader")
    parser.add_argument("--use-audio", action="store_true", help="Also load audio features (off by default)")
    args = parser.parse_args()

    ds = EarningsCallDataset(data_root=args.data_root, use_audio=args.use_audio)
    print(f"\nClass weights: {ds.get_class_weights()}")
    if args.use_audio:
        mean, std = ds.get_feature_stats()
        print(f"Audio feature mean: {mean[:5]}...  std: {std[:5]}...")

    if len(ds) > 0:
        train_dl, val_dl, test_dl = get_dataloaders(ds, batch_size=args.batch_size)
        batch = next(iter(train_dl))
        print(f"\nSample batch keys: {list(batch.keys())}")
        if args.use_audio:
            print(f"  audio_features shape: {batch['audio_features'].shape}")
        print(f"  labels: {batch['label']}")
        print(f"  tickers: {batch['ticker']}")
        print(f"  text[0][:100]: {batch['text'][0][:100]}...")