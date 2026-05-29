"""
data.py — PyTorch Dataset & DataLoader for D-JEPA
Loads from pre-built processed files (dataset/processed/*.pt) when available,
falls back to synthetic generation for quick smoke-tests.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from model.config import DJEPAConfig
from model.data_generation import HinglishMockGenerator, set_global_seed

PROCESSED_DIR = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "processed"
)


# ---------------------------------------------------------------------------
# GPU-resident pre-loaded loader
# ---------------------------------------------------------------------------

class GPUPreloadedLoader:
    """
    Replaces DataLoader when the full dataset fits in GPU memory.

    All samples are right-padded to cfg.max_seq_len and moved to the target
    device once at construction time.  Each iteration yields batches via pure
    GPU tensor indexing — zero CPU→GPU transfer per step, eliminating the
    DataLoader IPC / collation stall that causes near-0% GPU utilisation on
    fast accelerators like A100.

    The interface is identical to a standard DataLoader (iterable of dicts),
    so training, validation, and Fisher-computation loops need no changes.
    """

    def __init__(
        self,
        samples:    List[Dict[str, Any]],
        cfg:        DJEPAConfig,
        device:     torch.device,
        shuffle:    bool = True,
        seed:       int  = 42,
        drop_last:  bool = True,
    ) -> None:
        N = len(samples)
        T = cfg.max_seq_len
        D = cfg.input_acoustic_dim

        features     = torch.zeros(N, T, D, dtype=torch.float32)
        asr_tokens   = torch.zeros(N, T,    dtype=torch.long)
        padding_mask = torch.ones(N, T,     dtype=torch.bool)   # True = PAD
        speaker_ids  = torch.full((N, T), -1, dtype=torch.long)
        language_ids = torch.full((N, T), -1, dtype=torch.long)

        source_en: List[str] = []
        source_hi: List[str] = []

        for i, s in enumerate(samples):
            seq_len = int(s["seq_len"].item())
            t = min(seq_len, T)
            features[i,     :t, :] = s["features"][:t]
            asr_tokens[i,   :t]    = s["asr_tokens"][:t]
            padding_mask[i, :t]    = False
            if "speaker_ids" in s:
                speaker_ids[i, :t]  = s["speaker_ids"][:t]
            if "language_ids" in s:
                language_ids[i, :t] = s["language_ids"][:t]
            source_en.append(s.get("source_en", ""))
            source_hi.append(s.get("source_hi", ""))

        # One-time host→device transfer; stays resident for the whole run.
        self.features     = features.to(device)
        self.asr_tokens   = asr_tokens.to(device)
        self.padding_mask = padding_mask.to(device)
        self.speaker_ids  = speaker_ids.to(device)
        self.language_ids = language_ids.to(device)
        # Source text stays on CPU (strings can't live on GPU)
        self.source_en    = source_en
        self.source_hi    = source_hi

        self.N          = N
        self.batch_size = cfg.batch_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last
        self._rng       = torch.Generator()   # CPU RNG for randperm
        self._rng.manual_seed(seed)

        mb = (features.nbytes + asr_tokens.nbytes + padding_mask.nbytes) / 1e6
        print(f"[data] GPU-preloaded {N} samples → {mb:.0f} MB on {device}")

    def __len__(self) -> int:
        if self.drop_last:
            return self.N // self.batch_size
        return (self.N + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.shuffle:
            perm = torch.randperm(self.N, generator=self._rng)
        else:
            perm = torch.arange(self.N)

        end = (self.N // self.batch_size) * self.batch_size if self.drop_last else self.N
        for start in range(0, end, self.batch_size):
            idx = perm[start : start + self.batch_size]
            idx_list = idx.tolist()
            yield {
                "features":     self.features[idx],
                "asr_tokens":   self.asr_tokens[idx],
                "padding_mask": self.padding_mask[idx],
                "speaker_ids":  self.speaker_ids[idx],
                "language_ids": self.language_ids[idx],
                "source_en":    [self.source_en[i] for i in idx_list],
                "source_hi":    [self.source_hi[i] for i in idx_list],
            }


# ---------------------------------------------------------------------------
# Legacy variable-length Dataset (used as fallback on CPU/MPS)
# ---------------------------------------------------------------------------

class HinglishDataset(Dataset):
    """Wraps a list of variable-length sample dicts."""

    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_variable_length(
    batch: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """Right-pads all tensors to the longest sequence in the batch."""
    seq_lens = [int(s["seq_len"].item()) for s in batch]
    max_len  = max(seq_lens)
    D        = batch[0]["features"].shape[-1]
    B        = len(batch)

    features     = torch.zeros(B, max_len, D,   dtype=torch.float32)
    speaker_ids  = torch.full((B, max_len), -1, dtype=torch.long)
    language_ids = torch.full((B, max_len), -1, dtype=torch.long)
    asr_tokens   = torch.zeros(B, max_len,      dtype=torch.long)
    padding_mask = torch.ones(B, max_len,       dtype=torch.bool)

    for i, (s, T) in enumerate(zip(batch, seq_lens)):
        features[i,     :T, :] = s["features"]
        speaker_ids[i,  :T]    = s["speaker_ids"]
        language_ids[i, :T]    = s["language_ids"]
        asr_tokens[i,   :T]    = s["asr_tokens"]
        padding_mask[i, :T]    = False

    return {
        "features":     features,
        "speaker_ids":  speaker_ids,
        "language_ids": language_ids,
        "asr_tokens":   asr_tokens,
        "padding_mask": padding_mask,
        "seq_lens":     torch.tensor(seq_lens, dtype=torch.long),
        "source_en":    [s.get("source_en", "") for s in batch],
        "source_hi":    [s.get("source_hi", "") for s in batch],
    }


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def _load_real_split(split: str) -> Optional[List[Dict]]:
    """Load a pre-built split from disk. Returns None if not found."""
    path = os.path.join(PROCESSED_DIR, f"hinglish_{split}.pt")
    if os.path.exists(path):
        data = torch.load(path, weights_only=False)
        print(f"[data] Loaded {len(data)} real samples from {path}")
        return data
    return None


def build_dataloaders(
    cfg:       DJEPAConfig,
    num_train: int = 160,
    num_val:   int = 40,
    seed:      int = 42,
    device:    Optional[torch.device] = None,
) -> Tuple[Any, Any]:
    """
    Returns (train_loader, val_loader).

    When device is a CUDA device, returns GPUPreloadedLoader instances that
    keep all data GPU-resident for zero-transfer-per-step throughput.
    Falls back to standard DataLoader on CPU/MPS where GPU memory is shared.
    """
    train_samples = _load_real_split("train")
    val_samples   = _load_real_split("val")

    if train_samples is None or val_samples is None:
        print("[data] Processed data not found — using synthetic fallback")
        set_global_seed(seed)
        gen = HinglishMockGenerator(
            num_samples=num_train + num_val,
            input_acoustic_dim=cfg.input_acoustic_dim,
            num_speakers=cfg.num_speakers,
            num_languages=cfg.num_languages,
            vocab_size=cfg.vocab_size,
            code_switch_prob=cfg.code_switch_prob,
            seed=seed,
        )
        all_samples   = gen.generate_dataset()
        train_samples = all_samples[:num_train]
        val_samples   = all_samples[num_train:]

    if device is not None and device.type == "cuda":
        train_loader = GPUPreloadedLoader(
            train_samples, cfg, device, shuffle=True,  seed=seed, drop_last=True
        )
        val_loader = GPUPreloadedLoader(
            val_samples,   cfg, device, shuffle=False, seed=seed, drop_last=False
        )
        return train_loader, val_loader

    # CPU / MPS fallback: standard DataLoader with workers
    g = torch.Generator()
    g.manual_seed(seed)
    pw = cfg.num_workers > 0
    train_loader = DataLoader(
        HinglishDataset(train_samples),
        batch_size         = cfg.batch_size,
        shuffle            = True,
        collate_fn         = collate_variable_length,
        num_workers        = cfg.num_workers,
        pin_memory         = cfg.pin_memory,
        generator          = g,
        drop_last          = True,
        persistent_workers = pw,
        prefetch_factor    = 4 if pw else None,
    )
    val_loader = DataLoader(
        HinglishDataset(val_samples),
        batch_size         = cfg.batch_size,
        shuffle            = False,
        collate_fn         = collate_variable_length,
        num_workers        = cfg.num_workers,
        pin_memory         = cfg.pin_memory,
        persistent_workers = pw,
        prefetch_factor    = 4 if pw else None,
    )
    return train_loader, val_loader


def build_test_loader(
    cfg: DJEPAConfig, device: Optional[torch.device] = None
) -> Optional[Any]:
    """Returns a test loader if hinglish_test.pt exists."""
    test_samples = _load_real_split("test")
    if test_samples is None:
        return None
    if device is not None and device.type == "cuda":
        return GPUPreloadedLoader(
            test_samples, cfg, device, shuffle=False, seed=0, drop_last=False
        )
    return DataLoader(
        HinglishDataset(test_samples),
        batch_size  = cfg.batch_size,
        shuffle     = False,
        collate_fn  = collate_variable_length,
        num_workers = cfg.num_workers,
        pin_memory  = cfg.pin_memory,
    )


def load_real_kg() -> Optional[Dict]:
    """Load pre-built medical KG. Returns None if not found."""
    path = os.path.join(PROCESSED_DIR, "primekg_real.pt")
    if os.path.exists(path):
        kg = torch.load(path, weights_only=False)
        print(f"[data] Loaded real KG: {kg['num_nodes']} nodes from {path}")
        return kg
    return None
