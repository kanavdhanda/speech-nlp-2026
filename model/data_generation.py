"""
data_generation.py — Deterministic Mock Dataset Generator
Produces synthetic tensors that structurally mimic:
  1. sonexis-ai/hinglish-code-switched-conversations-v1
     → (batch, seq_len, hidden_dim) acoustic features + speaker/language labels
  2. mims-harvard/PrimeKG
     → A lightweight patient knowledge graph as a Python dict
All generation is seeded for full reproducibility.
"""

from __future__ import annotations

import os
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Any


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """Lock every PRNG used downstream so runs are byte-identical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 1. Hinglish Code-Switched Acoustic Features
# ---------------------------------------------------------------------------

class HinglishMockGenerator:
    """
    Mimics the structure of the Hinglish code-switched corpus.

    Each sample is a variable-length sequence of log-Mel acoustic frames
    (shape: seq_len × input_acoustic_dim) where the identity of the active
    speaker and language can change at each frame — modelling real
    code-switching behaviour.

    Labels returned:
      speaker_ids  : LongTensor [seq_len]  values in {0, 1, 2}
      language_ids : LongTensor [seq_len]  values in {0=Hindi, 1=English}
      asr_tokens   : LongTensor [seq_len]  synthetic BPE token indices
    """

    def __init__(
        self,
        num_samples: int = 200,
        min_len: int = 50,
        max_len: int = 200,
        input_acoustic_dim: int = 80,
        num_speakers: int = 3,
        num_languages: int = 2,
        vocab_size: int = 8192,
        code_switch_prob: float = 0.4,
        seed: int = 42,
    ) -> None:
        self.num_samples       = num_samples
        self.min_len           = min_len
        self.max_len           = max_len
        self.input_acoustic_dim = input_acoustic_dim
        self.num_speakers      = num_speakers
        self.num_languages     = num_languages
        self.vocab_size        = vocab_size
        self.code_switch_prob  = code_switch_prob
        self.rng               = np.random.default_rng(seed)

    def _generate_one(self) -> Dict[str, torch.Tensor]:
        """Generate a single variable-length sample."""
        seq_len = int(self.rng.integers(self.min_len, self.max_len + 1))

        # Acoustic features: standard Gaussian centred at 0, scaled like
        # normalised log-Mel spectrograms (mean≈0, std≈1 after CMVN)
        features = torch.from_numpy(
            self.rng.normal(0.0, 1.0, (seq_len, self.input_acoustic_dim)).astype(np.float32)
        )

        # Speaker identity: each new utterance segment is assigned a speaker.
        # A new speaker boundary appears with probability proportional to the
        # inverse of the mean segment length (≈10 frames).
        speaker_changes = self.rng.random(seq_len) < 0.1
        speaker_ids     = np.zeros(seq_len, dtype=np.int64)
        current_speaker = int(self.rng.integers(self.num_speakers))
        for t in range(seq_len):
            if speaker_changes[t]:
                current_speaker = int(self.rng.integers(self.num_speakers))
            speaker_ids[t] = current_speaker

        # Language identity: models real code-switching where language
        # flips with probability code_switch_prob at each frame boundary.
        language_ids    = np.zeros(seq_len, dtype=np.int64)
        current_lang    = int(self.rng.integers(self.num_languages))
        for t in range(seq_len):
            if self.rng.random() < self.code_switch_prob:
                current_lang = 1 - current_lang          # binary flip
            language_ids[t] = current_lang

        # ASR token targets: uniform random BPE indices; in real data these
        # would come from a forced-alignment phone-to-token mapping.
        asr_tokens = torch.from_numpy(
            self.rng.integers(0, self.vocab_size, size=seq_len).astype(np.int64)
        )

        return {
            "features":     features,                          # (T, D)
            "speaker_ids":  torch.from_numpy(speaker_ids),    # (T,)
            "language_ids": torch.from_numpy(language_ids),   # (T,)
            "asr_tokens":   asr_tokens,                       # (T,)
            "seq_len":      torch.tensor(seq_len, dtype=torch.long),
        }

    def generate_dataset(self) -> List[Dict[str, torch.Tensor]]:
        """Return a list of sample dicts — consumed by HinglishDataset."""
        return [self._generate_one() for _ in range(self.num_samples)]


# ---------------------------------------------------------------------------
# 2. PrimeKG Mock Patient Knowledge Graph
# ---------------------------------------------------------------------------

class PrimeKGMockGenerator:
    """
    Produces a lightweight mock of the PrimeKG biomedical knowledge graph.

    The real PrimeKG has ~129k nodes and ~4M edges across 10 node types.
    For SGG grounding we only need:
      - node_ids     : list of string node identifiers
      - node_type    : mapping node_id → type (drug / disease / gene / symptom)
      - embeddings   : FloatTensor [num_nodes, kg_embed_dim]
      - contraindications : dict mapping drug_id → list[disease_id]
      - physiological_nodes : list of disease/symptom node_ids

    The embeddings are used by the SGG gate to compute L2 distances between
    the world model's projected future state ẑ_{t+1} and KG node vectors.
    """

    # Canonical node types present in PrimeKG
    NODE_TYPES = ["drug", "disease", "gene", "symptom", "pathway"]

    # Small curated vocabularies for realistic node naming
    DRUGS     = ["metformin", "lisinopril", "atorvastatin", "amoxicillin",
                 "omeprazole", "amlodipine", "gabapentin", "sertraline",
                 "furosemide", "warfarin", "levothyroxine", "prednisone"]
    DISEASES  = ["type2_diabetes", "hypertension", "atrial_fibrillation",
                 "heart_failure", "chronic_kidney_disease", "depression",
                 "epilepsy", "COPD", "hypothyroidism", "osteoporosis"]
    GENES     = ["CYP2C19", "APOE", "BRCA1", "TP53", "EGFR", "KRAS",
                 "VEGFA", "TNF", "IL6", "ACE"]
    SYMPTOMS  = ["dyspnea", "chest_pain", "peripheral_edema", "fatigue",
                 "palpitations", "syncope", "hyperglycemia", "bradycardia"]

    def __init__(self, num_nodes: int = 64, kg_embed_dim: int = 128, seed: int = 42) -> None:
        self.num_nodes    = num_nodes
        self.kg_embed_dim = kg_embed_dim
        self.rng          = np.random.default_rng(seed)

    def generate(self) -> Dict[str, Any]:
        """Return the mock KG dict."""
        all_named_nodes = (
            self.DRUGS + self.DISEASES + self.GENES + self.SYMPTOMS
        )
        # Cycle through named nodes, then pad with synthetic IDs
        node_ids: List[str] = []
        for i in range(self.num_nodes):
            if i < len(all_named_nodes):
                node_ids.append(all_named_nodes[i])
            else:
                node_ids.append(f"synthetic_node_{i}")

        # Assign type labels
        node_type: Dict[str, str] = {}
        for nid in node_ids:
            if nid in self.DRUGS:
                node_type[nid] = "drug"
            elif nid in self.DISEASES:
                node_type[nid] = "disease"
            elif nid in self.GENES:
                node_type[nid] = "gene"
            elif nid in self.SYMPTOMS:
                node_type[nid] = "symptom"
            else:
                node_type[nid] = self.NODE_TYPES[int(self.rng.integers(len(self.NODE_TYPES)))]

        # Node embeddings: unit-normalised so L2 distance ∈ [0, 2]
        raw_embeddings = self.rng.standard_normal((self.num_nodes, self.kg_embed_dim)).astype(np.float32)
        norms          = np.linalg.norm(raw_embeddings, axis=1, keepdims=True) + 1e-8
        embeddings     = torch.from_numpy(raw_embeddings / norms)   # (num_nodes, D)

        # Contraindication edges: each drug contraindicates 1–3 random diseases.
        # This encodes the PrimeKG "contraindication" relation type.
        drug_nodes    = [n for n in node_ids if node_type[n] == "drug"]
        disease_nodes = [n for n in node_ids if node_type[n] == "disease"]
        contraindications: Dict[str, List[str]] = {}
        for drug in drug_nodes:
            k = int(self.rng.integers(1, min(4, len(disease_nodes) + 1)))
            chosen = self.rng.choice(disease_nodes, size=k, replace=False).tolist()
            contraindications[drug] = chosen

        # Physiological nodes: all disease + symptom nodes
        physiological_nodes = [
            n for n in node_ids if node_type[n] in ("disease", "symptom")
        ]

        return {
            "node_ids":             node_ids,
            "node_type":            node_type,
            "embeddings":           embeddings,                # FloatTensor [N, D]
            "contraindications":    contraindications,
            "physiological_nodes":  physiological_nodes,
            "num_nodes":            self.num_nodes,
            "kg_embed_dim":         self.kg_embed_dim,
        }


# ---------------------------------------------------------------------------
# Convenience: save / load generated data
# ---------------------------------------------------------------------------

def generate_and_save(
    out_dir: str = "data",
    num_samples: int = 200,
    num_kg_nodes: int = 64,
    kg_embed_dim: int = 128,
    seed: int = 42,
) -> None:
    """
    Generate all mock data and persist to disk as .pt files.
    Re-running with the same seed yields identical files.
    """
    set_global_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # --- Hinglish dataset ---
    hgen     = HinglishMockGenerator(num_samples=num_samples, seed=seed)
    samples  = hgen.generate_dataset()
    torch.save(samples, os.path.join(out_dir, "hinglish_samples.pt"))
    print(f"[data_generation] Saved {len(samples)} Hinglish samples → {out_dir}/hinglish_samples.pt")

    # --- PrimeKG ---
    kgen = PrimeKGMockGenerator(num_nodes=num_kg_nodes, kg_embed_dim=kg_embed_dim, seed=seed)
    kg   = kgen.generate()
    torch.save(kg, os.path.join(out_dir, "primekg_mock.pt"))
    print(f"[data_generation] Saved PrimeKG mock ({num_kg_nodes} nodes) → {out_dir}/primekg_mock.pt")


def load_hinglish(path: str = "data/hinglish_samples.pt") -> List[Dict[str, torch.Tensor]]:
    return torch.load(path, weights_only=False)


def load_primekg(path: str = "data/primekg_mock.pt") -> Dict[str, Any]:
    return torch.load(path, weights_only=False)


if __name__ == "__main__":
    generate_and_save()
