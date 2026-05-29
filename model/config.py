"""
config.py — D-JEPA Global Configuration
All hyperparameters, ablation toggles, and architectural constants.
Centralising config in a frozen dataclass guarantees reproducibility
across ablation sweeps; every flag here corresponds to exactly one
architectural component that can be toggled off.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Model size presets (analogous to ViT-S / ViT-B / ViT-L naming)
# ---------------------------------------------------------------------------
MODEL_SIZES = {
    "tiny":  dict(hidden_dim=128, num_heads=4,  num_layers=4,  ffn_dim=512),
    "small": dict(hidden_dim=256, num_heads=8,  num_layers=6,  ffn_dim=1024),
    "base":  dict(hidden_dim=512, num_heads=8,  num_layers=12, ffn_dim=2048),
    "large": dict(hidden_dim=1024, num_heads=16, num_layers=24, ffn_dim=4096),
}


@dataclass
class DJEPAConfig:
    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    seed: int = 42

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    model_size: str = "small"           # key into MODEL_SIZES
    hidden_dim: int = 256               # encoder / world-model width
    num_heads: int = 8                  # MHA heads
    num_layers: int = 6                 # transformer depth
    ffn_dim: int = 1024                 # feed-forward expansion

    # Disentanglement: z_speaker and z_language each get half the hidden_dim
    # so their concatenation reconstructs the full latent without information
    # overlap — required by the Mutual Information Bottleneck objective.
    speaker_dim: int = 128              # dim(z_speaker)  = hidden_dim // 2
    language_dim: int = 128             # dim(z_language) = hidden_dim // 2

    # Input / output
    input_acoustic_dim: int = 80        # log-Mel filterbank bins (standard ASR)
    # 256 for A100 training (40 GB HBM handles B×256×100277 easily).
    # Was 128 on MPS (22 GB unified memory limit).
    max_seq_len: int = 256
    # Real tiktoken cl100k_base vocabulary (covers English + Devanagari Hindi).
    vocab_size: int = 100277

    # ------------------------------------------------------------------
    # Ablation toggles  — set to False to remove a component entirely
    # ------------------------------------------------------------------
    use_ewc: bool = True                # Elastic Weight Consolidation
    use_decorr_loss: bool = True        # Cosine Decorrelation penalty
    use_mi_bottleneck: bool = True      # Mutual Information Bottleneck
    use_sgg: bool = True                # Speculative Graph Grounding gate
    use_world_model: bool = True        # CM-JEPA latent world model
    use_amp: bool = True                # Automatic Mixed Precision

    # ------------------------------------------------------------------
    # Loss weights  (λ coefficients in L_total)
    # ------------------------------------------------------------------
    lambda_decorr: float = 0.05         # λ₁: decorrelation penalty weight
    lambda_ewc: float = 400.0           # λ₂: EWC regularisation strength
    lambda_jepa: float = 0.1            # λ₃: JEPA predictive loss weight
    lambda_mi: float = 0.01             # weight on MI bottleneck loss
    lambda_kg: float = 0.05             # λ₄: KG grounding loss (L_KG)

    # ------------------------------------------------------------------
    # Speculative Graph Grounding (SGG)
    # ------------------------------------------------------------------
    sgg_threshold: float = 2.5          # L2 distance threshold for Tier 1 rejection
    sgg_kg_embed_dim: int = 128         # dimension of PrimeKG node embeddings
    sgg_num_kg_nodes: int = 64          # KG nodes (overridden by real KG at runtime)

    # Cold-Start / OOD Fallback (Three-Tier Mechanism)
    # Tier 1: if no patient graph, gate against global ontology instead
    # Tier 2: uncertainty scalar γ ∈ [0,1] — lowers rejection strictness
    #         when patient history is absent (γ=1.0 = full strictness)
    sgg_cold_start_gamma: float = 0.4   # γ when patient graph is null
    # Tier 3: episodic memory initialisation strategy
    #   "healthy_baseline" — mean of non-disease KG node embeddings (recommended)
    #   "zeros"            — cold zero vector (biologically uninformative)
    sgg_episodic_init: str = "healthy_baseline"
    # Number of frames after which episodic memory is considered "populated"
    # (i.e., gate reverts to full γ=1.0 strictness after this many steps)
    sgg_warmup_frames: int = 50

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    # A100 (40 GB HBM): batch=64 fits comfortably with seq=256, vocab=100277.
    # MPS fallback: set batch_size=8, max_seq_len=128 via env or override.
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    warmup_steps: int = 500
    log_interval: int = 10
    # Gradient accumulation: effective batch = batch_size × grad_accum_steps.
    # Set >1 to simulate larger batch without extra memory.
    grad_accum_steps: int = 1
    # torch.compile the model for ~30% speedup on A100 (PyTorch 2.x).
    # Disable on MPS/CPU (not supported or slower).
    compile_model: bool = True

    # ------------------------------------------------------------------
    # EWC specifics
    # ------------------------------------------------------------------
    ewc_fisher_samples: int = 256       # samples used to estimate Fisher matrix
    ewc_dataset: str = "medical"        # which task's weights to protect

    # ------------------------------------------------------------------
    # Data / DataLoader
    # ------------------------------------------------------------------
    # GPUPreloadedLoader keeps all data GPU-resident so workers and pin_memory
    # are irrelevant on CUDA. Keep 0/False; only the CPU/MPS fallback uses them.
    num_workers: int = 0
    pin_memory: bool = False
    num_speakers: int = 3               # speakers in mock dataset
    num_languages: int = 2              # Hindi (0) + English (1)
    code_switch_prob: float = 0.4       # probability of a code-switch at each frame

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # ------------------------------------------------------------------
    # Post-init: sync size-preset values
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.model_size in MODEL_SIZES:
            preset = MODEL_SIZES[self.model_size]
            # Only override if the user left values at their defaults
            self.hidden_dim  = preset["hidden_dim"]
            self.num_heads   = preset["num_heads"]
            self.num_layers  = preset["num_layers"]
            self.ffn_dim     = preset["ffn_dim"]
            # Keep speaker/language dims consistent with hidden_dim
            self.speaker_dim  = self.hidden_dim // 2
            self.language_dim = self.hidden_dim // 2


def get_config(size: str = "small", **overrides) -> DJEPAConfig:
    """Factory for DJEPAConfig with optional keyword overrides."""
    cfg = DJEPAConfig(model_size=size)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown config field: {k!r}")
        object.__setattr__(cfg, k, v)
    return cfg
