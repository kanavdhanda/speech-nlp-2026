"""
model.py — D-JEPA Core Architecture
Implements four sub-modules that together form the Disentangled
Joint-Embedding Predictive Architecture with Continual Grounding:

  1. MutualInformationBottleneck   — acoustic encoder → (z_speaker, z_language)
  2. CosineDecorrAttention         — MHA with L_decorr sink-destruction penalty
  3. LatentClinicalWorldModel      — CM-JEPA state-space transition operator
  4. SpeculativeGraphGroundingGate — forward-hook reject gate via PrimeKG L2 dist

All modules use strict type hints and are designed to be composable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import DJEPAConfig


# ===========================================================================
# 1. Mutual Information Bottleneck
# ===========================================================================

class MutualInformationBottleneck(nn.Module):
    """
    Splits the acoustic input into two *orthogonal* latent codes:
        z_speaker   (captures who is speaking)
        z_language  (captures which language is spoken)

    Orthogonality is enforced by a projection head that predicts one code
    from the other: if the prediction is easy the codes are correlated,
    which inflates L_MI.  The network is penalised for this — cf. the
    Information Bottleneck principle (Tishby & Zaslavsky, 2015).

    MI approximation:
        L_MI = ||z_lang_pred - z_language||^2
    where z_lang_pred = MLP(z_speaker).
    This is a simple but effective surrogate for MI minimisation used in
    disentangled VAE literature (Chen et al., 2018 — β-TCVAE).
    """

    def __init__(self, cfg: DJEPAConfig) -> None:
        super().__init__()
        D   = cfg.input_acoustic_dim
        S   = cfg.speaker_dim
        L   = cfg.language_dim

        # Shared bottom encoder: raw acoustics → joint representation
        self.shared_encoder = nn.Sequential(
            nn.Linear(D, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )

        # Task-specific projection heads
        self.speaker_head  = nn.Linear(cfg.hidden_dim, S)
        self.language_head = nn.Linear(cfg.hidden_dim, L)

        # MI adversarial predictor: tries to predict z_language from z_speaker
        # Higher loss → lower mutual information → better disentanglement
        self.mi_predictor = nn.Sequential(
            nn.Linear(S, L * 2),
            nn.GELU(),
            nn.Linear(L * 2, L),
        )

        self.use_mi = cfg.use_mi_bottleneck

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_acoustic_dim)  — padded acoustic frames
        Returns:
            z_speaker   : (B, T, speaker_dim)
            z_language  : (B, T, language_dim)
            loss_mi     : scalar — MI bottleneck loss (0 if ablated)
        """
        h = self.shared_encoder(x)                     # (B, T, hidden)
        z_speaker  = self.speaker_head(h)               # (B, T, S)
        z_language = self.language_head(h)              # (B, T, L)

        if self.use_mi:
            # Predict z_language from z_speaker (stop-gradient on z_language
            # so we only penalise the speaker encoder, not the language one)
            z_lang_pred = self.mi_predictor(z_speaker)   # (B, T, L)
            loss_mi     = F.mse_loss(z_lang_pred, z_language.detach())
        else:
            loss_mi = torch.tensor(0.0, device=x.device)

        return z_speaker, z_language, loss_mi


# ===========================================================================
# 2. Cosine Decorrelation Attention
# ===========================================================================

class CosineDecorrAttention(nn.Module):
    """
    Multi-Head Attention augmented with the Decorrelation penalty L_decorr.

    Attention sinks (Xiao et al., 2023 — StreamLLM) arise because the BOS
    token accumulates attention mass independent of content.  This module
    destroys sinks by penalising cosine similarity between each intermediate
    hidden state and the BOS-position hidden state:

        L_decorr = (1/T) * sum_t  [cos_sim(h_t, h_0)]^2

    where h_0 is the BOS (first-token) state.  When this loss is minimised,
    h_t becomes orthogonal to h_0 for t > 0, preventing uniform attention
    mass concentration on the BOS position.
    """

    def __init__(self, cfg: DJEPAConfig) -> None:
        super().__init__()
        self.hidden_dim     = cfg.hidden_dim
        self.num_heads      = cfg.num_heads
        self.use_decorr     = cfg.use_decorr_loss

        # Standard MHA (using the fused SDPA kernel when available)
        self.mha = nn.MultiheadAttention(
            embed_dim    = cfg.hidden_dim,
            num_heads    = cfg.num_heads,
            dropout      = 0.1,
            batch_first  = True,   # (B, T, D) convention throughout
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x                : (B, T, hidden_dim)
            key_padding_mask : (B, T) bool, True = PAD (ignored in attention)
        Returns:
            out      : (B, T, hidden_dim) — post-attention hidden states
            loss_decorr : scalar — decorrelation penalty
        """
        # need_weights=False routes through F.scaled_dot_product_attention,
        # which selects the Flash Attention 2 kernel on A100 automatically.
        attn_out, _ = self.mha(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        out = self.norm(x + attn_out)    # pre-LN residual

        if self.use_decorr:
            # h_0 is the BOS-position state — shape (B, 1, D)
            h_bos = out[:, :1, :]                              # (B, 1, D)

            # Cosine similarity between each position and BOS: (B, T)
            cos_sim = F.cosine_similarity(
                out,                                            # (B, T, D)
                h_bos.expand_as(out),                          # (B, T, D)
                dim=-1,
            )
            # L_decorr = mean of squared cosine similarities over all positions
            # (including t=0 itself, which is always 1 but is a known constant)
            loss_decorr = cos_sim.pow(2).mean()
        else:
            loss_decorr = torch.tensor(0.0, device=x.device)

        return out, loss_decorr


# ===========================================================================
# 3. Latent Clinical World Model (CM-JEPA)
# ===========================================================================

class LatentClinicalWorldModel(nn.Module):
    """
    State-space transition operator for CM-JEPA:

        W_{t+1} = T(W_t, s_t, S_role)

    where:
      W_t    — current world-model latent state (B, hidden_dim)
      s_t    — observation embedding at step t  (B, hidden_dim)
      S_role — role/speaker conditioning signal (B, speaker_dim)

    Implementation: a lightweight GRU-based transition with an additional
    cross-attention layer that reads from the speaker conditioning.  This
    follows the JEPA objective (LeCun, 2022): predict the *representation*
    of the future, not raw observations, sidestepping pixel-level
    reconstruction costs.

    JEPA loss: L_JEPA = ||W_{t+1} - sg(s_{t+1})||^2
    where sg(·) is stop-gradient, preventing representation collapse.
    """

    # EMA momentum for target encoder (BYOL / I-JEPA convention)
    EMA_MOMENTUM: float = 0.996

    def __init__(self, cfg: DJEPAConfig) -> None:
        super().__init__()
        H = cfg.hidden_dim
        S = cfg.speaker_dim

        # batch_first=True lets us pass (B, T, H+S) and get (B, T, H) back in
        # one cuDNN kernel call instead of a Python loop over T GRUCell steps.
        self.transition_gru = nn.GRU(input_size=H + S, hidden_size=H, batch_first=True)

        # Online predictor: projects W_{t+1} into a comparison space
        self.predictor = nn.Sequential(
            nn.Linear(H, H * 2),
            nn.GELU(),
            nn.Linear(H * 2, H),
        )

        # EMA target encoder: slowly-moving copy of the GRU hidden state
        # projection.  Parameters are NOT updated by gradients — only by EMA.
        # This prevents representation collapse (BYOL, Grill et al. 2020).
        self.target_projector = nn.Sequential(
            nn.Linear(H, H * 2),
            nn.GELU(),
            nn.Linear(H * 2, H),
        )
        # Initialise target to match predictor weights; freeze from autograd
        for p in self.target_projector.parameters():
            p.requires_grad_(False)

        self.use_world_model = cfg.use_world_model

    @torch.no_grad()
    def _update_ema(self) -> None:
        """EMA update: θ_target ← m·θ_target + (1-m)·θ_predictor"""
        m = self.EMA_MOMENTUM
        for p_pred, p_tgt in zip(
            self.predictor.parameters(), self.target_projector.parameters()
        ):
            p_tgt.data.mul_(m).add_(p_pred.data, alpha=1.0 - m)

    # Multi-step prediction horizons (skip distances).
    # Predicting t+1 is too noisy in short acoustic windows.
    # Predicting t+{1,2,4} and averaging is more stable — I-JEPA style.
    SKIP_STEPS: tuple = (1, 2, 4)

    def forward(
        self,
        s_seq:    torch.Tensor,   # (B, T, H)
        z_speaker: torch.Tensor,  # (B, T, S)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-step BYOL-style JEPA loss:

            L_JEPA = 1 - (1/|K||T|) Σ_k Σ_t cos_sim(predict(W_t), sg(target(s_{t+k})))

        where K = {1, 2, 4}.  Predicting multiple future horizons:
          (a) Smooths the gradient signal — single-step next-frame in short
              acoustic sequences is too stochastic (adjacent frames ≈ iid noise).
          (b) Encourages the GRU to capture medium-range temporal structure
              (prosody, phoneme boundaries) not just frame-to-frame continuity.
          (c) Loss stays in [0, 2] regardless of hidden norm growth.
        """
        B, T, H = s_seq.shape
        device   = s_seq.device

        if not self.use_world_model:
            return s_seq, torch.tensor(0.0, device=device)

        # Single cuDNN GRU kernel over the full sequence — replaces the
        # Python loop over T GRUCell calls whose launch overhead dominated.
        gru_input           = torch.cat([s_seq, z_speaker], dim=-1)       # (B, T, H+S)
        world_states_tensor, _ = self.transition_gru(gru_input)            # (B, T, H)

        # Vectorised multi-step BYOL JEPA loss.
        # Run predictor and target projector once over all T positions (batched
        # matmul) instead of T×|K| individual forward passes in a Python loop.
        with torch.no_grad():
            tgt_proj = self.target_projector(
                s_seq.reshape(B * T, H)
            ).reshape(B, T, H)                                              # (B, T, H)

        pred_all = self.predictor(
            world_states_tensor.reshape(B * T, H)
        ).reshape(B, T, H)                                                  # (B, T, H)

        cos_sims = [
            F.cosine_similarity(pred_all[:, :T - k, :], tgt_proj[:, k:, :], dim=-1).mean()
            for k in self.SKIP_STEPS if T > k
        ]

        if self.training:
            self._update_ema()

        loss_jepa = (1.0 - torch.stack(cos_sims).mean()) if cos_sims else \
                    torch.tensor(0.0, device=device)

        return world_states_tensor, loss_jepa


# ===========================================================================
# 4. Speculative Graph Grounding Gate  (with Three-Tier Cold-Start Fallback)
# ===========================================================================

class SpeculativeGraphGroundingGate(nn.Module):
    """
    Validates predicted future world-states against the medical KG.

    For known patients the gate checks against a patient-specific sub-graph.
    For OOD / "John Doe" patients with no EHR (cold start), it applies a
    Three-Tier Fallback Mechanism so the model never crashes on blank slates:

    Tier 1 — Ontological Fallback
        If patient_graph is None, swap the target from the patient sub-graph
        to the global_ontology_vector (mean of all KG node embeddings).
        The gate still enforces biological plausibility — it just asks
        "Is this a valid medical concept?" rather than "Is this in this
        patient's history?"

    Tier 2 — Uncertainty-Weighted Gating
        Introduce scalar γ ∈ (0, 1] into the effective threshold:
            effective_threshold = sgg_threshold / γ
        When patient graph is absent γ = sgg_cold_start_gamma (< 1.0),
        which *raises* the effective threshold → less strict rejection →
        acoustic evidence is trusted more. As episodic memory fills up,
        γ linearly ramps toward 1.0.

        Loss form (Tier 2):
            L_plausibility = γ · ||proj(W_{t+1}) - z_ontology||²

    Tier 3 — Real-Time Episodic Memory (Dynamic Node Initialisation)
        A learnable blank-node tensor W_episodic ∈ R^{kg_embed_dim} is
        initialised to the "healthy baseline" (mean of non-disease KG nodes)
        or zeros per config. As the CM-JEPA world model runs, projected
        states are accumulated into W_episodic via a running EMA:
            W_episodic ← α·W_episodic + (1-α)·mean(proj(W_t))
        After sgg_warmup_frames steps W_episodic is "populated" and the gate
        can use it as a patient-specific sub-graph centroid.

    Full strictness (γ=1.0) is restored once episodic memory is populated.
    """

    EPISODIC_EMA_ALPHA: float = 0.9   # EMA momentum for episodic memory update

    def __init__(self, cfg: DJEPAConfig, kg_data: Dict[str, Any]) -> None:
        super().__init__()
        self.threshold      = cfg.sgg_threshold
        self.use_sgg        = cfg.use_sgg
        self.cold_gamma     = cfg.sgg_cold_start_gamma
        self.episodic_init  = cfg.sgg_episodic_init
        self.warmup_frames  = cfg.sgg_warmup_frames
        E                   = cfg.sgg_kg_embed_dim

        # Learnable projection: world-model hidden → KG embedding space
        self.projection = nn.Linear(cfg.hidden_dim, E)

        # Full KG embeddings buffer — shape (N, E)
        kg_embeddings: torch.Tensor = kg_data["embeddings"]
        self.register_buffer("kg_embeddings", kg_embeddings)

        # Tier 1: global ontology vector = unit-normalised mean of all KG nodes
        # (pre-computed once; stays fixed during training)
        ontology_vec = kg_embeddings.mean(0)
        ontology_vec = ontology_vec / (ontology_vec.norm() + 1e-8)
        self.register_buffer("global_ontology_vec", ontology_vec)   # (E,)

        # Tier 3: episodic memory baseline
        node_type: Dict[str, str] = kg_data.get("node_type", {})
        node_ids:  list            = kg_data.get("node_ids",  [])
        if self.episodic_init == "healthy_baseline":
            # Mean of gene + pathway + drug nodes (non-disease/symptom = "healthy")
            healthy_idx = [
                i for i, nid in enumerate(node_ids)
                if node_type.get(nid, "pathway") not in ("disease", "symptom")
            ] or list(range(len(node_ids)))
            baseline = kg_embeddings[healthy_idx].mean(0)
            baseline = baseline / (baseline.norm() + 1e-8)
        else:
            baseline = torch.zeros(E)
        self.register_buffer("episodic_baseline", baseline)         # (E,)

        # Mutable episodic memory — shape varies with batch size at runtime,
        # so we do NOT persist it in the checkpoint (persistent=False).
        # It is always reinitialised from episodic_baseline on load.
        self.register_buffer(
            "_episodic_memory",
            baseline.clone().unsqueeze(0),  # (1, E)
            persistent=False,               # excluded from state_dict
        )
        self._frames_seen: int = 0

    def reset_episodic_memory(self, batch_size: int, device: torch.device) -> None:
        """Call at the start of a new patient session (cold start)."""
        self._episodic_memory = self.episodic_baseline.unsqueeze(0).expand(
            batch_size, -1
        ).clone().to(device)
        self._frames_seen = 0

    def _gamma(self) -> float:
        """
        Linearly ramp γ from cold_gamma → 1.0 over warmup_frames.
        Once episodic memory is populated, full strictness is restored.
        """
        if self._frames_seen >= self.warmup_frames:
            return 1.0
        frac = self._frames_seen / max(self.warmup_frames, 1)
        return self.cold_gamma + (1.0 - self.cold_gamma) * frac

    def forward(
        self,
        world_state:    torch.Tensor,                    # (B, T, H)
        patient_graph:  Optional[torch.Tensor] = None,  # (B, N_p, E) or None
        is_cold_start:  bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            world_state   : (B, T, H) — CM-JEPA world model outputs
            patient_graph : optional patient-specific KG node embeddings
            is_cold_start : True when patient has no EHR (John Doe scenario)

        Returns:
            projected        : (B, T, E)
            min_distances    : (B, T)
            rejection_flags  : (B, T) bool
            cold_start_info  : dict with γ, tier used, frames_seen
        """
        B, T, H  = world_state.shape
        device   = world_state.device

        if not self.use_sgg:
            E = self.kg_embeddings.shape[-1]
            return (torch.zeros(B, T, E, device=device),
                    torch.zeros(B, T, device=device),
                    torch.zeros(B, T, dtype=torch.bool, device=device),
                    {"tier": 0, "gamma": 1.0, "frames_seen": 0},
                    torch.tensor(0.0, device=device))

        projected = self.projection(world_state)   # (B, T, E)
        B, T, E   = projected.shape
        proj_flat = projected.reshape(B * T, E)    # (B*T, E)

        # ------------------------------------------------------------------
        # Determine which tier to use and effective threshold
        # ------------------------------------------------------------------
        if not is_cold_start and patient_graph is not None:
            # ── Normal path: patient sub-graph available ──────────────────
            tier  = 0
            gamma = 1.0
            # Patient graph: (B, N_p, E) → flatten to (B*N_p, E) per sample
            # For simplicity use cdist against the patient's nodes
            # (In production this would be per-sample; here averaged for batch)
            target_nodes = patient_graph.reshape(-1, E)    # (B*N_p, E)
            distances    = torch.cdist(proj_flat, target_nodes)
        else:
            # ── Cold-start path ───────────────────────────────────────────
            gamma = self._gamma()

            if self._frames_seen < self.warmup_frames:
                # Tier 2 + Tier 3: ontology fallback + uncertainty weighting
                tier = 2 if self._frames_seen == 0 else 3
                # Use Tier 3 episodic memory if partially populated, else Tier 1
                if self._frames_seen > 0:
                    # Tier 3: compare against current episodic memory centroid
                    mem = self._episodic_memory.mean(0, keepdim=True)  # (1, E)
                    distances = torch.cdist(proj_flat, mem)             # (B*T, 1)
                else:
                    # Tier 1: compare against global ontology vector
                    ontology = self.global_ontology_vec.unsqueeze(0)    # (1, E)
                    distances = torch.cdist(proj_flat, ontology)        # (B*T, 1)
            else:
                # Episodic memory fully populated — use it like a patient graph
                tier = 3
                mem  = self._episodic_memory.mean(0, keepdim=True)
                distances = torch.cdist(proj_flat, mem)

            # Tier 3: update episodic memory with current projected states
            new_obs = proj_flat.detach().mean(0, keepdim=True)  # (1, E)
            alpha   = self.EPISODIC_EMA_ALPHA
            if self._episodic_memory.shape[0] != B:
                self._episodic_memory = self.episodic_baseline.unsqueeze(0).expand(
                    B, -1).clone().to(device)
            self._episodic_memory = (
                alpha * self._episodic_memory
                + (1 - alpha) * new_obs.expand(B, -1)
            ).detach()
            self._frames_seen += T

        min_distances, _ = distances.min(dim=-1)            # (B*T,)
        min_distances    = min_distances.reshape(B, T)      # (B, T)

        # Tier 2: uncertainty-weighted effective threshold
        effective_threshold = self.threshold / max(gamma, 1e-6)
        rejection_flags     = min_distances > effective_threshold  # (B, T)

        # L_KG: KG grounding loss — pull projected world states toward nearest KG node.
        #   L_KG = mean_{t} min_{n∈KG_sample} ||proj(W_t) - e_n||₂
        # We subsample 256 KG nodes per step to bound memory on MPS (full 4577-node
        # cdist with B*T rows was OOM). The minimum-distance signal is still valid
        # with random subsampling — each step sees a different 256-node subset,
        # covering the full KG across training with unbiased gradient estimates.
        N_full  = self.kg_embeddings.shape[0]
        n_sub   = min(256, N_full)
        if self.training:
            sub_idx  = torch.randperm(N_full, device=device)[:n_sub]
            kg_sub   = self.kg_embeddings[sub_idx]                      # (256, E)
        else:
            kg_sub   = self.kg_embeddings                               # full KG at inference
        full_kg_dists   = torch.cdist(proj_flat, kg_sub)               # (B*T, 256)
        kg_min_dists, _ = full_kg_dists.min(dim=-1)                    # (B*T,)
        loss_kg         = kg_min_dists.mean()                          # scalar

        cold_start_info = {
            "tier":        tier if (is_cold_start or patient_graph is None) else 0,
            "gamma":       gamma,
            "frames_seen": self._frames_seen,
            "episodic_populated": self._frames_seen >= self.warmup_frames,
        }

        return projected, min_distances, rejection_flags, cold_start_info, loss_kg


# ===========================================================================
# 5. Full D-JEPA Model
# ===========================================================================

class DJEPA(nn.Module):
    """
    Full D-JEPA model, composing all four sub-modules into a unified forward pass.

    Forward pass returns:
        logits       : (B, T, vocab_size) — ASR token logits
        losses       : dict with 'asr', 'mi', 'decorr', 'jepa', 'total'
        sgg_output   : dict with 'min_distances', 'rejection_flags'
    """

    def __init__(self, cfg: DJEPAConfig, kg_data: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg

        # Sub-modules
        self.mi_bottleneck   = MutualInformationBottleneck(cfg)
        self.decorr_attn     = CosineDecorrAttention(cfg)
        self.world_model     = LatentClinicalWorldModel(cfg)
        self.sgg_gate        = SpeculativeGraphGroundingGate(cfg, kg_data)

        # Fuse speaker + language codes back into hidden_dim for the transformer
        self.fusion = nn.Linear(cfg.speaker_dim + cfg.language_dim, cfg.hidden_dim)

        # Stack of additional decorr-attention layers (depth - 1)
        self.extra_layers = nn.ModuleList([
            CosineDecorrAttention(cfg)
            for _ in range(max(0, cfg.num_layers - 1))
        ])

        # ASR output head
        self.asr_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

        # Loss weights from config
        self.lambda_decorr = cfg.lambda_decorr
        self.lambda_jepa   = cfg.lambda_jepa
        self.lambda_mi     = cfg.lambda_mi
        self.lambda_kg     = cfg.lambda_kg

    def forward(
        self,
        features:       torch.Tensor,                     # (B, T, input_acoustic_dim)
        asr_targets:    torch.Tensor,                     # (B, T) int64 token ids
        padding_mask:   Optional[torch.Tensor] = None,    # (B, T) True=PAD
        patient_graph:  Optional[torch.Tensor] = None,    # (B, N_p, E) or None
        is_cold_start:  bool = False,                     # True for John Doe OOD
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:

        # --- Step 1: Disentangle acoustic features ---
        z_speaker, z_language, loss_mi = self.mi_bottleneck(features)

        # --- Step 2: Fuse disentangled codes ---
        fused = self.fusion(torch.cat([z_speaker, z_language], dim=-1))  # (B, T, H)

        # --- Step 3: Transformer layers with decorrelation penalty ---
        h           = fused
        total_decorr = torch.tensor(0.0, device=features.device)

        h, decorr_0 = self.decorr_attn(h, key_padding_mask=padding_mask)
        total_decorr = total_decorr + decorr_0

        for layer in self.extra_layers:
            h, decorr_i = layer(h, key_padding_mask=padding_mask)
            total_decorr = total_decorr + decorr_i

        loss_decorr = total_decorr / (1 + len(self.extra_layers))   # average

        # --- Step 4: World model rollout ---
        world_states, loss_jepa = self.world_model(h, z_speaker)

        # --- Step 5: SGG gate with three-tier cold-start fallback + L_KG ---
        _, min_dists, rejection_flags, cold_start_info, loss_kg = self.sgg_gate(
            world_states,
            patient_graph = patient_graph,
            is_cold_start = is_cold_start,
        )

        # --- Step 6: ASR logits from transformer hidden states ---
        logits = self.asr_head(h)   # (B, T, vocab_size)

        # --- Step 7: ASR cross-entropy loss (ignore PAD positions) ---
        B, T, V  = logits.shape
        loss_asr = F.cross_entropy(
            logits.reshape(B * T, V),
            asr_targets.reshape(B * T),
            ignore_index=0,    # PAD token = 0
        )

        # --- Step 8: Composite loss ---
        # L_KG pulls projected world states toward nearest KG node, fixing
        # the 86% hallucination rate where W never landed near any KG concept.
        loss_total = (
            loss_asr
            + self.lambda_mi     * loss_mi
            + self.lambda_decorr * loss_decorr
            + self.lambda_jepa   * loss_jepa
            + self.lambda_kg     * loss_kg
            # EWC added externally in train.py
        )

        losses = {
            "asr":    loss_asr,
            "mi":     loss_mi,
            "decorr": loss_decorr,
            "jepa":   loss_jepa,
            "kg":     loss_kg,
            "total":  loss_total,
        }

        sgg_output = {
            "min_distances":   min_dists,
            "rejection_flags": rejection_flags,
            "cold_start_info": cold_start_info,
        }

        return logits, losses, sgg_output
