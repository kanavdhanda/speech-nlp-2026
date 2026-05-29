"""
train.py — D-JEPA ICLR-Grade Training Loop
Implements:
  - Two-phase continual learning with isolated val-loss tracking per phase
  - Elastic Weight Consolidation (EWC) with diagonal Fisher estimation
  - Multi-objective loss: L_total = L_ASR + λ1·L_decorr + λ2·L_EWC + λ3·L_JEPA
  - Best-checkpoint saving per phase (Phase 1 and Phase 2 never share a best-val)
  - PyTorch AMP GradScaler for CUDA; plain backward on MPS/CPU
  - Deterministic seeding and gradient clipping
  - Loss history saved to JSON + matplotlib PNG

EWC: Kirkpatrick et al., 2017. Fisher diagonal:
    F_i ≈ (1/N) Σ_n (∂ L_ASR / ∂ θ_i)²
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model.config import DJEPAConfig
from model.data import build_dataloaders, load_real_kg
from model.data_generation import PrimeKGMockGenerator, set_global_seed
from model.device import get_device
from model.model import DJEPA


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(log_dir: str, name: str = "djepa") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def save_checkpoint(
    path:        str,
    epoch:       int,
    phase:       int,
    model:       DJEPA,
    optimizer:   AdamW,
    scheduler:   CosineAnnealingLR,
    scaler:      GradScaler,
    val_loss:    float,
    cfg:         DJEPAConfig,
    fisher_dict: Optional[Dict[str, torch.Tensor]] = None,
    old_params:  Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Atomic checkpoint: writes to .tmp then renames so crashes never corrupt."""
    payload = {
        "epoch":           epoch,
        "phase":           phase,
        "val_loss":        val_loss,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "cfg":             dataclasses.asdict(cfg),
        "fisher_dict":     {k: v.cpu() for k, v in fisher_dict.items()} if fisher_dict else None,
        "old_params":      {k: v.cpu() for k, v in old_params.items()}  if old_params  else None,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ===========================================================================
# EWC
# ===========================================================================

def compute_fisher_information(
    model:       DJEPA,
    dataloader:  DataLoader,
    device:      torch.device,
    num_samples: int = 256,
) -> Dict[str, torch.Tensor]:
    """Diagonal FIM: F_i ≈ (1/N) Σ (∂ L_ASR / ∂ θ_i)²  (Kirkpatrick 2017 eq.3)"""
    model.eval()
    fisher: Dict[str, torch.Tensor] = {
        n: torch.zeros_like(p.data)
        for n, p in model.named_parameters() if p.requires_grad
    }
    n_seen = 0
    for batch in dataloader:
        if n_seen >= num_samples:
            break
        features     = batch["features"].to(device)
        asr_targets  = batch["asr_tokens"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        model.zero_grad()
        torch.compiler.cudagraph_mark_step_begin()
        _, losses, _ = model(features, asr_targets, padding_mask)
        losses["asr"].backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data.pow(2)
        n_seen += features.shape[0]
    for n in fisher:
        fisher[n] /= max(n_seen, 1)
    model.train()
    return fisher


def compute_ewc_loss(
    model:      DJEPA,
    fisher:     Dict[str, torch.Tensor],
    old_params: Dict[str, torch.Tensor],
    ewc_lambda: float,
) -> Tuple[torch.Tensor, float]:
    """
    L_EWC = (λ/2) Σ_i F_i · (θ_i − θ*_i)²
    Returns (scaled_loss, raw_quadratic_sum) for transparent logging.
    """
    device = next(model.parameters()).device
    raw    = torch.tensor(0.0, device=device)
    for n, p in model.named_parameters():
        if n in fisher and n in old_params:
            raw = raw + (fisher[n] * (p - old_params[n]).pow(2)).sum()
    return (ewc_lambda / 2.0) * raw, raw.item()


# ===========================================================================
# Backward / validation helpers
# ===========================================================================

def _step(loss: torch.Tensor, model: DJEPA, optimizer: AdamW,
          scaler: GradScaler, grad_clip: float, use_scaler: bool) -> None:
    if use_scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


def _validate(model: DJEPA, loader: DataLoader, device: torch.device,
              cfg: DJEPAConfig, use_amp: bool,
              amp_dtype: torch.dtype = torch.float16) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            features     = batch["features"].to(device)
            asr_targets  = batch["asr_tokens"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            torch.compiler.cudagraph_mark_step_begin()
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                _, losses, _ = model(features, asr_targets, padding_mask)
            total += losses["total"].item()
    model.train()
    return total / max(len(loader), 1)


# ===========================================================================
# Loss curve plotting
# ===========================================================================

def plot_loss_curves(history: Dict[str, List], log_dir: str) -> None:
    """
    Save separate Phase 1 and Phase 2 loss curve figures.
    Phase 1 and Phase 2 are plotted independently with their own y-axes
    so that EWC penalty scale in P2 doesn't compress P1 curves.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pb   = history.get("phase_boundary", len(history["train_total"]) // 2)
        keys = ["train_asr", "train_decorr", "train_jepa", "train_mi",
                "train_kg", "train_ewc", "train_total", "val_total"]

        p1 = {k: history[k][:pb]  for k in keys if k in history}
        p2 = {k: history[k][pb:]  for k in keys if k in history}

        def _phase_plot(data: Dict, phase: int, out_path: str) -> None:
            n   = len(next(iter(data.values()))) if data else 0
            if n == 0:
                return
            ep  = list(range(1, n + 1))
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            ewc_suffix = " (EWC active)" if phase == 2 else " (pre-Fisher)"
            fig.suptitle(f"D-JEPA Phase {phase} Loss Curves{ewc_suffix}",
                         fontsize=13, fontweight="bold")
            pairs = [
                ("train_total",  "val_total",  "Total Loss",    axes[0, 0]),
                ("train_asr",    None,         "ASR Loss",      axes[0, 1]),
                ("train_decorr", None,         "Decorr Loss",   axes[0, 2]),
                ("train_jepa",   None,         "JEPA Loss",     axes[1, 0]),
                ("train_mi",     None,         "MI Loss",       axes[1, 1]),
                ("train_ewc",    None,         "EWC Loss",      axes[1, 2]),
            ]
            for tk, vk, title, ax in pairs:
                if tk in data:
                    ax.plot(ep, data[tk], label="train",
                            color="steelblue", linewidth=2)
                if vk and vk in data:
                    ax.plot(ep, data[vk], label="val",
                            color="darkorange", linewidth=2, linestyle="--")
                ax.set_title(title, fontsize=10)
                ax.set_xlabel(f"Phase {phase} Epoch")
                ax.set_ylabel("Loss")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
                # Annotate best val point
                if vk and vk in data and data[vk]:
                    best_ep = int(min(range(len(data[vk])), key=lambda i: data[vk][i]))
                    ax.axvline(x=best_ep + 1, color="green", linestyle=":",
                               linewidth=1, label=f"best val ep{best_ep+1}")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[train] Phase {phase} curves → {out_path}")

        _phase_plot(p1, 1, os.path.join(log_dir, "loss_curves_phase1.png"))
        _phase_plot(p2, 2, os.path.join(log_dir, "loss_curves_phase2.png"))

        # Also save a combined summary figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("D-JEPA: Phase 1 vs Phase 2 Total Loss", fontsize=12, fontweight="bold")
        if "train_total" in p1:
            ax1.plot(range(1, len(p1["train_total"])+1), p1["train_total"],
                     color="steelblue", label="train", linewidth=2)
        if "val_total" in p1:
            ax1.plot(range(1, len(p1["val_total"])+1), p1["val_total"],
                     color="darkorange", label="val", linewidth=2, linestyle="--")
        ax1.set_title("Phase 1 (Base Task)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Total Loss")
        ax1.legend()
        ax1.grid(alpha=0.3)

        if "train_total" in p2:
            ax2.plot(range(1, len(p2["train_total"])+1), p2["train_total"],
                     color="steelblue", label="train (incl. EWC)", linewidth=2)
        if "val_total" in p2:
            ax2.plot(range(1, len(p2["val_total"])+1), p2["val_total"],
                     color="darkorange", label="val", linewidth=2, linestyle="--")
        ax2.set_title("Phase 2 (Continual + EWC)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Total Loss")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        combined = os.path.join(log_dir, "loss_curves.png")
        plt.savefig(combined, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[train] Combined summary → {combined}")

    except ImportError:
        print("[train] matplotlib not installed — skipping loss curve plot")


# ===========================================================================
# Training loop
# ===========================================================================

def train(cfg: DJEPAConfig) -> DJEPA:
    """
    Two-phase continual-learning training.

    Phase 1 — base (medical/ASR) task.  Best val checkpoint saved independently.
    Phase 2 — code-switched task + EWC.  Its own best val tracked from epoch 1
              of Phase 2, so Phase 1 overfitting never taints Phase 2's metric.
              Phase 2 keeps the same LR as Phase 1 (no bump) to avoid rapid
              parameter drift that makes EWC penalty blow up.

    Checkpoints:
      checkpoints/phase1_best.pt  — best Phase 1 model
      checkpoints/phase2_best.pt  — best Phase 2 model (EWC-regularised)
    """
    set_global_seed(cfg.seed)
    device  = get_device()
    use_amp = cfg.use_amp and device.type == "cuda"
    # A100 has native BF16 tensor cores; avoids GradScaler overhead and
    # numerical underflow issues that affect FP16 for large vocab cross-entropy.
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
    logger  = _setup_logger(cfg.log_dir)

    # A100-specific global optimisations ─────────────────────────────────────
    if device.type == "cuda":
        # TF32 for matmuls — 19× faster than FP32 on A100, ~same accuracy.
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # cuDNN auto-tunes conv kernels for the fixed input size.
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA optimisations: TF32 + cudnn.benchmark ON | AMP dtype: %s", amp_dtype)
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("Device: %s | Model: %s | AMP: %s | accum_steps: %d",
                device, cfg.model_size, use_amp, cfg.grad_accum_steps)

    train_loader, val_loader = build_dataloaders(cfg, seed=cfg.seed, device=device)

    kg_data = load_real_kg()
    if kg_data is None:
        logger.warning("Real KG not found — using synthetic KG fallback")
        kg_data = PrimeKGMockGenerator(
            num_nodes=cfg.sgg_num_kg_nodes, kg_embed_dim=cfg.sgg_kg_embed_dim, seed=cfg.seed
        ).generate()
    cfg.sgg_num_kg_nodes = kg_data["num_nodes"]
    cfg.sgg_kg_embed_dim = kg_data["kg_embed_dim"]
    logger.info("KG: %d nodes | embed_dim=%d", cfg.sgg_num_kg_nodes, cfg.sgg_kg_embed_dim)

    model     = DJEPA(cfg, kg_data).to(device)

    # torch.compile: traces the model graph once, then emits fused CUDA kernels.
    # ~20-35% speedup on A100 for transformer + GRU workloads.
    # Skipped on MPS (unsupported) and CPU (overhead not worth it).
    if device.type == "cuda" and getattr(cfg, "compile_model", False):
        logger.info("Compiling model with torch.compile (mode=default)…")
        model = torch.compile(model, mode="default")
        logger.info("torch.compile done.")

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    # BF16 doesn't need loss scaling (larger exponent range prevents underflow).
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler    = GradScaler("cuda", enabled=use_scaler)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    p1_ckpt = os.path.join(cfg.checkpoint_dir, "phase1_best.pt")
    p2_ckpt = os.path.join(cfg.checkpoint_dir, "phase2_best.pt")

    logger.info("Trainable parameters: %s",
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    half_epochs  = cfg.num_epochs // 2
    fisher_dict: Optional[Dict[str, torch.Tensor]] = None
    old_params:  Optional[Dict[str, torch.Tensor]] = None
    global_step  = 0

    # Loss history for plotting
    history: Dict[str, List] = {
        k: [] for k in
        ["train_asr","train_mi","train_decorr","train_jepa","train_kg","train_ewc","train_total",
         "val_total"]
    }
    history["phase_boundary"] = half_epochs

    # -----------------------------------------------------------------------
    # Phase 1
    # -----------------------------------------------------------------------
    logger.info("=== Phase 1: Base task (epochs 1–%d) ===", half_epochs)
    best_p1_val = float("inf")

    accum = max(1, cfg.grad_accum_steps)

    for epoch in range(1, half_epochs + 1):
        model.train()
        run: Dict[str, float] = {k: 0.0 for k in ["asr","mi","decorr","jepa","kg","total"]}
        t0 = time.time()
        optimizer.zero_grad()

        for step_idx, batch in enumerate(train_loader):
            features     = batch["features"].to(device, non_blocking=True)
            asr_targets  = batch["asr_tokens"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)

            torch.compiler.cudagraph_mark_step_begin()
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                _, losses, _ = model(features, asr_targets, padding_mask)
                # Scale loss by accum steps so gradients are averaged correctly
                scaled = losses["total"] / accum

            if use_scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            for k in run: run[k] += losses[k].item()
            global_step += 1

            # Step only after accumulating `accum` mini-batches
            if (step_idx + 1) % accum == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            if global_step % cfg.log_interval == 0:
                logger.debug("[P1 E%02d S%d] %s", epoch, global_step,
                             " | ".join(f"{k}={v:.4f}" for k, v in losses.items()))

        scheduler.step()
        nb  = max(len(train_loader), 1)
        avg = {k: v / nb for k, v in run.items()}
        samples_per_sec = (nb * cfg.batch_size) / max(time.time() - t0, 1e-6)
        logger.info("P1 Epoch %02d/%d | %s | %.1fs (%.0f samp/s)", epoch, half_epochs,
                    " | ".join(f"{k}={v:.4f}" for k, v in avg.items()),
                    time.time() - t0, samples_per_sec)

        val_loss = _validate(model, val_loader, device, cfg, use_amp, amp_dtype)
        logger.info("  Val: %.4f%s", val_loss, "  ← best" if val_loss < best_p1_val else "")

        # Record history
        for k in ["asr","mi","decorr","jepa","kg","total"]:
            history[f"train_{k}"].append(avg[k])
        history["train_ewc"].append(0.0)
        history["val_total"].append(val_loss)

        if val_loss < best_p1_val:
            best_p1_val = val_loss
            save_checkpoint(p1_ckpt, epoch, 1, model, optimizer, scheduler,
                            scaler, val_loss, cfg)
            logger.info("  Saved Phase1 best → %s", p1_ckpt)

    # -----------------------------------------------------------------------
    # Fisher (protect Phase 1 weights before Phase 2 begins)
    # -----------------------------------------------------------------------
    if cfg.use_ewc:
        logger.info("Computing Fisher Information Matrix …")
        fisher_dict = compute_fisher_information(
            model, train_loader, device, num_samples=cfg.ewc_fisher_samples
        )
        old_params = {n: p.data.clone() for n, p in model.named_parameters()}
        logger.info("Fisher computed for %d tensors.", len(fisher_dict))

    # -----------------------------------------------------------------------
    # Phase 2 — independent best-val tracking starts at inf
    # -----------------------------------------------------------------------
    logger.info("=== Phase 2: Continual + EWC (epochs %d–%d) ===",
                half_epochs + 1, cfg.num_epochs)
    best_p2_val = float("inf")   # completely independent from Phase 1

    for epoch in range(half_epochs + 1, cfg.num_epochs + 1):
        model.train()
        run = {k: 0.0 for k in ["asr","mi","decorr","jepa","kg","ewc","total"]}
        t0  = time.time()
        optimizer.zero_grad()
        ewc_raw = 0.0

        for step_idx, batch in enumerate(train_loader):
            features     = batch["features"].to(device, non_blocking=True)
            asr_targets  = batch["asr_tokens"].to(device, non_blocking=True)
            padding_mask = batch["padding_mask"].to(device, non_blocking=True)

            torch.compiler.cudagraph_mark_step_begin()
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                _, losses, _ = model(features, asr_targets, padding_mask)
                if cfg.use_ewc and fisher_dict is not None:
                    loss_ewc, ewc_raw = compute_ewc_loss(
                        model, fisher_dict, old_params, cfg.lambda_ewc
                    )
                else:
                    loss_ewc, ewc_raw = torch.tensor(0.0, device=device), 0.0
                total_loss = (losses["total"] + loss_ewc) / accum

            if use_scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            for k in losses:
                if k != "total":
                    run[k] += losses[k].item()
            run["ewc"]   += loss_ewc.item()
            run["total"] += (total_loss.item() * accum)  # avoids double-counting
            global_step  += 1

            if (step_idx + 1) % accum == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            if global_step % cfg.log_interval == 0:
                logger.debug("[P2 E%02d S%d] %s | ewc=%.4f raw=%.6f",
                             epoch, global_step,
                             " | ".join(f"{k}={v:.4f}" for k, v in losses.items()),
                             loss_ewc.item(), ewc_raw)

        scheduler.step()
        nb  = max(len(train_loader), 1)
        avg = {k: v / nb for k, v in run.items()}
        elapsed = time.time() - t0
        samples_per_sec = (nb * cfg.batch_size) / max(elapsed, 1e-6)
        logger.info("P2 Epoch %02d/%d | %s | ewc=%.4f (raw=%.6f) | %.1fs (%.0f samp/s)",
                    epoch, cfg.num_epochs,
                    " | ".join(f"{k}={v:.4f}" for k in ["asr","mi","decorr","jepa","kg","total"]
                               for v in [avg[k]]),
                    avg["ewc"], ewc_raw, elapsed, samples_per_sec)

        val_loss = _validate(model, val_loader, device, cfg, use_amp, amp_dtype)
        logger.info("  Val: %.4f%s", val_loss, "  ← best" if val_loss < best_p2_val else "")

        # Record history
        for k in ["asr","mi","decorr","jepa","kg","total"]:
            history[f"train_{k}"].append(avg[k])
        history["train_ewc"].append(avg["ewc"])
        history["val_total"].append(val_loss)

        if val_loss < best_p2_val:
            best_p2_val = val_loss
            save_checkpoint(p2_ckpt, epoch, 2, model, optimizer, scheduler,
                            scaler, val_loss, cfg,
                            fisher_dict=fisher_dict, old_params=old_params)
            logger.info("  Saved Phase2 best → %s", p2_ckpt)

    logger.info("Done. P1 best val=%.4f | P2 best val=%.4f", best_p1_val, best_p2_val)

    # Save loss history JSON
    hist_path = os.path.join(cfg.log_dir, "loss_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Loss history → %s", hist_path)

    # Plot
    plot_loss_curves(history, cfg.log_dir)

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from model.config import get_config
    cfg = get_config(size="small")
    train(cfg)
