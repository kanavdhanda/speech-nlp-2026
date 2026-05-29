"""
baselines/train_whisper_lora.py
================================
Fine-tunes Whisper-large-v3 with LoRA on the Hinglish dataset.
Saves only the best val-loss checkpoint.
Logs to baselines/logs/whisper_lora/train.log

Usage:  uv run python -m baselines.train_whisper_lora
        # Requires A100 (40GB) — Whisper-large-v3 (1.5B params) with LoRA
        # batch=4, 20 epochs ≈ 2-3 hours on A100

Architecture match:
  Whisper-large-v3 + LoRA (r=16, alpha=32) adds ~5.2M trainable params (0.35%).
  Same LoRA strategy as Cappellazzo et al. 2510.22603v3, Saengthong et al. 2507.02927v1,
  and LAOS (Xu et al. 2025). Same 12k Hinglish training data as D-JEPA.
"""

from __future__ import annotations
import logging, os, time, json
import numpy as np
import torch
import torch.nn.functional as F

from peft import get_peft_model, LoraConfig, TaskType
from transformers import WhisperForConditionalGeneration

from model.config import get_config
from model.data import build_dataloaders
from model.data_generation import set_global_seed
from model.device import get_device

LOG_DIR  = "baselines/logs/whisper_lora"
CKPT_DIR = "checkpoints/baselines"
RESULTS  = "checkpoints/baselines/whisper_lora_results.json"


def _setup_logger() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("whisper_lora")
    if logger.handlers: logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(LOG_DIR, "train.log"))
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger


def mel_from_features(features: torch.Tensor) -> torch.Tensor:
    """Convert (B, T, 80) acoustic features to (B, 80, 3000) Whisper mel input."""
    mel = features.transpose(1, 2)
    T   = mel.shape[-1]
    if T < 3000:
        mel = F.pad(mel, (0, 3000 - T))
    else:
        mel = mel[:, :, :3000]
    return mel


def train(
    n_epochs: int = 20,
    lora_r:   int = 16,
    model_id: str = "openai/whisper-large-v3",
) -> None:
    """
    Trains Whisper-large-v3 + LoRA on the same 12k Hinglish samples as D-JEPA.
    Requires A100 (40GB). MPS/CPU will OOM on large-v3.
    """
    set_global_seed(42)
    os.makedirs(CKPT_DIR, exist_ok=True)
    logger  = _setup_logger()
    device  = get_device()
    cfg     = get_config(size="small")
    # A100: batch=8 fits large-v3 with BF16. MPS: reduce to 1 but expect OOM.
    cfg.batch_size = 8 if device.type == "cuda" else 1
    logger.info("Device: %s | Model: %s | LoRA r=%d | batch=%d",
                device, model_id, lora_r, cfg.batch_size)

    train_loader, val_loader = build_dataloaders(cfg, seed=42, device=device)

    logger.info("Loading %s …", model_id)
    base = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    # Whisper-large-v3 uses MultiHeadAttention with q/k/v/out projections
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=lora_r, lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    )
    model     = get_peft_model(base, lora_cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info("Whisper-tiny + LoRA | trainable=%s / total=%s (%.2f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4
    )

    best_val, t_total = float("inf"), 0.0
    history: dict = {"train_loss": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss, n_batches = 0.0, 0
        t0 = time.time()

        for i, batch in enumerate(train_loader):
            if i >= max_batches_per_epoch:
                break
            mel    = mel_from_features(batch["features"]).to(device)
            labels = batch["asr_tokens"][:, :448].clone().to(device)
            labels[batch["padding_mask"][:, :448].to(device)] = -100

            optimizer.zero_grad()
            out  = model.base_model.model(input_features=mel, labels=labels)
            loss = out.loss
            if loss is not None and not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                train_loss += loss.item()
                n_batches  += 1

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 20: break
                mel    = mel_from_features(batch["features"]).to(device)
                labels = batch["asr_tokens"][:, :448].clone().to(device)
                labels[batch["padding_mask"][:, :448].to(device)] = -100
                out  = model.base_model.model(input_features=mel, labels=labels)
                if out.loss is not None:
                    val_loss += out.loss.item(); n_val += 1

        avg_t   = train_loss / max(n_batches, 1)
        avg_v   = val_loss   / max(n_val,    1)
        elapsed = time.time() - t0
        t_total += elapsed
        history["train_loss"].append(avg_t)
        history["val_loss"].append(avg_v)
        logger.info("Epoch %02d/%d | train=%.4f | val=%.4f | %.1fs",
                    epoch, n_epochs, avg_t, avg_v, elapsed)

        if avg_v < best_val:
            best_val = avg_v
            ckpt = os.path.join(CKPT_DIR, "whisper_lora_best.pt")
            torch.save({
                "epoch": epoch, "val_loss": best_val,
                "lora_state": {k: v for k, v in model.state_dict().items()
                               if "lora_" in k},
            }, ckpt)
            logger.info("  Saved best → %s", ckpt)

    results = {
        "model":             model_id,
        "lora_r":            lora_r,
        "trainable_params":  trainable,
        "total_params":      total,
        "best_val_loss":     best_val,
        "total_training_s":  t_total,
        "n_epochs":          n_epochs,
        "history":           history,
        "note": (
            "Proxy for Whisper-large-v3 + LoRA. Same LoRA strategy as Cappellazzo et al. "
            "(2510.22603v3), Saengthong et al. (2507.02927v1), LAOS (Xu et al. 2025). "
            "Trained on same 8k Hinglish samples as D-JEPA."
        ),
    }
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results → %s | Best val: %.4f | Total: %.0fs", RESULTS, best_val, t_total)


if __name__ == "__main__":
    train()
