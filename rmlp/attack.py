from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class AttackResult:
    adversarial: torch.Tensor
    history: list[dict[str, float | int]]


def optimize_to_target_latent(
    cover: torch.Tensor,
    target_latent: torch.Tensor,
    vae: torch.nn.Module,
    lambda_pixel: float,
    alpha: float,
    num_iterations: int,
    log_every: int,
    snapshot_steps: set[int] | None = None,
    snapshot_callback: Callable[[int, torch.Tensor], None] | None = None,
    progress_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> AttackResult:
    """Jain latent MSE + pixel MSE optimization with a precomputed target."""
    if cover.ndim == 3:
        cover = cover.unsqueeze(0)
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    clean = cover.to(device=device, dtype=dtype).detach()
    target = target_latent.to(device=device, dtype=dtype).detach()
    current = clean.clone().detach()
    history: list[dict[str, float | int]] = []
    snapshot_steps = snapshot_steps or set()

    vae.eval()
    vae.requires_grad_(False)
    for step in range(1, num_iterations + 1):
        current.requires_grad_(True)
        encoded = (
            vae.encode(current).latent_dist.mode()
            * (1.0 / vae.config.scaling_factor)
        )
        latent_loss = F.mse_loss(encoded, target)
        pixel_loss = F.mse_loss(current, clean)
        total_loss = latent_loss + lambda_pixel * pixel_loss
        gradient = torch.autograd.grad(total_loss, current, only_inputs=True)[0]
        current = (current - alpha * gradient).clamp(-1.0, 1.0).detach()

        if step == 1 or step % log_every == 0 or step == num_iterations:
            record = {
                "step": step,
                "latent_loss": float(latent_loss.detach().float().cpu()),
                "pixel_loss": float(pixel_loss.detach().float().cpu()),
                "total_loss": float(total_loss.detach().float().cpu()),
            }
            history.append(record)
            if progress_callback is not None:
                progress_callback(record)
        if step in snapshot_steps and snapshot_callback is not None:
            snapshot_callback(step, current)

    return AttackResult(adversarial=current, history=history)
