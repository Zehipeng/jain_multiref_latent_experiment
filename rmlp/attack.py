from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import torch.nn.functional as F


@dataclass
class AttackResult:
    adversarial: torch.Tensor
    history: list[dict[str, float | int]]
    detection_history: list[dict[str, float | int | bool]]
    executed_iterations: int
    first_success_step: int | None
    first_success_p_value: float | None


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
    detection_every: int | None = None,
    detection_callback: Callable[[int, torch.Tensor], float] | None = None,
    detection_threshold: float = 0.05,
    stop_on_detection: bool = False,
    maximize_latent_distance: bool = False,
    detection_success: Literal["le", "ge"] = "le",
) -> AttackResult:
    """Optimize a latent-MSE objective with an image-space fidelity penalty.

    The default reproduces the Jain-style attraction objective.  Setting
    ``maximize_latent_distance`` implements the repulsion ablation while
    retaining the same pixel-fidelity term and gradient-descent update.
    """
    if cover.ndim == 3:
        cover = cover.unsqueeze(0)
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    clean = cover.to(device=device, dtype=dtype).detach()
    target = target_latent.to(device=device, dtype=dtype).detach()
    current = clean.clone().detach()
    history: list[dict[str, float | int]] = []
    detection_history: list[dict[str, float | int | bool]] = []
    snapshot_steps = snapshot_steps or set()
    if detection_every is not None and detection_every <= 0:
        raise ValueError("detection_every must be positive when enabled")
    if detection_every is not None and detection_callback is None:
        raise ValueError("detection_callback is required when periodic detection is enabled")
    if detection_success not in {"le", "ge"}:
        raise ValueError("detection_success must be either 'le' or 'ge'")
    first_success_step: int | None = None
    first_success_p_value: float | None = None
    executed_iterations = 0

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
        latent_sign = -1.0 if maximize_latent_distance else 1.0
        total_loss = latent_sign * latent_loss + lambda_pixel * pixel_loss
        gradient = torch.autograd.grad(total_loss, current, only_inputs=True)[0]
        current = (current - alpha * gradient).clamp(-1.0, 1.0).detach()
        executed_iterations = step

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

        if detection_every is not None and step % detection_every == 0:
            p_value = float(detection_callback(step, current))
            success = (
                p_value <= detection_threshold
                if detection_success == "le"
                else p_value >= detection_threshold
            )
            detection_history.append(
                {"step": step, "p_value": p_value, "success": success}
            )
            if success and first_success_step is None:
                first_success_step = step
                first_success_p_value = p_value
                if stop_on_detection:
                    break

    return AttackResult(
        adversarial=current,
        history=history,
        detection_history=detection_history,
        executed_iterations=executed_iterations,
        first_success_step=first_success_step,
        first_success_p_value=first_success_p_value,
    )
