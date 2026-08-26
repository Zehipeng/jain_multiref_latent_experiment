from __future__ import annotations

from typing import Literal

import torch


REMOVAL_METHODS = (
    "jain_mean_image",
    "latent_repulsion",
    "clean_attraction",
    "mean_shift",
)

RemovalMode = Literal["attract", "repel"]


def mean_image_target(image: torch.Tensor) -> torch.Tensor:
    """Return Jain's constant mean-image guidance target in image space."""
    return torch.full_like(image, image.detach().mean())


def mean_shift_target(
    target_watermarked_latent: torch.Tensor,
    watermarked_mean: torch.Tensor,
    clean_mean: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Construct z_t - beta * (zbar_w - zbar_c)."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    return target_watermarked_latent.float() - beta * (
        watermarked_mean.float() - clean_mean.float()
    )


def removal_mode(method: str) -> RemovalMode:
    if method == "latent_repulsion":
        return "repel"
    if method in {"jain_mean_image", "clean_attraction", "mean_shift"}:
        return "attract"
    raise ValueError(f"Unknown removal method: {method}")
