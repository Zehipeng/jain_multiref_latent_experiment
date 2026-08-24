from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrototypeResult:
    prototype: torch.Tensor
    median_center: torch.Tensor
    distances: torch.Tensor
    retained_indices: torch.Tensor
    rejected_indices: torch.Tensor


@torch.no_grad()
def encode_vae_latent(vae: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    dtype = next(vae.parameters()).dtype
    device = next(vae.parameters()).device
    image = image.to(device=device, dtype=dtype)
    return vae.encode(image).latent_dist.mode() * (1.0 / vae.config.scaling_factor)


@torch.no_grad()
def robust_latent_prototype(
    latents: torch.Tensor, retain_count: int
) -> PrototypeResult:
    """Median-centered sample filtering followed by a retained sample mean."""
    if latents.ndim < 2:
        raise ValueError("latents must have shape [N, ...]")
    n_refs = latents.shape[0]
    if not 1 <= retain_count <= n_refs:
        raise ValueError("retain_count must be in [1, N]")
    median_center = latents.median(dim=0).values
    reduce_dims = tuple(range(1, latents.ndim))
    distances = (latents - median_center).square().mean(dim=reduce_dims)
    order = torch.argsort(distances)
    retained = order[:retain_count]
    rejected = order[retain_count:]
    prototype = latents[retained].mean(dim=0)
    return PrototypeResult(
        prototype=prototype,
        median_center=median_center,
        distances=distances,
        retained_indices=retained,
        rejected_indices=rejected,
    )

