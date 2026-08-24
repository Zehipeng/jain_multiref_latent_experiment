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


def _finite_float32(latents: torch.Tensor, name: str) -> torch.Tensor:
    working = latents.detach().float()
    if not torch.isfinite(working).all():
        nonfinite = int((~torch.isfinite(working)).sum().item())
        raise ValueError(f"{name} contains {nonfinite} non-finite values")
    return working


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
    # Reference latents are normally encoded in fp16. Squaring fp16 residuals
    # can overflow even when every latent value is finite, which previously
    # produced an Infinity distance and an unreliable rejection decision.
    working = _finite_float32(latents, "latents")
    median_center = working.median(dim=0).values
    reduce_dims = tuple(range(1, latents.ndim))
    distances = (working - median_center).square().mean(dim=reduce_dims)
    if not torch.isfinite(distances).all():
        raise ValueError("Prototype distances contain non-finite values")
    order = torch.argsort(distances)
    retained = order[:retain_count]
    rejected = order[retain_count:]
    prototype = working[retained].mean(dim=0)
    return PrototypeResult(
        prototype=prototype,
        median_center=median_center,
        distances=distances,
        retained_indices=retained,
        rejected_indices=rejected,
    )


@torch.no_grad()
def simple_average_prototype(latents: torch.Tensor) -> torch.Tensor:
    """Average all finite reference latents in fp32."""
    if latents.ndim < 2 or latents.shape[0] < 1:
        raise ValueError("latents must have shape [N, ...] with N >= 1")
    return _finite_float32(latents, "latents").mean(dim=0)


@torch.no_grad()
def latent_statistics(latent: torch.Tensor) -> dict[str, float | int | bool]:
    working = latent.detach().float()
    finite_mask = torch.isfinite(working)
    finite = bool(finite_mask.all().item())
    stats: dict[str, float | int | bool] = {
        "finite": finite,
        "nonfinite_count": int((~finite_mask).sum().item()),
        "element_count": int(working.numel()),
    }
    if finite:
        stats.update(
            {
                "min": float(working.min().item()),
                "max": float(working.max().item()),
                "mean": float(working.mean().item()),
                "std": float(working.std(unbiased=False).item()),
                "l2_norm": float(torch.linalg.vector_norm(working).item()),
            }
        )
    return stats
