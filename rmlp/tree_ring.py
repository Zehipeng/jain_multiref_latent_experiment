from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers import DDIMInverseScheduler, StableDiffusionPipeline
from PIL import Image
from scipy import stats

from .image_io import preprocess_pil


def circle_mask(size: int, radius: int) -> np.ndarray:
    center = size // 2
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    return (x - center) ** 2 + (y - center) ** 2 <= radius**2


def latent_shape(pipe: StableDiffusionPipeline, img_size: int) -> tuple[int, ...]:
    channels = int(pipe.unet.config.in_channels)
    return (1, channels, img_size // 8, img_size // 8)


def build_watermark_mask(
    shape: tuple[int, ...], channel: int, radius: int, device: str | torch.device
) -> torch.Tensor:
    if not 0 <= channel < shape[1]:
        raise ValueError(f"w_channel={channel} is outside latent channels={shape[1]}")
    mask_2d = torch.as_tensor(
        circle_mask(shape[-1], radius), dtype=torch.bool, device=device
    )
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    mask[:, channel] = mask_2d
    return mask


def _prepare_random_latents(
    pipe: StableDiffusionPipeline, img_size: int, generator: torch.Generator
) -> torch.Tensor:
    return pipe.prepare_latents(
        1,
        int(pipe.unet.config.in_channels),
        img_size,
        img_size,
        pipe.unet.dtype,
        pipe.device,
        generator,
    )


def build_jain_ring_key(
    pipe: StableDiffusionPipeline,
    shape: tuple[int, ...],
    w_seed: int,
    img_size: int,
) -> torch.Tensor:
    """Reproduce Jain Tree-Ring/utils.py::get_pattern."""
    generator = torch.Generator(device=pipe.device).manual_seed(w_seed)
    initial = _prepare_random_latents(pipe, img_size, generator)
    # PyTorch 2.1 creates experimental complex32 values for float16 FFTs and
    # cannot serialize that dtype. Compute Tree-Ring FFTs in float32 so keys
    # are stable complex64 tensors on every supported AutoDL image.
    key = torch.fft.fftshift(torch.fft.fft2(initial.float()), dim=(-1, -2))
    source = key.clone().detach()
    for radius in range(shape[-1] // 2, 0, -1):
        ring = torch.as_tensor(
            circle_mask(initial.shape[-1], radius),
            dtype=torch.bool,
            device=pipe.device,
        )
        for channel in range(key.shape[1]):
            key[:, channel, ring] = source[0, channel, 0, radius]
    return key


def generate_watermarked_image(
    pipe: StableDiffusionPipeline,
    prompt: str,
    w_key: torch.Tensor,
    w_mask: torch.Tensor,
    img_size: int,
    generation_seed: int,
    num_inference_steps: int,
) -> Image.Image:
    # The generation seed is deliberately independent of w_seed.
    generator = torch.Generator(device=pipe.device).manual_seed(generation_seed)
    initial = _prepare_random_latents(pipe, img_size, generator)
    initial_fft = torch.fft.fftshift(
        torch.fft.fft2(initial.float()), dim=(-1, -2)
    )
    initial_fft[w_mask] = w_key[w_mask].clone()
    watermarked_latents = torch.fft.ifft2(
        torch.fft.ifftshift(initial_fft, dim=(-1, -2))
    ).real
    watermarked_latents = torch.nan_to_num(
        watermarked_latents, nan=0.0, posinf=4.0, neginf=-4.0
    ).to(dtype=pipe.unet.dtype)
    result = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        latents=watermarked_latents,
    )
    return result.images[0]


@torch.inference_mode()
def detect_p_value(
    image: Image.Image | torch.Tensor,
    pipe: StableDiffusionPipeline,
    w_key: torch.Tensor,
    w_mask: torch.Tensor,
    img_size: int,
    inversion_steps: int,
) -> float:
    """Reproduce Jain's DDIM inversion and non-central chi-square p-value."""
    original_scheduler = pipe.scheduler
    try:
        pipe.scheduler = DDIMInverseScheduler.from_config(original_scheduler.config)
        if isinstance(image, Image.Image):
            tensor = preprocess_pil(image, img_size).unsqueeze(0)
            tensor = tensor.to(device=pipe.device, dtype=pipe.vae.dtype)
        else:
            tensor = image.to(device=pipe.device, dtype=pipe.vae.dtype)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
        image_latents = (
            pipe.vae.encode(tensor).latent_dist.mode()
            * (1.0 / pipe.vae.config.scaling_factor)
        )
        inverted = pipe(
            prompt="",
            latents=image_latents,
            guidance_scale=1.0,
            num_inference_steps=inversion_steps,
            output_type="latent",
        ).images
        inverted_fft = torch.fft.fftshift(
            torch.fft.fft2(inverted.float()), dim=(-1, -2)
        )[w_mask].flatten()
        target = w_key[w_mask].flatten()
        observed = torch.cat((inverted_fft.real, inverted_fft.imag)).float()
        target_real = torch.cat((target.real, target.imag)).float()
        sigma = observed.std().clamp_min(1e-12)
        noncentrality = (target_real.square() / sigma.square()).sum().item()
        statistic = (((observed - target_real) / sigma).square()).sum().item()
        return float(
            stats.ncx2.cdf(
                x=statistic, df=target_real.numel(), nc=noncentrality
            )
        )
    finally:
        pipe.scheduler = original_scheduler


def build_key_and_mask(
    pipe: StableDiffusionPipeline, config: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    wm = config["watermark"]
    shape = latent_shape(pipe, int(wm["img_size"]))
    key = build_jain_ring_key(
        pipe, shape, int(wm["w_seed"]), int(wm["img_size"])
    )
    mask = build_watermark_mask(
        shape, int(wm["w_channel"]), int(wm["w_radius"]), pipe.device
    )
    return key, mask
