from __future__ import annotations

import re
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import list_images as list_images


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def preprocess_pil(image: Image.Image, img_size: int) -> torch.Tensor:
    width, height = image.size
    scale = img_size / min(width, height)
    resized = image.resize(
        (round(width * scale), round(height * scale)), Image.Resampling.BICUBIC
    )
    left = (resized.width - img_size) // 2
    top = (resized.height - img_size) // 2
    cropped = resized.crop((left, top, left + img_size, top + img_size))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor.mul(2.0).sub(1.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().cpu()
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported for image export")
        image = image[0]
    image = image.clamp(-1, 1).add(1).div(2)
    array = image.permute(1, 2, 0).numpy()
    array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def shared_latent_scale(latents: list[torch.Tensor], quantile: float = 0.995) -> float:
    """Return one robust symmetric scale for comparable latent visualizations."""
    if not latents:
        raise ValueError("At least one latent tensor is required")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    values = torch.cat(
        [latent.detach().float().cpu().reshape(-1).abs() for latent in latents]
    )
    if not torch.isfinite(values).all():
        raise ValueError("Latent visualization input contains non-finite values")
    return max(float(torch.quantile(values, quantile).item()), 1e-12)


def latent_to_pil(
    latent: torch.Tensor,
    scale: float,
    tile_size: int = 192,
) -> Image.Image:
    """Render latent channels as a blue-white-red grid with a fixed scale.

    Blue denotes negative values, white denotes zero, and red denotes positive
    values.  A shared ``scale`` keeps different latent tensors comparable.  The
    visualization is descriptive only; exact values must be read from the
    corresponding ``.pt`` artifact.
    """
    working = latent.detach().float().cpu()
    if working.ndim == 4:
        if working.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported")
        working = working[0]
    if working.ndim != 3:
        raise ValueError("latent must have shape [C,H,W] or [1,C,H,W]")
    if scale <= 0 or tile_size <= 0:
        raise ValueError("scale and tile_size must be positive")
    if not torch.isfinite(working).all():
        raise ValueError("latent contains non-finite values")

    channels = int(working.shape[0])
    columns = math.ceil(math.sqrt(channels))
    rows = math.ceil(channels / columns)
    label_height = 22
    canvas = Image.new(
        "RGB", (columns * tile_size, rows * (tile_size + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    normalized = (working / scale).clamp(-1.0, 1.0).numpy()
    for channel in range(channels):
        value = normalized[channel]
        red = np.where(value >= 0.0, 255.0, 255.0 * (1.0 + value))
        green = 255.0 * (1.0 - np.abs(value))
        blue = np.where(value <= 0.0, 255.0, 255.0 * (1.0 - value))
        rgb = np.stack((red, green, blue), axis=-1)
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        tile = Image.fromarray(rgb, mode="RGB").resize(
            (tile_size, tile_size), Image.Resampling.NEAREST
        )
        x = (channel % columns) * tile_size
        y = (channel // columns) * (tile_size + label_height)
        canvas.paste(tile, (x, y + label_height))
        draw.text((x + 5, y + 4), f"channel {channel}", fill="black")
    return canvas


def safe_stem(path: str | Path) -> str:
    stem = Path(path).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "image"
