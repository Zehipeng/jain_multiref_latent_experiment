from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(directory: str | Path, limit: int | None = None) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No supported images found in: {root}")
    return paths


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


def safe_stem(path: str | Path) -> str:
    stem = Path(path).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "image"

