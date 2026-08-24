from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_target_pipeline(config: dict[str, Any]) -> StableDiffusionPipeline:
    model_cfg = config["model"]
    dtype = resolve_dtype(model_cfg["dtype"])
    kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if model_cfg.get("disable_safety_checker", True):
        kwargs.update(safety_checker=None, requires_safety_checker=False)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_cfg["target_model_id"], **kwargs
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(model_cfg["device"])
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_proxy_vae(config: dict[str, Any]) -> AutoencoderKL:
    model_cfg = config["model"]
    dtype = resolve_dtype(model_cfg["dtype"])
    vae = AutoencoderKL.from_pretrained(
        model_cfg["proxy_vae_model_id"], subfolder="vae", torch_dtype=dtype
    ).to(model_cfg["device"])
    vae.eval()
    vae.requires_grad_(False)
    return vae

