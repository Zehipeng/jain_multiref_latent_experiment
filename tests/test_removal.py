import torch
import pytest

from rmlp.config import load_config
from rmlp.image_io import latent_to_pil, shared_latent_scale
from rmlp.removal import mean_image_target, mean_shift_target, removal_mode
from evaluate_multikey_removal import _summarize_checkpoint_metrics


def test_mean_image_target_is_constant_and_preserves_shape() -> None:
    image = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    target = mean_image_target(image)
    assert target.shape == image.shape
    assert torch.all(target == 4.0)


def test_mean_shift_target_matches_defined_direction() -> None:
    target = torch.tensor([[[[5.0]]]])
    watermarked = torch.tensor([[[[4.0]]]])
    clean = torch.tensor([[[[1.0]]]])
    assert torch.equal(mean_shift_target(target, watermarked, clean, 1.0), torch.tensor([[[[2.0]]]]))


def test_removal_modes() -> None:
    assert removal_mode("latent_repulsion") == "repel"
    assert removal_mode("mean_shift") == "attract"


def test_latent_visualization_uses_shared_scale_and_four_channel_grid() -> None:
    first = torch.linspace(-2.0, 2.0, 4 * 8 * 8).reshape(4, 8, 8)
    second = first * 0.5
    scale = shared_latent_scale([first, second], quantile=1.0)
    image = latent_to_pil(first, scale=scale, tile_size=32)

    assert scale == 2.0
    assert image.mode == "RGB"
    assert image.size == (64, 108)


def test_removal_config_requires_local_model_cache() -> None:
    config = load_config("configs/tree_ring_multikey_removal_10x15000.yaml")

    assert config["model"]["local_files_only"] is True
    assert config["model"]["cache_dir"] == (
        "/root/autodl-tmp/cache/huggingface/hub"
    )


def test_small_core_removal_config_uses_fixed_budget_and_dense_checkpoints() -> None:
    config = load_config("configs/tree_ring_multikey_removal_10x1000.yaml")

    assert len(config["multikey"]["key_seeds"]) == 10
    assert config["attack"]["num_iterations"] == 1000
    assert config["attack"]["detection_every"] == 100
    assert config["attack"]["log_every"] == 100
    assert config["attack"]["early_stop_on_success"] is False
    assert config["attack"]["save_steps"] == list(range(100, 1001, 100))


def test_checkpoint_summary_groups_methods_and_steps() -> None:
    rows = [
        {"method": "mean_shift", "step": 0, "p_value": 0.01, "removed": 0, "l2": 0.0, "linf": 0.0, "psnr": float("inf"), "ssim": 1.0, "lpips": 0.0},
        {"method": "mean_shift", "step": 0, "p_value": 0.03, "removed": 0, "l2": 0.0, "linf": 0.0, "psnr": float("inf"), "ssim": 1.0, "lpips": 0.0},
        {"method": "mean_shift", "step": 100, "p_value": 0.10, "removed": 1, "l2": 2.0, "linf": 0.1, "psnr": 40.0, "ssim": 0.95, "lpips": 0.02},
        {"method": "mean_shift", "step": 100, "p_value": 0.20, "removed": 1, "l2": 4.0, "linf": 0.2, "psnr": 38.0, "ssim": 0.93, "lpips": 0.04},
    ]
    summary = _summarize_checkpoint_metrics(rows)
    final = next(row for row in summary if row["method"] == "mean_shift" and row["step"] == 100)

    assert final["n"] == 2
    assert final["mean_p_value"] == pytest.approx(0.15)
    assert final["mean_l2"] == pytest.approx(3.0)
    assert final["asr"] == 1.0
