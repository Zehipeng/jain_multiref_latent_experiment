import torch

from rmlp.config import load_config
from rmlp.image_io import latent_to_pil, shared_latent_scale
from rmlp.removal import mean_image_target, mean_shift_target, removal_mode


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
