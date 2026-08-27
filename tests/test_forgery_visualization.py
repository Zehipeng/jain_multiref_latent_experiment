import torch

from rmlp.config import load_config
from rmlp.image_io import fit_shared_latent_pca, latent_pca_to_pil
from run_forgery_visualization import EXPECTED_SAVE_STEPS, _validate_visualization_config


def test_shared_pca_renders_each_four_channel_latent_as_one_rgb_panel() -> None:
    generator = torch.Generator().manual_seed(7)
    clean = torch.randn(1, 4, 8, 8, generator=generator)
    watermarked = torch.randn(1, 4, 8, 8, generator=generator) + 0.5
    projection = fit_shared_latent_pca(
        [clean, watermarked], lower_quantile=0.0, upper_quantile=1.0
    )
    image = latent_pca_to_pil(clean, projection, output_size=64)

    assert projection["components"].shape == (4, 3)
    assert image.mode == "RGB"
    assert image.size == (64, 64)


def test_shared_pca_is_deterministic_and_uses_one_mapping() -> None:
    first = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)
    second = first.flip(-1) * 0.25
    fit_a = fit_shared_latent_pca([first, second], 0.0, 1.0)
    fit_b = fit_shared_latent_pca([first, second], 0.0, 1.0)

    for key in ("channel_mean", "components", "lower", "upper"):
        assert torch.allclose(fit_a[key], fit_b[key])
    assert latent_pca_to_pil(first, fit_a, 32).tobytes() == latent_pca_to_pil(
        first, fit_b, 32
    ).tobytes()


def test_forgery_visualization_config_is_locked_to_confirmed_protocol() -> None:
    config = load_config("configs/tree_ring_forgery_visualization_1x3000.yaml")
    _validate_visualization_config(config)

    assert config["watermark"]["w_seed"] == 0
    assert config["prototype"]["reference_count"] == 5
    assert config["attack"]["lambda_pixel"] == 10000.0
    assert config["attack"]["alpha"] == 5 / 255
    assert config["attack"]["num_iterations"] == 3000
    assert config["attack"]["log_every"] == 500
    assert config["attack"]["detection_every"] is None
    assert config["attack"]["early_stop_on_success"] is False
    assert set(config["attack"]["save_steps"]) == EXPECTED_SAVE_STEPS
    assert config["visualization"]["cover_position_1based"] == 1314
    assert config["model"]["local_files_only"] is True
    assert config["model"]["cache_dir"] == "/root/autodl-tmp/cache/huggingface/hub"
