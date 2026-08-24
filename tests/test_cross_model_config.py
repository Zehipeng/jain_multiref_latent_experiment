from rmlp.config import load_config


def test_cross_model_config_pins_sd2_archive_and_keeps_sd14_proxy() -> None:
    config = load_config("configs/tree_ring_cross_model_smoke.yaml")
    model = config["model"]

    assert model["target_model_id"] == "sd2-community/stable-diffusion-2-base"
    assert model["target_model_revision"] == (
        "f5bc1bd97485577aa0b946fa8a9004e2ec147402"
    )
    assert model["target_model_variant"] == "fp16"
    assert model["proxy_vae_model_id"] == "CompVis/stable-diffusion-v1-4"
