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
    assert model["proxy_vae_revision"] == (
        "133a221b8aa7292a167afc5127cb63fb5005638b"
    )


def test_formal_config_uses_same_models_and_reference_bank() -> None:
    smoke = load_config("configs/tree_ring_cross_model_smoke.yaml")
    formal = load_config("configs/tree_ring_cross_model_formal.yaml")

    assert formal["model"] == smoke["model"]
    assert formal["data"]["references_dir"] == smoke["data"]["references_dir"]
    assert smoke["data"]["cover_dir"] == "data/MS-COCO"
    assert formal["data"]["cover_dir"] == "data/MS-COCO"
    assert formal["data"]["cover_limit"] == 10
    assert formal["attack"]["num_iterations"] == 3000
    assert formal["evaluation"]["compute_lpips"] is True


def test_simple_average_lambda1e4_smoke_uses_periodic_early_stop() -> None:
    smoke = load_config("configs/tree_ring_cross_model_smoke.yaml")
    config = load_config("configs/tree_ring_simple_average_lambda1e4_smoke.yaml")

    assert config["model"] == smoke["model"]
    assert config["data"]["references_dir"] == smoke["data"]["references_dir"]
    assert config["data"]["cover_dir"] == "data/MS-COCO"
    assert config["data"]["cover_limit"] == 2
    assert config["prototype"]["reference_count"] == 5
    assert config["prototype"]["aggregation"] == "simple_average_all_references"
    assert config["attack"]["lambda_pixel"] == 10000.0
    assert config["attack"]["num_iterations"] == 3000
    assert config["attack"]["detection_every"] == 100
    assert config["attack"]["early_stop_on_success"] is True
