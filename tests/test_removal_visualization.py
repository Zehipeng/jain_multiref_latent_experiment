from pathlib import Path

from rmlp.config import load_config
from run_removal_visualization import EXPECTED_STEPS, _validate_visualization_config, select_clean_prior_paths


def test_removal_visualization_config_matches_locked_protocol() -> None:
    config = load_config("configs/tree_ring_removal_visualization_1x3000.yaml")
    _validate_visualization_config(config)
    assert config["watermark"]["w_seed"] == 52
    assert config["prototype"]["reference_count"] == 6
    assert config["removal"]["reference_count_for_mean"] == 5
    assert config["removal"]["heldout_target_index"] == 5
    assert config["removal"]["target_in_reference_aggregate"] is False
    assert config["removal"]["method"] == "mean_shift"
    assert config["removal"]["beta"] == 1.0
    assert config["attack"]["early_stop_on_success"] is False
    assert set(config["attack"]["save_steps"]) == EXPECTED_STEPS


def test_clean_prior_selection_uses_human_positions_1314_through_1318() -> None:
    paths = [Path(f"image_{index:04d}.png") for index in range(1, 1320)]
    selected = select_clean_prior_paths(paths, 1314, 5)
    assert selected == [Path(f"image_{index:04d}.png") for index in range(1314, 1319)]
