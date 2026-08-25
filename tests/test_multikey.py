from pathlib import Path

import pytest

from rmlp.config import load_config
from rmlp.multikey import (
    build_key_cover_pairs,
    cumulative_asr_rows,
    iteration_budgets,
    key_directory_name,
)


def test_one_to_one_pairing_is_not_cartesian_product() -> None:
    covers = [Path("a.png"), Path("b.png"), Path("c.png")]
    pairs = build_key_cover_pairs([0, 1, 2], covers)

    assert pairs == [(0, covers[0]), (1, covers[1]), (2, covers[2])]
    assert len(pairs) == 3


def test_one_to_one_pairing_rejects_count_mismatch_and_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="exactly one cover"):
        build_key_cover_pairs([0, 1], [Path("a.png")])
    with pytest.raises(ValueError, match="unique"):
        build_key_cover_pairs([0, 0], [Path("a.png"), Path("b.png")])


def test_iteration_budgets_include_nondivisible_final_budget() -> None:
    assert iteration_budgets(250, 100) == [100, 200, 250]


def test_cumulative_asr_uses_first_success_step() -> None:
    rows = [
        {
            "method": "baseline",
            "success": 1,
            "clean_false_positive": 0,
            "first_success_step": 300,
        },
        {
            "method": "baseline",
            "success": 0,
            "clean_false_positive": 0,
            "first_success_step": None,
        },
        {
            "method": "simple_average",
            "success": 1,
            "clean_false_positive": 0,
            "first_success_step": 100,
        },
        {
            "method": "simple_average",
            "success": 1,
            "clean_false_positive": 0,
            "first_success_step": 200,
        },
    ]

    curve = cumulative_asr_rows(rows, max_iterations=300, interval=100)

    assert curve[0]["baseline_asr"] == 0.0
    assert curve[0]["simple_average_asr"] == 0.5
    assert curve[1]["simple_average_asr"] == 1.0
    assert curve[2]["baseline_asr"] == 0.5
    assert curve[2]["asr_delta_simple_minus_baseline"] == 0.5
    assert curve[2]["eligible_asr_delta_simple_minus_baseline"] == 0.5


def test_multikey_formal_config_is_paired_and_fair() -> None:
    config = load_config("configs/tree_ring_multikey_paired_10x15000.yaml")

    assert config["multikey"]["key_seeds"] == list(range(10))
    assert config["data"]["cover_limit"] == 10
    assert config["prototype"]["reference_count"] == 5
    assert config["prototype"]["retain_count"] == 5
    assert config["attack"]["lambda_pixel"] == 10000.0
    assert config["attack"]["num_iterations"] == 15000
    assert config["attack"]["detection_every"] == 100
    assert config["attack"]["early_stop_on_success"] is True
    assert config["evaluation"]["compute_lpips"] is True
    assert key_directory_name(9) == "key_009"
