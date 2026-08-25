from __future__ import annotations

from pathlib import Path
from typing import Any


METHODS = ("baseline", "simple_average")


def key_directory_name(w_seed: int) -> str:
    if w_seed < 0:
        raise ValueError("w_seed must be non-negative")
    return f"key_{w_seed:03d}"


def build_key_cover_pairs(
    key_seeds: list[int], cover_paths: list[Path]
) -> list[tuple[int, Path]]:
    if not key_seeds:
        raise ValueError("At least one key seed is required")
    if len(set(key_seeds)) != len(key_seeds):
        raise ValueError("Key seeds must be unique")
    if len(cover_paths) != len(key_seeds):
        raise ValueError(
            "One-to-one design requires exactly one cover for every key seed"
        )
    return list(zip(key_seeds, cover_paths))


def iteration_budgets(max_iterations: int, interval: int) -> list[int]:
    if max_iterations <= 0 or interval <= 0:
        raise ValueError("Iteration limits must be positive")
    budgets = list(range(interval, max_iterations + 1, interval))
    if not budgets or budgets[-1] != max_iterations:
        budgets.append(max_iterations)
    return budgets


def cumulative_asr_rows(
    metric_rows: list[dict[str, Any]],
    max_iterations: int,
    interval: int,
) -> list[dict[str, float | int]]:
    """Build success-by-budget curves from paired earliest-hit records."""
    budgets = iteration_budgets(max_iterations, interval)
    output: list[dict[str, float | int]] = []
    grouped = {
        method: [row for row in metric_rows if row["method"] == method]
        for method in METHODS
    }
    for step in budgets:
        record: dict[str, float | int] = {"step": step}
        for method, rows in grouped.items():
            eligible = [row for row in rows if not int(row["clean_false_positive"])]
            successes = sum(
                int(row["success"])
                and row.get("first_success_step") is not None
                and int(row["first_success_step"]) <= step
                for row in rows
            )
            eligible_successes = sum(
                int(row["success"])
                and row.get("first_success_step") is not None
                and int(row["first_success_step"]) <= step
                for row in eligible
            )
            record[f"{method}_successes"] = successes
            record[f"{method}_asr"] = successes / len(rows) if rows else 0.0
            record[f"{method}_eligible_n"] = len(eligible)
            record[f"{method}_eligible_successes"] = eligible_successes
            record[f"{method}_eligible_asr"] = (
                eligible_successes / len(eligible) if eligible else 0.0
            )
        record["asr_delta_simple_minus_baseline"] = float(
            record["simple_average_asr"]
        ) - float(record["baseline_asr"])
        record["eligible_asr_delta_simple_minus_baseline"] = float(
            record["simple_average_eligible_asr"]
        ) - float(record["baseline_eligible_asr"])
        output.append(record)
    return output
