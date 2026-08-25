from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    prototype = config["prototype"]
    n_refs = int(prototype["reference_count"])
    n_keep = int(prototype["retain_count"])
    seeds = config["watermark"]["generation_seeds"]
    selection = config.get("reference_selection", {})
    max_candidates = int(selection.get("max_candidates", len(seeds)))
    if n_refs < 1:
        raise ValueError("prototype.reference_count must be positive")
    if not 1 <= n_keep <= n_refs:
        raise ValueError("prototype.retain_count must be in [1, reference_count]")
    if len(seeds) < n_refs:
        raise ValueError("Not enough watermark.generation_seeds for reference_count")
    if max_candidates < n_refs:
        raise ValueError(
            "reference_selection.max_candidates must be at least reference_count"
        )
    if float(config["attack"]["alpha"]) <= 0:
        raise ValueError("attack.alpha must be positive")
    if int(config["attack"]["num_iterations"]) <= 0:
        raise ValueError("attack.num_iterations must be positive")
    snapshot_cover_limit = int(config["attack"].get("snapshot_cover_limit", 1))
    if snapshot_cover_limit < 0:
        raise ValueError("attack.snapshot_cover_limit must be non-negative")
    if "multikey" in config:
        key_seeds = [int(seed) for seed in config["multikey"]["key_seeds"]]
        if not key_seeds:
            raise ValueError("multikey.key_seeds must not be empty")
        if len(set(key_seeds)) != len(key_seeds):
            raise ValueError("multikey.key_seeds must be unique")
        if any(seed < 0 for seed in key_seeds):
            raise ValueError("multikey.key_seeds must be non-negative")
        if int(config["data"]["cover_limit"]) != len(key_seeds):
            raise ValueError(
                "Paired multi-key config requires cover_limit == number of keys"
            )


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path
