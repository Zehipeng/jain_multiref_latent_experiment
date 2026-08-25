from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from rmlp.config import load_config
from rmlp.image_io import load_rgb, preprocess_pil
from rmlp.models import load_target_pipeline, seed_everything
from rmlp.multikey import METHODS, cumulative_asr_rows
from rmlp.reproducibility import git_provenance, runtime_provenance, sha256_file
from rmlp.tree_ring import build_key_and_mask, detect_p_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate paired multi-key attacks and ASR-by-iteration curves."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--no-lpips", action="store_true")
    return parser.parse_args()


def image_array(path: Path) -> np.ndarray:
    return np.asarray(load_rgb(path), dtype=np.float32) / 255.0


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(config["_config_path"])
    if manifest.get("config", {}).get("sha256") != config_sha256:
        raise ValueError("Evaluation config does not match attack manifest")
    if manifest.get("methods") != list(METHODS):
        raise ValueError("Expected exactly baseline and simple_average methods")

    pipe = load_target_pipeline(config)
    img_size = int(config["watermark"]["img_size"])
    threshold = float(config["evaluation"]["p_value_threshold"])
    inversion_steps = int(config["evaluation"]["inversion_steps"])
    use_lpips = bool(config["evaluation"].get("compute_lpips", True)) and not args.no_lpips
    lpips_model = None
    if use_lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to(config["model"]["device"]).eval()

    detector_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    clean_p_cache: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    for entry in tqdm(manifest["entries"], desc="evaluate paired attacks"):
        pair_index = int(entry["pair_index"])
        w_seed = int(entry["w_seed"])
        if w_seed not in detector_cache:
            key_config = deepcopy(config)
            key_config["watermark"]["w_seed"] = w_seed
            detector_cache[w_seed] = build_key_and_mask(pipe, key_config)
        w_key, w_mask = detector_cache[w_seed]
        clean_path = run_dir / entry["clean_export"]
        output_path = run_dir / entry["output"]
        if pair_index not in clean_p_cache:
            clean_p_cache[pair_index] = detect_p_value(
                load_rgb(clean_path), pipe, w_key, w_mask, img_size, inversion_steps
            )
        attack_p = detect_p_value(
            load_rgb(output_path), pipe, w_key, w_mask, img_size, inversion_steps
        )
        clean = image_array(clean_path)
        attacked = image_array(output_path)
        psnr = float(peak_signal_noise_ratio(clean, attacked, data_range=1.0))
        ssim = float(
            structural_similarity(clean, attacked, data_range=1.0, channel_axis=2)
        )
        lpips_value = None
        if lpips_model is not None:
            clean_tensor = preprocess_pil(load_rgb(clean_path), img_size).unsqueeze(0)
            attacked_tensor = preprocess_pil(load_rgb(output_path), img_size).unsqueeze(0)
            with torch.inference_mode():
                lpips_value = float(
                    lpips_model(
                        clean_tensor.to(config["model"]["device"]),
                        attacked_tensor.to(config["model"]["device"]),
                    ).item()
                )
        clean_p = clean_p_cache[pair_index]
        rows.append(
            {
                "pair_index": pair_index,
                "w_seed": w_seed,
                "sample_id": entry["sample_id"],
                "method": entry["method"],
                "clean_p_value": clean_p,
                "attack_p_value": attack_p,
                "success": int(attack_p <= threshold),
                "clean_false_positive": int(clean_p <= threshold),
                "eligible_success": int(clean_p > threshold and attack_p <= threshold),
                "neg_log10_p": -math.log10(max(attack_p, 1e-300)),
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_value,
                "requested_iterations": entry.get("requested_iterations"),
                "executed_iterations": entry.get("executed_iterations"),
                "first_success_step": entry.get("first_success_step"),
                "first_success_p_value": entry.get("first_success_p_value"),
                "runtime_seconds": entry.get("runtime_seconds"),
                "peak_memory_mb": entry.get("peak_memory_mb"),
                "output": entry["output"],
            }
        )

    expected_entries = int(manifest["pair_count"]) * len(METHODS)
    if len(rows) != expected_entries:
        raise ValueError(f"Expected {expected_entries} entries, found {len(rows)}")
    metrics_path = run_dir / "metrics.csv"
    write_csv(metrics_path, rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    summary: dict[str, Any] = {}
    for method in METHODS:
        method_rows = grouped[method]
        eligible = [row for row in method_rows if not row["clean_false_positive"]]
        successful = [row for row in eligible if row["eligible_success"]]
        first_steps = [float(row["first_success_step"]) for row in successful]
        summary[method] = {
            "n": len(method_rows),
            "asr": sum(int(row["success"]) for row in method_rows) / len(method_rows),
            "eligible_n": len(eligible),
            "eligible_asr": (
                sum(int(row["eligible_success"]) for row in eligible) / len(eligible)
                if eligible
                else None
            ),
            "clean_false_positives": sum(
                int(row["clean_false_positive"]) for row in method_rows
            ),
            "mean_p_value": mean_or_none(
                [float(row["attack_p_value"]) for row in method_rows]
            ),
            "median_p_value": median_or_none(
                [float(row["attack_p_value"]) for row in method_rows]
            ),
            "mean_psnr_all": mean_or_none([float(row["psnr"]) for row in method_rows]),
            "mean_ssim_all": mean_or_none([float(row["ssim"]) for row in method_rows]),
            "mean_lpips_all": mean_or_none(
                [float(row["lpips"]) for row in method_rows if row["lpips"] is not None]
            ),
            "mean_psnr_successful": mean_or_none(
                [float(row["psnr"]) for row in successful]
            ),
            "mean_ssim_successful": mean_or_none(
                [float(row["ssim"]) for row in successful]
            ),
            "mean_lpips_successful": mean_or_none(
                [float(row["lpips"]) for row in successful if row["lpips"] is not None]
            ),
            "mean_first_success_step": mean_or_none(first_steps),
            "median_first_success_step": median_or_none(first_steps),
            "mean_executed_iterations": mean_or_none(
                [float(row["executed_iterations"]) for row in method_rows]
            ),
            "mean_runtime_seconds": mean_or_none(
                [float(row["runtime_seconds"]) for row in method_rows]
            ),
            "failed_at_max_iterations": sum(
                row["first_success_step"] is None for row in eligible
            ),
        }

    by_pair: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[int(row["pair_index"])][row["method"]] = row
    paired_rows: list[dict[str, Any]] = []
    for pair_index in sorted(by_pair):
        pair = by_pair[pair_index]
        baseline = pair["baseline"]
        proposed = pair["simple_average"]
        both_success = bool(
            baseline["eligible_success"] and proposed["eligible_success"]
        )
        paired_rows.append(
            {
                "pair_index": pair_index,
                "w_seed": baseline["w_seed"],
                "sample_id": baseline["sample_id"],
                "baseline_success": baseline["eligible_success"],
                "simple_average_success": proposed["eligible_success"],
                "both_success": int(both_success),
                "baseline_first_success_step": baseline["first_success_step"],
                "simple_average_first_success_step": proposed["first_success_step"],
                "step_delta_simple_minus_baseline": (
                    int(proposed["first_success_step"])
                    - int(baseline["first_success_step"])
                    if both_success
                    else None
                ),
                "psnr_delta_simple_minus_baseline": float(proposed["psnr"])
                - float(baseline["psnr"]),
                "ssim_delta_simple_minus_baseline": float(proposed["ssim"])
                - float(baseline["ssim"]),
                "lpips_delta_simple_minus_baseline": (
                    float(proposed["lpips"]) - float(baseline["lpips"])
                    if proposed["lpips"] is not None and baseline["lpips"] is not None
                    else None
                ),
            }
        )
    paired_path = run_dir / "paired_metrics.csv"
    write_csv(paired_path, paired_rows)
    both_success_rows = [row for row in paired_rows if row["both_success"]]
    summary["paired"] = {
        "n_pairs": len(paired_rows),
        "both_success_n": len(both_success_rows),
        "simple_success_baseline_failure_n": sum(
            row["simple_average_success"] and not row["baseline_success"]
            for row in paired_rows
        ),
        "baseline_success_simple_failure_n": sum(
            row["baseline_success"] and not row["simple_average_success"]
            for row in paired_rows
        ),
        "mean_step_delta_on_both_success": mean_or_none(
            [float(row["step_delta_simple_minus_baseline"]) for row in both_success_rows]
        ),
        "mean_psnr_delta_all_pairs": mean_or_none(
            [float(row["psnr_delta_simple_minus_baseline"]) for row in paired_rows]
        ),
        "mean_psnr_delta_on_both_success": mean_or_none(
            [
                float(row["psnr_delta_simple_minus_baseline"])
                for row in both_success_rows
            ]
        ),
        "mean_ssim_delta_all_pairs": mean_or_none(
            [float(row["ssim_delta_simple_minus_baseline"]) for row in paired_rows]
        ),
        "mean_ssim_delta_on_both_success": mean_or_none(
            [
                float(row["ssim_delta_simple_minus_baseline"])
                for row in both_success_rows
            ]
        ),
        "mean_lpips_delta_all_pairs": mean_or_none(
            [
                float(row["lpips_delta_simple_minus_baseline"])
                for row in paired_rows
                if row["lpips_delta_simple_minus_baseline"] is not None
            ]
        ),
        "mean_lpips_delta_on_both_success": mean_or_none(
            [
                float(row["lpips_delta_simple_minus_baseline"])
                for row in both_success_rows
                if row["lpips_delta_simple_minus_baseline"] is not None
            ]
        ),
    }

    asr_rows = cumulative_asr_rows(
        rows,
        int(manifest["attack"]["iterations"]),
        int(manifest["attack"]["detection_every"]),
    )
    asr_path = run_dir / "asr_by_iteration.csv"
    write_csv(asr_path, asr_rows)
    requested_budgets = {1000, 3000, 5000, 10000, int(manifest["attack"]["iterations"])}
    summary["asr_at_iterations"] = {
        str(row["step"]): {
            "baseline_asr": row["baseline_asr"],
            "simple_average_asr": row["simple_average_asr"],
            "delta_simple_minus_baseline": row[
                "asr_delta_simple_minus_baseline"
            ],
            "baseline_eligible_asr": row["baseline_eligible_asr"],
            "simple_average_eligible_asr": row[
                "simple_average_eligible_asr"
            ],
            "eligible_delta_simple_minus_baseline": row[
                "eligible_asr_delta_simple_minus_baseline"
            ],
        }
        for row in asr_rows
        if int(row["step"]) in requested_budgets
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    evaluation_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "attack_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": config_sha256,
        "target_model": manifest["models"]["target"],
        "inversion_steps": inversion_steps,
        "p_value_threshold": threshold,
        "lpips_enabled": use_lpips,
        "git": git_provenance(config["_project_root"]),
        "runtime": runtime_provenance(),
        "outputs": {
            path.name: sha256_file(path)
            for path in (metrics_path, paired_path, asr_path, summary_path)
        },
    }
    (run_dir / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"metrics written to {run_dir}")


if __name__ == "__main__":
    main()
