from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from rmlp.config import load_config
from rmlp.image_io import load_rgb, preprocess_pil
from rmlp.models import load_target_pipeline, seed_everything
from rmlp.removal import REMOVAL_METHODS
from rmlp.reproducibility import sha256_file
from rmlp.tree_ring import build_key_and_mask, detect_p_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation for four-method removal runs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--skip-wrong-keys", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty metrics table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _image_array(path: Path) -> np.ndarray:
    return np.asarray(load_rgb(path), dtype=np.float32) / 255.0


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _checkpoint_metrics(
    run_dir: Path,
    manifest: dict[str, Any],
    lpips_model: torch.nn.Module | None,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    size = int(config["watermark"]["img_size"])
    device = config["model"]["device"]
    for entry in manifest["entries"]:
        source_path = run_dir / entry["source_export"]
        source = _image_array(source_path)
        with (run_dir / entry["detection_history"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            detections = {
                int(record["step"]): float(record["p_value"])
                for record in csv.DictReader(handle)
            }
        for step in entry["checkpoint_steps"]:
            checkpoint = run_dir / entry["checkpoint_dir"] / f"step_{int(step):04d}.png"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            if int(step) not in detections:
                raise ValueError(
                    f"Missing p-value for step {step}: {entry['detection_history']}"
                )
            output = _image_array(checkpoint)
            diff = output - source
            psnr = float("inf") if not np.any(diff) else float(
                peak_signal_noise_ratio(source, output, data_range=1.0)
            )
            lpips_value = None
            if lpips_model is not None:
                with torch.inference_mode():
                    lpips_value = float(
                        lpips_model(
                            preprocess_pil(load_rgb(source_path), size).unsqueeze(0).to(device),
                            preprocess_pil(load_rgb(checkpoint), size).unsqueeze(0).to(device),
                        ).item()
                    )
            rows.append(
                {
                    "pair_index": entry["pair_index"],
                    "w_seed": entry["w_seed"],
                    "sample_id": entry["sample_id"],
                    "method": entry["method"],
                    "step": int(step),
                    "p_value": detections[int(step)],
                    "removed": int(detections[int(step)] >= float(config["evaluation"]["p_value_threshold"])),
                    "l2": float(np.linalg.norm(diff.reshape(-1))),
                    "linf": float(np.abs(diff).max()),
                    "psnr": psnr,
                    "ssim": float(structural_similarity(source, output, data_range=1.0, channel_axis=2)),
                    "lpips": lpips_value,
                    "checkpoint": str(checkpoint.relative_to(run_dir)),
                }
            )
    return rows


def _summarize_checkpoint_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["step"]))].append(row)
    summary: list[dict[str, Any]] = []
    for method in REMOVAL_METHODS:
        for step in sorted({key[1] for key in grouped if key[0] == method}):
            records = grouped[(method, step)]
            item: dict[str, Any] = {"method": method, "step": step, "n": len(records)}
            for metric in ("p_value", "l2", "linf", "psnr", "ssim", "lpips"):
                values = [float(row[metric]) for row in records if row[metric] is not None]
                finite = [value for value in values if np.isfinite(value)]
                item[f"mean_{metric}"] = float(np.mean(values)) if values else None
                item[f"std_{metric}"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else (0.0 if finite else None)
            item["asr"] = float(np.mean([row["removed"] for row in records]))
            summary.append(item)
    return summary


def _plot_checkpoint_trends(
    run_dir: Path, summary: list[dict[str, Any]], threshold: float
) -> None:
    plot_dir = run_dir / "trend_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(REMOVAL_METHODS, ("#4472C4", "#ED7D31", "#70AD47", "#A64D79")))

    fig, axis = plt.subplots(figsize=(8, 5))
    for method in REMOVAL_METHODS:
        records = [row for row in summary if row["method"] == method]
        axis.plot([row["step"] for row in records], [row["mean_p_value"] for row in records], marker="o", label=method, color=colors[method])
    axis.axhline(threshold, color="black", linestyle="--", linewidth=1, label=f"threshold={threshold}")
    axis.set(xlabel="Iteration", ylabel="Mean target-key p-value", title="Watermark-removal p-value trend")
    axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
    fig.savefig(plot_dir / "p_value_vs_iteration.png", dpi=200); plt.close(fig)

    metrics = (("l2", "Mean L2"), ("linf", "Mean Linf"), ("psnr", "Mean PSNR (dB)"), ("ssim", "Mean SSIM"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        for method in REMOVAL_METHODS:
            records = [row for row in summary if row["method"] == method]
            values = [row[f"mean_{metric}"] for row in records]
            axis.plot([row["step"] for row in records], values, marker="o", label=method, color=colors[method])
        axis.set(xlabel="Iteration", ylabel=label); axis.grid(alpha=0.25)
    axes[0, 0].legend(); fig.suptitle("Image-quality and perturbation trends"); fig.tight_layout()
    fig.savefig(plot_dir / "quality_vs_iteration.png", dpi=200); plt.close(fig)

    if any(row["mean_lpips"] is not None for row in summary):
        fig, axis = plt.subplots(figsize=(8, 5))
        for method in REMOVAL_METHODS:
            records = [row for row in summary if row["method"] == method]
            axis.plot([row["step"] for row in records], [row["mean_lpips"] for row in records], marker="o", label=method, color=colors[method])
        axis.set(xlabel="Iteration", ylabel="Mean LPIPS", title="Perceptual-quality trend")
        axis.grid(alpha=0.25); axis.legend(); fig.tight_layout()
        fig.savefig(plot_dir / "lpips_vs_iteration.png", dpi=200); plt.close(fig)


def main() -> None:
    args = parse_args(); config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("methods") != list(REMOVAL_METHODS):
        raise ValueError("The run manifest does not contain the four removal methods")
    if manifest.get("config", {}).get("sha256") != sha256_file(config["_config_path"]):
        raise ValueError("Evaluation config differs from the attack manifest")
    threshold, size = float(config["evaluation"]["p_value_threshold"]), int(config["watermark"]["img_size"])
    inversion_steps = int(config["evaluation"]["inversion_steps"])
    pipe = load_target_pipeline(config)
    lpips_model = None
    if bool(config["evaluation"].get("compute_lpips", True)) and not args.no_lpips:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").to(config["model"]["device"]).eval()
    keys = [int(seed) for seed in config["multikey"]["key_seeds"]]
    detector_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    def detector(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seed not in detector_cache:
            keyed = deepcopy(config); keyed["watermark"]["w_seed"] = seed
            detector_cache[seed] = build_key_and_mask(pipe, keyed)
        return detector_cache[seed]
    rows: list[dict[str, Any]] = []
    for entry in tqdm(manifest["entries"], desc="offline removal evaluation"):
        w_seed = int(entry["w_seed"]); key, mask = detector(w_seed)
        source_path, output_path = run_dir / entry["source_export"], run_dir / entry["output"]
        source_p = detect_p_value(load_rgb(source_path), pipe, key, mask, size, inversion_steps)
        output_p = detect_p_value(load_rgb(output_path), pipe, key, mask, size, inversion_steps)
        source, output = _image_array(source_path), _image_array(output_path)
        diff = output - source
        lpips_value = None
        if lpips_model is not None:
            with torch.inference_mode():
                lpips_value = float(lpips_model(preprocess_pil(load_rgb(source_path), size).unsqueeze(0).to(config["model"]["device"]), preprocess_pil(load_rgb(output_path), size).unsqueeze(0).to(config["model"]["device"])).item())
        wrong_accepts = 0; wrong_checked = 0
        if not args.skip_wrong_keys:
            for wrong_seed in keys:
                if wrong_seed == w_seed: continue
                wrong_key, wrong_mask = detector(wrong_seed)
                wrong_p = detect_p_value(load_rgb(output_path), pipe, wrong_key, wrong_mask, size, inversion_steps)
                wrong_accepts += int(wrong_p <= threshold); wrong_checked += 1
        source_accepted, output_rejected = source_p <= threshold, output_p >= threshold
        rows.append({"pair_index": entry["pair_index"], "w_seed": w_seed, "sample_id": entry["sample_id"], "method": entry["method"], "source_target_p_value": source_p, "output_target_p_value": output_p, "source_accepted": int(source_accepted), "output_rejected": int(output_rejected), "success": int(source_accepted and output_rejected), "neg_log10_output_p": -math.log10(max(output_p, 1e-300)), "l2": float(np.linalg.norm(diff.reshape(-1))), "linf": float(np.abs(diff).max()), "psnr": float(peak_signal_noise_ratio(source, output, data_range=1.0)), "ssim": float(structural_similarity(source, output, data_range=1.0, channel_axis=2)), "lpips": lpips_value, "wrong_key_checked": wrong_checked, "wrong_key_accepts": wrong_accepts, "wrong_key_accept_rate": wrong_accepts / wrong_checked if wrong_checked else None, "executed_iterations": entry.get("executed_iterations"), "first_success_step": entry.get("first_success_step"), "runtime_seconds": entry.get("runtime_seconds"), "output": entry["output"]})
    expected = int(manifest["pair_count"]) * len(REMOVAL_METHODS)
    if len(rows) != expected: raise ValueError(f"Expected {expected} results, found {len(rows)}")
    metrics_path = run_dir / "removal_metrics.csv"; _write_csv(metrics_path, rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[row["method"]].append(row)
    summary: dict[str, Any] = {"protocol": {"success": "source accepted by target key and output rejected by target key", "threshold": threshold, "wrong_key_checks_enabled": not args.skip_wrong_keys}}
    for method in REMOVAL_METHODS:
        method_rows = grouped[method]; successful = [row for row in method_rows if row["success"]]
        summary[method] = {"n": len(method_rows), "asr": sum(row["success"] for row in method_rows) / len(method_rows), "mean_l2": _mean([row["l2"] for row in method_rows]), "mean_linf": _mean([row["linf"] for row in method_rows]), "mean_psnr": _mean([row["psnr"] for row in method_rows]), "mean_ssim": _mean([row["ssim"] for row in method_rows]), "mean_lpips": _mean([row["lpips"] for row in method_rows if row["lpips"] is not None]), "mean_wrong_key_accept_rate": _mean([row["wrong_key_accept_rate"] for row in method_rows if row["wrong_key_accept_rate"] is not None]), "mean_first_success_step": _mean([float(row["first_success_step"]) for row in successful if row["first_success_step"] is not None]), "mean_runtime_seconds": _mean([float(row["runtime_seconds"]) for row in method_rows if row["runtime_seconds"] is not None])}
    summary_path = run_dir / "removal_summary.json"; summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    checkpoint_rows = _checkpoint_metrics(run_dir, manifest, lpips_model, config)
    _write_csv(run_dir / "removal_checkpoint_metrics.csv", checkpoint_rows)
    checkpoint_summary = _summarize_checkpoint_metrics(checkpoint_rows)
    _write_csv(run_dir / "removal_checkpoint_summary.csv", checkpoint_summary)
    _plot_checkpoint_trends(run_dir, checkpoint_summary, threshold)
    print(json.dumps(summary, ensure_ascii=False, indent=2)); print(f"metrics written to {run_dir}")


if __name__ == "__main__": main()
