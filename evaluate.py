from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from rmlp.config import load_config
from rmlp.image_io import load_rgb, preprocess_pil
from rmlp.models import load_target_pipeline, seed_everything
from rmlp.reproducibility import (
    git_provenance,
    runtime_provenance,
    sha256_file,
)
from rmlp.tree_ring import build_key_and_mask, detect_p_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tree-Ring ASR and image quality.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--no-lpips", action="store_true")
    return parser.parse_args()


def image_array(path: Path) -> np.ndarray:
    return np.asarray(load_rgb(path), dtype=np.float32) / 255.0


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def optional_mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(config["_config_path"])
    recorded_config_sha256 = manifest.get("config", {}).get("sha256")
    if recorded_config_sha256 and recorded_config_sha256 != config_sha256:
        raise ValueError("Evaluation config does not match the attack config snapshot")
    recorded_target = manifest.get("models", {}).get("target", {})
    if recorded_target and recorded_target.get("id") != config["model"]["target_model_id"]:
        raise ValueError("Evaluation target model does not match the attack manifest")
    pipe = load_target_pipeline(config)
    w_key, w_mask = build_key_and_mask(pipe, config)
    img_size = int(config["watermark"]["img_size"])
    threshold = float(config["evaluation"]["p_value_threshold"])
    inversion_steps = int(config["evaluation"]["inversion_steps"])

    use_lpips = bool(config["evaluation"].get("compute_lpips", True)) and not args.no_lpips
    lpips_model = None
    if use_lpips:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to(config["model"]["device"]).eval()

    clean_p_cache: dict[str, float] = {}
    rows = []
    for entry in tqdm(manifest["entries"], desc="evaluate"):
        sample_id = entry["sample_id"]
        clean_path = run_dir / entry["clean_export"]
        output_path = run_dir / entry["output"]
        if sample_id not in clean_p_cache:
            clean_p_cache[sample_id] = detect_p_value(
                load_rgb(clean_path), pipe, w_key, w_mask, img_size, inversion_steps
            )
        attack_p = detect_p_value(
            load_rgb(output_path), pipe, w_key, w_mask, img_size, inversion_steps
        )
        clean = image_array(clean_path)
        attacked = image_array(output_path)
        psnr = float(peak_signal_noise_ratio(clean, attacked, data_range=1.0))
        ssim = float(structural_similarity(clean, attacked, data_range=1.0, channel_axis=2))
        lpips_value = float("nan")
        if lpips_model is not None:
            clean_tensor = preprocess_pil(load_rgb(clean_path), img_size).unsqueeze(0)
            attacked_tensor = preprocess_pil(load_rgb(output_path), img_size).unsqueeze(0)
            device = config["model"]["device"]
            with torch.inference_mode():
                lpips_value = float(lpips_model(clean_tensor.to(device), attacked_tensor.to(device)).item())
        rows.append(
            {
                "sample_id": sample_id,
                "method": entry["method"],
                "clean_p_value": clean_p_cache[sample_id],
                "attack_p_value": attack_p,
                "success": int(attack_p <= threshold),
                "clean_false_positive": int(clean_p_cache[sample_id] <= threshold),
                "neg_log10_p": -math.log10(max(attack_p, 1e-300)),
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_value,
                "runtime_seconds": entry.get("runtime_seconds"),
                "peak_memory_mb": entry.get("peak_memory_mb"),
                "output": entry["output"],
            }
        )

    csv_path = run_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    summary = {}
    for method, method_rows in grouped.items():
        p_values = [float(row["attack_p_value"]) for row in method_rows]
        lpips_values = [float(row["lpips"]) for row in method_rows if not math.isnan(float(row["lpips"]))]
        eligible_rows = [row for row in method_rows if not int(row["clean_false_positive"])]
        runtimes = [float(row["runtime_seconds"]) for row in method_rows if row["runtime_seconds"] is not None]
        peak_memory = [float(row["peak_memory_mb"]) for row in method_rows if row["peak_memory_mb"] is not None]
        summary[method] = {
            "n": len(method_rows),
            "asr": mean([float(row["success"]) for row in method_rows]),
            "eligible_n": len(eligible_rows),
            "eligible_asr": optional_mean(
                [float(row["success"]) for row in eligible_rows]
            ),
            "clean_false_positives": sum(int(row["clean_false_positive"]) for row in method_rows),
            "mean_p_value": mean(p_values),
            "median_p_value": float(np.median(p_values)),
            "mean_neg_log10_p": mean([float(row["neg_log10_p"]) for row in method_rows]),
            "mean_psnr": mean([float(row["psnr"]) for row in method_rows]),
            "mean_ssim": mean([float(row["ssim"]) for row in method_rows]),
            "mean_lpips": optional_mean(lpips_values),
            "mean_runtime_seconds": mean(runtimes),
            "mean_peak_memory_mb": mean(peak_memory),
        }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    evaluation_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "attack_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": config_sha256,
        "target_model": {
            "id": config["model"]["target_model_id"],
            "revision": config["model"].get("target_model_revision"),
            "variant": config["model"].get("target_model_variant"),
        },
        "inversion_steps": inversion_steps,
        "p_value_threshold": threshold,
        "lpips_enabled": use_lpips,
        "git": git_provenance(config["_project_root"]),
        "runtime": runtime_provenance(),
        "outputs": {
            "metrics_csv": "metrics.csv",
            "metrics_sha256": sha256_file(csv_path),
            "summary_json": "summary.json",
            "summary_sha256": sha256_file(run_dir / "summary.json"),
        },
    }
    (run_dir / "evaluation_manifest.json").write_text(
        json.dumps(
            evaluation_manifest, ensure_ascii=False, indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"metrics written to {csv_path}")


if __name__ == "__main__":
    main()
