from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from prepare_references import prepare_reference_bank
from rmlp.attack import optimize_to_target_latent
from rmlp.config import load_config, project_path
from rmlp.data import list_images
from rmlp.image_io import fit_shared_latent_pca, latent_pca_to_pil, load_rgb, preprocess_pil, tensor_to_pil
from rmlp.models import load_proxy_vae, load_target_pipeline, seed_everything
from rmlp.prototype import encode_vae_latent, latent_statistics, simple_average_prototype
from rmlp.removal import mean_shift_target
from rmlp.reproducibility import file_record, git_provenance, runtime_provenance, sha256_file
from rmlp.tree_ring import build_key_and_mask, detect_p_value
from run_forgery_visualization import _projection_json, _write_json


EXPECTED_STEPS = {500, 1000, 1500, 2000, 2500, 3000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one fixed-budget five-reference Tree-Ring mean-shift removal visualization."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def _validate_visualization_config(config: dict[str, Any]) -> None:
    attack, removal = config["attack"], config["removal"]
    if int(config["watermark"]["w_seed"]) != 52:
        raise ValueError("This experiment requires w_seed=52")
    if int(config["prototype"]["reference_count"]) != 6:
        raise ValueError("Generate exactly six accepted images: five references and one held-out target")
    if int(removal["reference_count_for_mean"]) != 5 or int(removal["heldout_target_index"]) != 5:
        raise ValueError("The first five accepted images must form the mean and index 5 must be held out")
    if bool(removal.get("target_in_reference_aggregate", True)):
        raise ValueError("The removal target must be excluded from the reference aggregate")
    if removal.get("method") != "mean_shift" or float(removal["beta"]) != 1.0:
        raise ValueError("Only mean_shift with beta=1.0 is allowed")
    clean = config["clean_prior"]
    if int(clean["start_position_1based"]) != 1314 or int(clean["count"]) != 5:
        raise ValueError("Clean priors must be deterministic COCO positions 1314--1318")
    if int(attack["num_iterations"]) != 3000 or int(attack["log_every"]) != 500:
        raise ValueError("The fixed budget is 3000 steps with 500-step logging")
    if int(attack["detection_every"]) != 500 or {int(x) for x in attack["save_steps"]} != EXPECTED_STEPS:
        raise ValueError("Detection and checkpoints must occur exactly every 500 steps")
    if bool(attack.get("early_stop_on_success", False)):
        raise ValueError("Detector-guided early stopping is forbidden")
    if float(attack["lambda_pixel"]) != 10000.0 or abs(float(attack["alpha"]) - 5 / 255) > 1e-12:
        raise ValueError("This experiment requires lambda=10000 and alpha=5/255")
    if int(config["watermark"]["generation_steps"]) != 50 or int(config["evaluation"]["inversion_steps"]) != 50:
        raise ValueError("Generation and inversion both require 50 steps")
    if not bool(config["reference_selection"].get("require_detected", False)):
        raise ValueError("All six generated images must satisfy the configured detector threshold")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_clean_prior_paths(paths: list[Path], start_position_1based: int, count: int) -> list[Path]:
    start = start_position_1based - 1
    selected = paths[start : start + count]
    if len(selected) != count:
        raise ValueError(f"Need COCO positions {start_position_1based}--{start_position_1based + count - 1}")
    return selected


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _validate_visualization_config(config)
    seed_everything(int(config["experiment"]["seed"]))
    run_name = args.run_name or datetime.now().strftime("removal_visualization_%Y%m%d_%H%M%S")
    run_dir = project_path(config, config["data"]["output_dir"]) / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    config_path = Path(config["_config_path"])
    config_snapshot = run_dir / "config_snapshot.yaml"
    shutil.copyfile(config_path, config_snapshot)
    started_at = datetime.now(timezone.utc).isoformat()
    partial = run_dir / "manifest.partial.json"
    _write_json(partial, {"status": "running_or_incomplete", "run_id": run_name, "started_at": started_at, "command": list(sys.argv)})

    img_size = int(config["watermark"]["img_size"])
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir()
    cover_root = project_path(config, config["data"]["cover_dir"])
    end_position = int(config["clean_prior"]["start_position_1based"]) + int(config["clean_prior"]["count"]) - 1
    all_covers = list_images(cover_root, end_position)
    clean_paths = select_clean_prior_paths(all_covers, 1314, 5)
    clean_tensors: list[torch.Tensor] = []
    clean_records: list[dict[str, Any]] = []
    for position, source in enumerate(clean_paths, start=1314):
        original = inputs_dir / f"clean_prior_coco_{position:06d}_original{source.suffix.lower()}"
        processed = inputs_dir / f"clean_prior_coco_{position:06d}_attack_input.png"
        shutil.copyfile(source, original)
        tensor = preprocess_pil(load_rgb(source), img_size)
        tensor_to_pil(tensor).save(processed)
        clean_tensors.append(tensor)
        clean_records.append({"position_1based": position, "python_index": position - 1, "source": file_record(source, cover_root), "original_copy": file_record(original, run_dir), "attack_input": file_record(processed, run_dir)})

    phase_times: dict[str, float] = {}
    phase_start = time.perf_counter()
    target_pipe = load_target_pipeline(config)
    generated_dir = run_dir / "watermarked_generation"
    generated = prepare_reference_bank(config, target_pipe, generated_dir, verify=True, overwrite=False, command=list(sys.argv))
    phase_times["watermarked_generation_and_verification_seconds"] = time.perf_counter() - phase_start
    accepted = [generated_dir / row["filename"] for row in generated["references"]]
    if len(accepted) != 6:
        raise RuntimeError("Expected exactly six detector-positive watermarked images")
    reference_paths, heldout_path = accepted[:5], accepted[5]
    input_reference_paths: list[Path] = []
    for index, source in enumerate(reference_paths):
        destination = inputs_dir / f"watermarked_reference_{index:02d}.png"
        shutil.copyfile(source, destination)
        input_reference_paths.append(destination)
    target_input_path = inputs_dir / "heldout_watermarked_removal_target.png"
    shutil.copyfile(heldout_path, target_input_path)

    phase_start = time.perf_counter()
    proxy_vae = load_proxy_vae(config)
    reference_latents = torch.stack([encode_vae_latent(proxy_vae, preprocess_pil(load_rgb(path), img_size))[0] for path in reference_paths])
    clean_latents = torch.stack([encode_vae_latent(proxy_vae, tensor)[0] for tensor in clean_tensors])
    target_tensor = preprocess_pil(load_rgb(heldout_path), img_size).unsqueeze(0)
    target_latent = encode_vae_latent(proxy_vae, target_tensor).float()
    water_mean = simple_average_prototype(reference_latents).unsqueeze(0).float()
    clean_mean = simple_average_prototype(clean_latents).unsqueeze(0).float()
    direction = water_mean - clean_mean
    removal_target = mean_shift_target(target_latent, water_mean, clean_mean, beta=1.0)
    phase_times["proxy_vae_encoding_seconds"] = time.perf_counter() - phase_start

    tensor_dir = run_dir / "latent_tensors"
    tensor_dir.mkdir()
    tensors = {"watermarked_reference_latents": reference_latents, "heldout_target_latent": target_latent, "clean_prior_latents": clean_latents, "mean_watermarked_latent": water_mean, "mean_clean_latent": clean_mean, "estimated_watermark_direction": direction, "removal_target_latent": removal_target}
    tensor_paths: dict[str, Path] = {}
    for name, tensor in tensors.items():
        path = tensor_dir / f"{name}.pt"
        torch.save(tensor.detach().float().cpu(), path)
        tensor_paths[name] = path

    individual_latents = [reference_latents[i].unsqueeze(0) for i in range(5)] + [target_latent] + [clean_latents[i].unsqueeze(0) for i in range(5)]
    aggregate_latents = [water_mean, clean_mean, direction, removal_target]
    projection = fit_shared_latent_pca(individual_latents + aggregate_latents, float(config["visualization"]["pca_lower_quantile"]), float(config["visualization"]["pca_upper_quantile"]))
    projection_pt = tensor_dir / "latent_projection_metadata.pt"
    projection_json = tensor_dir / "latent_projection_metadata.json"
    torch.save(projection, projection_pt)
    _write_json(projection_json, _projection_json(projection))
    visual_dir = run_dir / "latent_visualizations"
    visual_dir.mkdir()
    visual_paths: list[Path] = []
    named_visuals = [(f"watermarked_reference_{i:02d}_latent_projection", reference_latents[i].unsqueeze(0)) for i in range(5)]
    named_visuals += [("heldout_target_latent_projection", target_latent)]
    named_visuals += [(f"clean_prior_{i + 1314:06d}_latent_projection", clean_latents[i].unsqueeze(0)) for i in range(5)]
    named_visuals += [("mean_watermarked_latent_projection", water_mean), ("mean_clean_latent_projection", clean_mean), ("estimated_watermark_direction_projection", direction), ("removal_target_latent_projection", removal_target)]
    for name, latent in named_visuals:
        path = visual_dir / f"{name}.png"
        latent_pca_to_pil(latent, projection, int(config["visualization"]["latent_projection_size"])).save(path)
        visual_paths.append(path)

    w_key, w_mask = build_key_and_mask(target_pipe, config)
    threshold = float(config["evaluation"]["p_value_threshold"])
    detection_seconds = 0.0
    detection_rows: list[dict[str, Any]] = []

    def monitor(step: int, tensor: torch.Tensor) -> float:
        nonlocal detection_seconds
        started = time.perf_counter()
        p_value = detect_p_value(tensor, target_pipe, w_key, w_mask, img_size, int(config["evaluation"]["inversion_steps"]))
        detection_seconds += time.perf_counter() - started
        detection_rows.append({"step": step, "p_value": p_value, "target_key_accepted": p_value <= threshold, "removal_success": p_value >= threshold})
        print(f"diagnostic_detection step={step} p_value={p_value:.8g} removal_success={p_value >= threshold}", flush=True)
        return p_value

    step0_p = monitor(0, target_tensor)
    if step0_p > threshold:
        raise RuntimeError("Held-out target no longer passes source-key detection at attack start")
    pre_attack_detection_seconds = detection_seconds
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_paths: dict[int, Path] = {}
    checkpoint_seconds = 0.0

    def save_checkpoint(step: int, tensor: torch.Tensor) -> None:
        nonlocal checkpoint_seconds
        started = time.perf_counter()
        path = checkpoint_dir / f"removed_step_{step:06d}.png"
        tensor_to_pil(tensor).save(path)
        checkpoint_paths[step] = path
        checkpoint_seconds += time.perf_counter() - started

    progress_rows: list[dict[str, Any]] = []
    attack_started = time.perf_counter()

    def progress(row: dict[str, float | int]) -> None:
        if int(row["step"]) % 500 == 0:
            progress_rows.append({**row, "elapsed_wall_seconds": time.perf_counter() - attack_started})
            print(f"progress step={row['step']}/3000 total={row['total_loss']:.6g}", flush=True)

    result = optimize_to_target_latent(cover=target_tensor, target_latent=removal_target, vae=proxy_vae, lambda_pixel=10000.0, alpha=5 / 255, num_iterations=3000, log_every=500, snapshot_steps=EXPECTED_STEPS, snapshot_callback=save_checkpoint, progress_callback=progress, detection_every=500, detection_callback=monitor, detection_threshold=threshold, stop_on_detection=False, detection_success="ge")
    attack_wall = time.perf_counter() - attack_started
    in_attack_detection_seconds = detection_seconds - pre_attack_detection_seconds
    if result.executed_iterations != 3000 or set(checkpoint_paths) != EXPECTED_STEPS or [row["step"] for row in detection_rows] != [0, 500, 1000, 1500, 2000, 2500, 3000]:
        raise RuntimeError("Fixed-budget checkpoint/detection contract was violated")
    history_path = run_dir / "optimization_history.csv"
    detection_path = run_dir / "detection_history.csv"
    _write_csv(history_path, progress_rows)
    _write_csv(detection_path, detection_rows)
    final_dir = run_dir / "final"
    final_dir.mkdir()
    final_path = final_dir / "removed_final_step_003000.png"
    tensor_to_pil(result.adversarial).save(final_path)

    manifest = {
        "status": "complete", "exit_code": 0, "run_id": run_name, "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(), "command": list(sys.argv),
        "design": "single proposed mean_shift removal visualization; no baseline", "threat_model_label": "fixed-budget attack with online diagnostic detector monitoring; not strict zero-query black box",
        "watermark": {"method": "Tree-Ring", "w_seed": 52, "threshold_semantics": "accepted when p<=0.05; removal success recorded when p>=0.05", "generated_accepted_count": 6, "reference_count_for_mean": 5, "heldout_target_index": 5, "target_in_reference_aggregate": False},
        "method": {"name": "mean_shift", "formula": "z_rm=E(x_t^w)-beta*(mean(E(x_i^w))-mean(E(x_i^c)))", "beta": 1.0, "aggregation": "FP32 arithmetic mean", "quality_loss": "lambda * pixel MSE only"},
        "attack": {"iterations": 3000, "lambda_pixel": 10000.0, "alpha": 5 / 255, "checkpoint_every": 500, "detection_every": 500, "detector_guides_optimization": False, "early_stop": False, "final_output_step": 3000, "first_observed_success_step_diagnostic_only": result.first_success_step, "attack_wall_seconds_including_monitoring_and_io": attack_wall, "step0_detection_seconds": pre_attack_detection_seconds, "in_attack_detection_seconds": in_attack_detection_seconds, "online_detection_seconds_total": detection_seconds, "checkpoint_io_seconds": checkpoint_seconds, "attack_compute_estimate_seconds": attack_wall - in_attack_detection_seconds - checkpoint_seconds},
        "models": {"target_generator_and_detector": {"id": config["model"]["target_model_id"], "revision": config["model"].get("target_model_revision")}, "proxy_vae": {"id": config["model"]["proxy_vae_model_id"], "revision": config["model"].get("proxy_vae_revision")}},
        "clean_priors": clean_records, "phase_times": phase_times,
        "latent_statistics": {name: latent_statistics(tensor) for name, tensor in tensors.items()},
        "latent_projection": {**_projection_json(projection), "meaning": "one shared descriptive 4D-to-RGB PCA mapping; no VAE decoding"},
        "artifacts": {"watermarked_references": [file_record(path, run_dir) for path in input_reference_paths], "heldout_target": file_record(target_input_path, run_dir), "generation_metadata": file_record(generated_dir / "metadata.json", run_dir), "latent_tensors": [file_record(path, run_dir) for path in tensor_paths.values()] + [file_record(projection_pt, run_dir), file_record(projection_json, run_dir)], "latent_visualizations": [file_record(path, run_dir) for path in visual_paths], "checkpoints": [{"step": step, **file_record(checkpoint_paths[step], run_dir)} for step in sorted(checkpoint_paths)], "final": file_record(final_path, run_dir), "optimization_history": file_record(history_path, run_dir), "detection_history": file_record(detection_path, run_dir)},
        "config": {"snapshot": file_record(config_snapshot, run_dir), "source_sha256": sha256_file(config_path)}, "git": git_provenance(config["_project_root"]), "runtime": runtime_provenance(),
    }
    _write_json(run_dir / "manifest.json", manifest)
    partial.unlink()
    print(json.dumps({"run_dir": str(run_dir), "status": "complete", "final_p_value": detection_rows[-1]["p_value"]}, indent=2))


if __name__ == "__main__":
    main()
