from __future__ import annotations

import argparse
import csv
import gc
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
from rmlp.image_io import (
    fit_shared_latent_pca,
    latent_pca_to_pil,
    load_rgb,
    preprocess_pil,
    tensor_to_pil,
)
from rmlp.models import load_proxy_vae, load_target_pipeline, seed_everything
from rmlp.prototype import encode_vae_latent, latent_statistics, simple_average_prototype
from rmlp.reproducibility import (
    file_record,
    git_provenance,
    runtime_provenance,
    sha256_file,
)


EXPECTED_SAVE_STEPS = {500, 1000, 1500, 2000, 2500, 3000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate five detector-positive same-key Tree-Ring references, "
            "visualize proxy-VAE latents with one shared PCA-to-RGB mapping, "
            "and run one detector-free 3,000-step simple-average forgery."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        raise ValueError("Optimization history must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _validate_visualization_config(config: dict[str, Any]) -> None:
    if int(config["watermark"]["w_seed"]) != 0:
        raise ValueError("This locked visualization experiment requires w_seed=0")
    if int(config["prototype"]["reference_count"]) != 5:
        raise ValueError("This locked visualization experiment requires five references")
    if config["prototype"].get("aggregation") != "simple_average_all_references":
        raise ValueError("Only the proposed simple-average method is allowed")
    if int(config["attack"]["num_iterations"]) != 3000:
        raise ValueError("This locked visualization experiment requires 3000 iterations")
    if float(config["attack"]["lambda_pixel"]) != 10000.0:
        raise ValueError("This locked visualization experiment requires lambda_pixel=10000")
    if abs(float(config["attack"]["alpha"]) - 5.0 / 255.0) > 1e-12:
        raise ValueError("This locked visualization experiment requires alpha=5/255")
    if int(config["attack"]["log_every"]) != 500:
        raise ValueError("This locked visualization experiment requires log_every=500")
    if {int(step) for step in config["attack"].get("save_steps", [])} != EXPECTED_SAVE_STEPS:
        raise ValueError("save_steps must be exactly 500,1000,...,3000")
    if bool(config["attack"].get("early_stop_on_success", False)):
        raise ValueError("Online early stopping is forbidden in this experiment")
    if config["attack"].get("detection_every") is not None:
        raise ValueError("The attack stage must not configure periodic detection")
    if int(config["watermark"]["generation_steps"]) != 50:
        raise ValueError("This locked visualization experiment requires 50 generation steps")
    if int(config["evaluation"]["inversion_steps"]) != 50:
        raise ValueError("This locked visualization experiment requires 50 inversion steps")
    if not bool(config["reference_selection"].get("require_detected", False)):
        raise ValueError("All five generated references must pass detection")
    if int(config["visualization"]["cover_position_1based"]) != 1314:
        raise ValueError("This locked visualization experiment requires COCO position 1314")


def _projection_json(
    projection: dict[str, torch.Tensor | float | int],
) -> dict[str, Any]:
    return {
        "method": "shared PCA projection from four proxy-VAE channels to RGB",
        "interpretation": (
            "Descriptive and non-invertible visualization; exact latent values "
            "are stored in latent_tensors/*.pt."
        ),
        "input_channels": int(projection["input_channels"]),
        "output_channels": int(projection["output_channels"]),
        "lower_quantile": float(projection["lower_quantile"]),
        "upper_quantile": float(projection["upper_quantile"]),
        "explained_variance_ratio": [
            float(value)
            for value in torch.as_tensor(projection["explained_variance_ratio"])
        ],
        "channel_mean": [
            float(value) for value in torch.as_tensor(projection["channel_mean"])
        ],
        "components": torch.as_tensor(projection["components"]).tolist(),
        "lower": torch.as_tensor(projection["lower"]).tolist(),
        "upper": torch.as_tensor(projection["upper"]).tolist(),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _validate_visualization_config(config)
    seed_everything(int(config["experiment"]["seed"]))

    output_root = project_path(config, config["data"]["output_dir"])
    run_name = args.run_name or datetime.now().strftime("forgery_visualization_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    config_path = Path(config["_config_path"])
    config_snapshot = run_dir / "config_snapshot.yaml"
    shutil.copyfile(config_path, config_snapshot)
    started_at = datetime.now(timezone.utc).isoformat()
    phase_times: dict[str, float] = {}
    partial_manifest_path = run_dir / "manifest.partial.json"
    _write_json(
        partial_manifest_path,
        {
            "status": "running_or_incomplete",
            "run_id": run_name,
            "started_at": started_at,
            "command": list(sys.argv),
            "config": file_record(config_snapshot, run_dir),
            "note": (
                "This file remains if the run fails. Inspect the terminal log and "
                "any phase-specific metadata before retrying with a new run name."
            ),
        },
    )

    # Select the human-numbered 1,314th COCO image (Python index 1,313) using
    # the project's deterministic recursive directory/name ordering.
    cover_root = project_path(config, config["data"]["cover_dir"])
    cover_position = int(config["visualization"]["cover_position_1based"])
    covers = list_images(cover_root, cover_position)
    if len(covers) != cover_position:
        raise ValueError(
            f"COCO root contains only {len(covers)} supported images; "
            f"cannot select 1-based position {cover_position}"
        )
    cover_path = covers[-1]
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    original_copy = inputs_dir / f"clean_coco_{cover_position:06d}_original{cover_path.suffix.lower()}"
    shutil.copyfile(cover_path, original_copy)
    img_size = int(config["watermark"]["img_size"])
    clean_tensor = preprocess_pil(load_rgb(cover_path), img_size).unsqueeze(0)
    attack_input_path = inputs_dir / f"clean_coco_{cover_position:06d}_attack_input.png"
    tensor_to_pil(clean_tensor).save(attack_input_path)

    # Reference preparation is the only phase that loads the target pipeline
    # and detector. Rejected candidates are recorded but their images are not
    # retained. The pipeline is destroyed before the attack begins.
    reference_dir = run_dir / "watermarked_references"
    phase_started = time.perf_counter()
    target_pipe = load_target_pipeline(config)
    reference_metadata = prepare_reference_bank(
        config,
        target_pipe,
        reference_dir,
        verify=True,
        overwrite=False,
        command=list(sys.argv),
    )
    phase_times["reference_generation_and_verification_seconds"] = (
        time.perf_counter() - phase_started
    )
    del target_pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reference_paths = [
        reference_dir / record["filename"]
        for record in reference_metadata["references"]
    ]
    if len(reference_paths) != 5 or not all(path.is_file() for path in reference_paths):
        raise RuntimeError("Expected exactly five saved detector-positive references")

    phase_started = time.perf_counter()
    proxy_vae = load_proxy_vae(config)
    reference_latents = []
    for path in reference_paths:
        reference_tensor = preprocess_pil(load_rgb(path), img_size)
        reference_latents.append(encode_vae_latent(proxy_vae, reference_tensor)[0])
    stacked_references = torch.stack(reference_latents, dim=0)
    clean_latent = encode_vae_latent(proxy_vae, clean_tensor)
    mean_watermarked_latent = simple_average_prototype(stacked_references).unsqueeze(0)
    phase_times["proxy_vae_encoding_seconds"] = time.perf_counter() - phase_started

    latent_tensor_dir = run_dir / "latent_tensors"
    latent_tensor_dir.mkdir(parents=True)
    clean_latent_path = latent_tensor_dir / "clean_latent.pt"
    references_latent_path = latent_tensor_dir / "watermarked_reference_latents.pt"
    mean_latent_path = latent_tensor_dir / "mean_watermarked_latent.pt"
    projection_path = latent_tensor_dir / "latent_projection_metadata.pt"
    torch.save(clean_latent.detach().float().cpu(), clean_latent_path)
    torch.save(stacked_references.detach().float().cpu(), references_latent_path)
    torch.save(mean_watermarked_latent.detach().float().cpu(), mean_latent_path)

    pca_inputs = [clean_latent]
    pca_inputs.extend(stacked_references[index].unsqueeze(0) for index in range(5))
    pca_inputs.append(mean_watermarked_latent)
    projection = fit_shared_latent_pca(
        pca_inputs,
        lower_quantile=float(config["visualization"]["pca_lower_quantile"]),
        upper_quantile=float(config["visualization"]["pca_upper_quantile"]),
    )
    torch.save(projection, projection_path)
    projection_json_path = latent_tensor_dir / "latent_projection_metadata.json"
    _write_json(projection_json_path, _projection_json(projection))

    latent_visual_dir = run_dir / "latent_visualizations"
    latent_visual_dir.mkdir(parents=True)
    output_size = int(config["visualization"]["latent_projection_size"])
    clean_projection_path = latent_visual_dir / "clean_latent_projection.png"
    latent_pca_to_pil(clean_latent, projection, output_size).save(clean_projection_path)
    reference_projection_paths = []
    for index in range(5):
        path = latent_visual_dir / f"watermarked_ref_{index:02d}_latent_projection.png"
        latent_pca_to_pil(
            stacked_references[index].unsqueeze(0), projection, output_size
        ).save(path)
        reference_projection_paths.append(path)
    mean_projection_path = latent_visual_dir / "mean_watermarked_latent_projection.png"
    latent_pca_to_pil(mean_watermarked_latent, projection, output_size).save(
        mean_projection_path
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_paths: dict[int, Path] = {}
    checkpoint_io_seconds = 0.0

    def save_checkpoint(step: int, tensor: torch.Tensor) -> None:
        nonlocal checkpoint_io_seconds
        io_started = time.perf_counter()
        path = checkpoint_dir / f"forged_step_{step:06d}.png"
        tensor_to_pil(tensor).save(path)
        checkpoint_paths[step] = path
        checkpoint_io_seconds += time.perf_counter() - io_started

    progress_rows: list[dict[str, float | int]] = []
    attack_started = time.perf_counter()

    def record_progress(record: dict[str, float | int]) -> None:
        step = int(record["step"])
        if step % int(config["attack"]["log_every"]) != 0:
            return
        progress_rows.append(
            {
                **record,
                "elapsed_wall_seconds": time.perf_counter() - attack_started,
            }
        )
        print(
            f"progress step={step}/3000 latent={record['latent_loss']:.6g} "
            f"pixel={record['pixel_loss']:.6g} total={record['total_loss']:.6g}",
            flush=True,
        )

    # Detector-free attack: only the proxy VAE, the clean image, and the mean
    # reference latent are present in this optimization call.
    result = optimize_to_target_latent(
        cover=clean_tensor,
        target_latent=mean_watermarked_latent,
        vae=proxy_vae,
        lambda_pixel=float(config["attack"]["lambda_pixel"]),
        alpha=float(config["attack"]["alpha"]),
        num_iterations=int(config["attack"]["num_iterations"]),
        log_every=int(config["attack"]["log_every"]),
        snapshot_steps=EXPECTED_SAVE_STEPS,
        snapshot_callback=save_checkpoint,
        progress_callback=record_progress,
        detection_every=None,
        detection_callback=None,
        stop_on_detection=False,
    )
    attack_wall_seconds = time.perf_counter() - attack_started
    if result.executed_iterations != 3000 or result.detection_history:
        raise RuntimeError("Detector-free fixed-budget attack contract was violated")
    if set(checkpoint_paths) != EXPECTED_SAVE_STEPS:
        raise RuntimeError("Not all requested 500-step checkpoints were saved")

    history_path = run_dir / "optimization_history.csv"
    _write_history(history_path, progress_rows)
    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True)
    final_path = final_dir / "forged_final_step_003000.png"
    tensor_to_pil(result.adversarial).save(final_path)

    manifest = {
        "status": "complete",
        "exit_code": 0,
        "run_id": run_name,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "design": "single proposed-method forgery visualization; no baseline",
        "attack_detector_queries": 0,
        "reference_preparation_uses_detector": True,
        "models": {
            "target_reference_generator_and_verifier": {
                "id": config["model"]["target_model_id"],
                "revision": config["model"].get("target_model_revision"),
                "variant": config["model"].get("target_model_variant"),
            },
            "proxy_vae_for_attack": {
                "id": config["model"]["proxy_vae_model_id"],
                "revision": config["model"].get("proxy_vae_revision"),
                "variant": config["model"].get("proxy_vae_variant"),
            },
        },
        "watermark": {
            "method": "Tree-Ring",
            "w_seed": 0,
            "accepted_reference_count": 5,
            "reference_metadata": str((reference_dir / "metadata.json").relative_to(run_dir)),
        },
        "cover": {
            "position_1based": cover_position,
            "python_index": cover_position - 1,
            "source": file_record(cover_path, cover_root),
            "original_copy": file_record(original_copy, run_dir),
            "attack_input": file_record(attack_input_path, run_dir),
        },
        "method": {
            "name": "simple_average",
            "reference_count": 5,
            "aggregation": "FP32 arithmetic mean of all five reference latents",
            "reference_filtering_after_acceptance": False,
        },
        "attack": {
            "iterations": 3000,
            "lambda_pixel": float(config["attack"]["lambda_pixel"]),
            "alpha": float(config["attack"]["alpha"]),
            "log_every": 500,
            "save_steps": sorted(EXPECTED_SAVE_STEPS),
            "early_stop": False,
            "online_detection": False,
            "attack_wall_seconds": attack_wall_seconds,
            "checkpoint_io_seconds": checkpoint_io_seconds,
            "attack_compute_estimate_seconds": attack_wall_seconds - checkpoint_io_seconds,
        },
        "latent_statistics": {
            "clean": latent_statistics(clean_latent),
            "references": [
                latent_statistics(stacked_references[index]) for index in range(5)
            ],
            "mean_watermarked": latent_statistics(mean_watermarked_latent),
        },
        "latent_projection": {
            **_projection_json(projection),
            "metadata_pt": file_record(projection_path, run_dir),
            "metadata_json": file_record(projection_json_path, run_dir),
        },
        "phase_times": phase_times,
        "artifacts": {
            "accepted_references": [file_record(path, run_dir) for path in reference_paths],
            "latent_tensors": [
                file_record(clean_latent_path, run_dir),
                file_record(references_latent_path, run_dir),
                file_record(mean_latent_path, run_dir),
            ],
            "latent_visualizations": [
                file_record(clean_projection_path, run_dir),
                *[file_record(path, run_dir) for path in reference_projection_paths],
                file_record(mean_projection_path, run_dir),
            ],
            "checkpoints": [
                {"step": step, **file_record(checkpoint_paths[step], run_dir)}
                for step in sorted(checkpoint_paths)
            ],
            "final": file_record(final_path, run_dir),
            "optimization_history": file_record(history_path, run_dir),
        },
        "config": {
            "snapshot": file_record(config_snapshot, run_dir),
            "source_sha256": sha256_file(config_path),
        },
        "git": git_provenance(config["_project_root"]),
        "runtime": runtime_provenance(),
    }
    _write_json(run_dir / "manifest.json", manifest)
    partial_manifest_path.unlink()
    print(json.dumps({"run_dir": str(run_dir), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
