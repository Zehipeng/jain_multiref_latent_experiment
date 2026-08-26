from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from rmlp.attack import optimize_to_target_latent
from rmlp.config import load_config, project_path
from rmlp.data import list_images
from rmlp.image_io import (
    latent_to_pil,
    load_rgb,
    preprocess_pil,
    safe_stem,
    shared_latent_scale,
    tensor_to_pil,
)
from rmlp.models import load_proxy_vae, load_target_pipeline, seed_everything
from rmlp.multikey import key_directory_name
from rmlp.prototype import encode_vae_latent, latent_statistics, simple_average_prototype
from rmlp.removal import REMOVAL_METHODS, mean_image_target, mean_shift_target, removal_mode
from rmlp.reproducibility import file_record, git_provenance, runtime_provenance, sha256_file
from rmlp.tree_ring import build_key_and_mask, detect_p_value
from run_forgery import load_reference_bank, write_detection_history, write_history, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run four paired small-scale Tree-Ring watermark-removal attacks."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--pair-count", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save smoke-test input, latent, aggregate, target, and output PNGs.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _reference_latents(vae: torch.nn.Module, paths: list[Path], image_size: int) -> torch.Tensor:
    return torch.stack(
        [encode_vae_latent(vae, preprocess_pil(load_rgb(path), image_size))[0] for path in paths],
        dim=0,
    )


def _ensure_reference_manifest(config: dict, reference_root: Path, keys: list[int]) -> Path:
    manifest_path = reference_root / "multikey_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing multi-key reference manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    # This removal protocol intentionally has its own configuration file.  The
    # per-key metadata checked by ``load_reference_bank`` verifies the relevant
    # target-model, proxy-VAE, key, checksum, and detection provenance instead
    # of requiring an unrelated full-config hash match.
    available = {int(row["w_seed"]) for row in payload.get("keys", [])}
    if not set(keys).issubset(available):
        raise ValueError("Reference manifest is missing one or more requested keys")
    return manifest_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    all_keys = [int(seed) for seed in config["multikey"]["key_seeds"]]
    pair_count = args.pair_count if args.pair_count is not None else len(all_keys)
    if not 1 <= pair_count <= len(all_keys):
        raise ValueError("pair_count must be in [1, number of configured keys]")
    keys = all_keys[:pair_count]
    iterations = args.iterations or int(config["attack"]["num_iterations"])
    detection_every = int(config["attack"]["detection_every"])
    threshold = float(config["evaluation"]["p_value_threshold"])
    lambda_pixel = float(config["attack"]["lambda_pixel"])
    alpha = float(config["attack"]["alpha"])
    beta = float(config["removal"]["beta"])
    if iterations <= 0 or detection_every <= 0 or lambda_pixel < 0 or beta < 0:
        raise ValueError("Invalid removal attack parameters")

    image_size = int(config["watermark"]["img_size"])
    n_refs = int(config["prototype"]["reference_count"])
    clean_per_key = int(config["removal"]["clean_images_per_key"])
    target_index = int(config["removal"]["target_reference_index"])
    if not 0 <= target_index < n_refs:
        raise ValueError("removal.target_reference_index is outside the reference set")
    clean_root = project_path(config, config["data"]["cover_dir"])
    clean_paths = list_images(clean_root, pair_count * clean_per_key)
    if len(clean_paths) != pair_count * clean_per_key:
        raise ValueError("Insufficient clean images for disjoint per-key prior sets")
    clean_groups = [clean_paths[i * clean_per_key : (i + 1) * clean_per_key] for i in range(pair_count)]
    reference_root = project_path(config, config["data"]["references_dir"])
    reference_manifest_path = _ensure_reference_manifest(config, reference_root, keys)

    output_root = project_path(config, config["data"]["output_dir"]) / "attacks"
    run_name = args.run_name or datetime.now().strftime("removal_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    manifest_path = run_dir / "manifest.json"
    config_sha256 = sha256_file(config["_config_path"])
    if manifest_path.exists() and not args.skip_existing:
        raise FileExistsError(f"Run exists: {run_dir}; use a new name or --skip-existing")
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if existing is not None:
        expected = {"methods": list(REMOVAL_METHODS), "pair_count": pair_count, "iterations": iterations, "detection_every": detection_every, "beta": beta, "lambda_pixel": lambda_pixel, "save_visualizations": args.save_visualizations}
        recorded = {"methods": existing.get("methods"), "pair_count": existing.get("pair_count"), "iterations": existing.get("attack", {}).get("iterations"), "detection_every": existing.get("attack", {}).get("detection_every"), "beta": existing.get("removal", {}).get("beta"), "lambda_pixel": existing.get("attack", {}).get("lambda_pixel"), "save_visualizations": existing.get("visualizations", {}).get("enabled", False)}
        if expected != recorded or existing.get("config", {}).get("sha256") != config_sha256:
            raise ValueError("Cannot resume with different removal-design settings")

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config["_config_path"], run_dir / "config_snapshot.yaml")
    shutil.copyfile(reference_manifest_path, run_dir / "reference_multikey_manifest_snapshot.json")
    clean_manifest = {
        "root": str(clean_root.resolve()), "selection": "first disjoint N-image blocks in recursive deterministic order",
        "clean_images_per_key": clean_per_key,
        "groups": [[file_record(path, clean_root) for path in group] for group in clean_groups],
    }
    clean_manifest_path = run_dir / "clean_prior_manifest.json"
    clean_manifest_path.write_text(json.dumps(clean_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = existing or {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": "paired four-method removal; reference index 0 is the attacked target and remains in the five-image aggregate",
        "pair_count": pair_count, "total_attacks": pair_count * len(REMOVAL_METHODS), "methods": list(REMOVAL_METHODS),
        "config": {"snapshot": "config_snapshot.yaml", "sha256": config_sha256},
        "models": {"target": {"id": config["model"]["target_model_id"], "revision": config["model"].get("target_model_revision"), "variant": config["model"].get("target_model_variant")}, "proxy_vae": {"id": config["model"]["proxy_vae_model_id"], "revision": config["model"].get("proxy_vae_revision")}, "dtype": config["model"]["dtype"], "device": config["model"]["device"]},
        "removal": {"beta": beta, "target_reference_index": target_index, "target_in_reference_aggregate": bool(config["removal"].get("target_in_reference_aggregate", True)), "clean_images_per_key": clean_per_key, "methods": {"jain_mean_image": "attract proxy-VAE latent of the target image's constant mean-image", "latent_repulsion": "repel the five-reference watermarked latent mean", "clean_attraction": "attract the five-image clean-prior latent mean", "mean_shift": "attract z_target - beta*(zbar_watermarked-zbar_clean)"}},
        "attack": {"lambda_pixel": lambda_pixel, "alpha": alpha, "iterations": iterations, "log_every": int(config["attack"]["log_every"]), "detection_every": detection_every, "detection_threshold": threshold, "early_stop_on_success": True, "success_rule": "target-key p_value >= threshold"},
        "visualizations": {"enabled": args.save_visualizations, "format": "four-channel blue-white-red PNG grid with one shared per-key 99.5th-percentile absolute scale; exact tensors are stored in removal_artifacts.pt"},
        "reference_banks": {"snapshot": "reference_multikey_manifest_snapshot.json", "sha256": sha256_file(reference_manifest_path)},
        "clean_prior_manifest": {"path": clean_manifest_path.name, "sha256": sha256_file(clean_manifest_path)},
        "git": git_provenance(config["_project_root"]), "runtime": runtime_provenance(), "commands": [], "entries": [],
    }
    manifest.setdefault("commands", []).append(list(sys.argv))
    existing_entries = {(int(entry["w_seed"]), entry["method"]): entry for entry in manifest["entries"]}

    print("loading shared SD1.4 proxy VAE and SD2-base online detector", flush=True)
    vae, detector_pipe = load_proxy_vae(config), load_target_pipeline(config)
    inversion_steps = int(config["evaluation"]["inversion_steps"])
    for pair_index, (w_seed, clean_group) in enumerate(tqdm(zip(keys, clean_groups), total=pair_count, desc="removal keys")):
        key_config = deepcopy(config); key_config["watermark"]["w_seed"] = w_seed
        reference_paths, metadata_path, metadata = load_reference_bank(reference_root / key_directory_name(w_seed), n_refs, key_config)
        target_path = reference_paths[target_index]
        target_tensor = preprocess_pil(load_rgb(target_path), image_size).unsqueeze(0)
        watermarked_latents = _reference_latents(vae, reference_paths, image_size)
        clean_latents = _reference_latents(vae, clean_group, image_size)
        water_mean = simple_average_prototype(watermarked_latents).unsqueeze(0)
        clean_mean = simple_average_prototype(clean_latents).unsqueeze(0)
        watermark_direction = water_mean - clean_mean
        target_latent = encode_vae_latent(vae, target_tensor).float()
        jain_target = encode_vae_latent(vae, mean_image_target(target_tensor)).float()
        targets = {"jain_mean_image": jain_target, "latent_repulsion": water_mean, "clean_attraction": clean_mean, "mean_shift": mean_shift_target(target_latent, water_mean, clean_mean, beta)}
        detector_key, detector_mask = build_key_and_mask(detector_pipe, key_config)
        source_p = detect_p_value(tensor_to_pil(target_tensor), detector_pipe, detector_key, detector_mask, image_size, inversion_steps)
        sample_id = f"pair_{pair_index:02d}_key_{w_seed:03d}_{safe_stem(target_path)}"
        source_export = run_dir / "sources" / f"{sample_id}.png"; source_export.parent.mkdir(parents=True, exist_ok=True); tensor_to_pil(target_tensor).save(source_export)
        artifact_dir = run_dir / "keys" / key_directory_name(w_seed); artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"watermarked_reference_latents": watermarked_latents.cpu(), "clean_prior_latents": clean_latents.cpu(), "target_latent": target_latent.cpu(), "watermarked_mean": water_mean.cpu(), "clean_mean": clean_mean.cpu(), "watermark_direction": watermark_direction.cpu(), "removal_targets": {method: value.cpu() for method, value in targets.items()}}, artifact_dir / "removal_artifacts.pt")
        diagnostics = {"w_seed": w_seed, "target_reference_index": target_index, "target_path": str(target_path), "target_p_value_before_attack": source_p, "watermarked_reference_paths": [str(path) for path in reference_paths], "clean_prior_paths": [str(path) for path in clean_group], "watermarked_latent_statistics": [latent_statistics(value) for value in watermarked_latents], "clean_latent_statistics": [latent_statistics(value) for value in clean_latents]}
        (artifact_dir / "removal_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        visualization_dir = run_dir / "visualizations" / key_directory_name(w_seed)
        if args.save_visualizations:
            input_dir = visualization_dir / "input_images"
            latent_dir = visualization_dir / "latent_representations"
            aggregate_dir = visualization_dir / "aggregate_latents"
            target_dir = visualization_dir / "removal_targets"
            removed_dir = visualization_dir / "removed_images"
            for directory in (input_dir, latent_dir, aggregate_dir, target_dir, removed_dir):
                directory.mkdir(parents=True, exist_ok=True)
            for index, path in enumerate(reference_paths):
                load_rgb(path).save(input_dir / f"watermarked_reference_{index:02d}.png")
            for index, path in enumerate(clean_group):
                load_rgb(path).save(input_dir / f"clean_reference_{index:02d}.png")
            tensor_to_pil(target_tensor).save(input_dir / "target_watermarked.png")
            tensor_to_pil(mean_image_target(target_tensor)).save(
                input_dir / "jain_constant_mean_guidance.png"
            )
            scale_inputs = [
                *[value for value in watermarked_latents],
                *[value for value in clean_latents],
                water_mean,
                clean_mean,
                watermark_direction,
                target_latent,
                *targets.values(),
            ]
            latent_scale = shared_latent_scale(scale_inputs)
            for index, value in enumerate(watermarked_latents):
                latent_to_pil(value, latent_scale).save(
                    latent_dir / f"watermarked_latent_{index:02d}.png"
                )
            for index, value in enumerate(clean_latents):
                latent_to_pil(value, latent_scale).save(
                    latent_dir / f"clean_latent_{index:02d}.png"
                )
            for name, value in {
                "watermarked_mean": water_mean,
                "clean_mean": clean_mean,
                "watermark_direction_difference": watermark_direction,
            }.items():
                latent_to_pil(value, latent_scale).save(
                    aggregate_dir / f"{name}.png"
                )
            for method, value in targets.items():
                latent_to_pil(value, latent_scale).save(
                    target_dir / f"{method}_target.png"
                )
            visualization_manifest = {
                "w_seed": w_seed,
                "exact_tensor_artifact": str(
                    (artifact_dir / "removal_artifacts.pt").relative_to(run_dir)
                ),
                "latent_color_mapping": "blue=negative, white=zero, red=positive",
                "shared_absolute_scale": latent_scale,
                "scale_quantile": 0.995,
                "watermarked_reference_count": len(reference_paths),
                "clean_reference_count": len(clean_group),
                "target_reference_index": target_index,
                "removal_target_semantics": {
                    "jain_mean_image": "attract to encoded constant mean-image",
                    "latent_repulsion": "repel from this latent target",
                    "clean_attraction": "attract to this latent target",
                    "mean_shift": "attract to this latent target",
                },
            }
            (visualization_dir / "visualization_manifest.json").write_text(
                json.dumps(visualization_manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        for method in REMOVAL_METHODS:
            entry_key = (w_seed, method); output_path = run_dir / method / f"{sample_id}.png"
            if args.skip_existing and output_path.is_file() and entry_key in existing_entries:
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            def progress(record, *, _method=method):
                print(f"progress key={w_seed} method={_method} step={record['step']}/{iterations} latent={record['latent_loss']:.6g} pixel={record['pixel_loss']:.6g} total={record['total_loss']:.6g}", flush=True)
            def detect(step: int, tensor: torch.Tensor, *, _method=method) -> float:
                p_value = detect_p_value(tensor_to_pil(tensor), detector_pipe, detector_key, detector_mask, image_size, inversion_steps)
                print(f"detect key={w_seed} method={_method} step={step}/{iterations} p_value={p_value:.8g} removed={p_value >= threshold}", flush=True)
                return p_value
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result = optimize_to_target_latent(cover=target_tensor, target_latent=targets[method], vae=vae, lambda_pixel=lambda_pixel, alpha=alpha, num_iterations=iterations, log_every=int(config["attack"]["log_every"]), progress_callback=progress, detection_every=detection_every, detection_callback=detect, detection_threshold=threshold, stop_on_detection=True, maximize_latent_distance=removal_mode(method) == "repel", detection_success="ge")
            runtime = time.perf_counter() - started
            removed_image = tensor_to_pil(result.adversarial)
            removed_image.save(output_path)
            visualization_output = None
            if args.save_visualizations:
                visualization_output_path = visualization_dir / "removed_images" / f"{method}.png"
                removed_image.save(visualization_output_path)
                visualization_output = str(
                    visualization_output_path.relative_to(run_dir)
                )
            history_path = run_dir / "logs" / method / f"{sample_id}.csv"; detection_path = run_dir / "detections" / method / f"{sample_id}.csv"
            write_history(history_path, result.history); write_detection_history(detection_path, result.detection_history)
            entry = {"pair_index": pair_index, "w_seed": w_seed, "sample_id": sample_id, "method": method, "source_path": str(target_path), "source_export": str(source_export.relative_to(run_dir)), "source_target_p_value": source_p, "output": str(output_path.relative_to(run_dir)), "visualization_output": visualization_output, "history": str(history_path.relative_to(run_dir)), "detection_history": str(detection_path.relative_to(run_dir)), "reference_metadata": str(metadata_path), "reference_key_sha256": metadata.get("key_sha256"), "clean_prior_paths": [str(path) for path in clean_group], "requested_iterations": iterations, "executed_iterations": result.executed_iterations, "early_stopped": result.executed_iterations < iterations, "first_success_step": result.first_success_step, "first_success_p_value": result.first_success_p_value, "runtime_seconds": runtime, "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None}
            manifest["entries"] = [entry if (int(old["w_seed"]), old["method"]) == entry_key else old for old in manifest["entries"]] if entry_key in existing_entries else manifest["entries"] + [entry]
            existing_entries[entry_key] = entry; write_manifest(manifest_path, manifest)
        del watermarked_latents, clean_latents, targets, detector_key, detector_mask
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    write_manifest(manifest_path, manifest)
    print(f"removal attack run ready: {run_dir}")


if __name__ == "__main__":
    main()
