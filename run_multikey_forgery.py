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
from rmlp.image_io import load_rgb, preprocess_pil, safe_stem, tensor_to_pil
from rmlp.models import load_proxy_vae, load_target_pipeline, seed_everything
from rmlp.multikey import METHODS, build_key_cover_pairs, key_directory_name
from rmlp.prototype import encode_vae_latent, latent_statistics, simple_average_prototype
from rmlp.reproducibility import (
    file_record,
    git_provenance,
    runtime_provenance,
    sha256_file,
)
from rmlp.tree_ring import build_key_and_mask, detect_p_value
from run_forgery import (
    load_reference_bank,
    write_detection_history,
    write_history,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired baseline and five-reference attacks across keys."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--pair-count", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--lambda-pixel", type=float, default=None)
    parser.add_argument("--detection-every", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    all_key_seeds = [int(seed) for seed in config["multikey"]["key_seeds"]]
    pair_count = args.pair_count if args.pair_count is not None else len(all_key_seeds)
    if not 1 <= pair_count <= len(all_key_seeds):
        raise ValueError("pair_count must be in [1, number of configured keys]")
    key_seeds = all_key_seeds[:pair_count]

    cover_dir = project_path(config, config["data"]["cover_dir"])
    cover_paths = list_images(cover_dir, pair_count)
    pairs = build_key_cover_pairs(key_seeds, cover_paths)
    reference_root = project_path(config, config["data"]["references_dir"])
    multikey_reference_manifest = reference_root / "multikey_manifest.json"
    if not multikey_reference_manifest.is_file():
        raise FileNotFoundError(
            f"Missing multi-key reference manifest: {multikey_reference_manifest}"
        )
    reference_root_metadata = json.loads(
        multikey_reference_manifest.read_text(encoding="utf-8")
    )
    if reference_root_metadata.get("config_sha256") != sha256_file(
        config["_config_path"]
    ):
        raise ValueError("Reference banks were prepared with a different config")
    available_keys = {
        int(record["w_seed"]) for record in reference_root_metadata.get("keys", [])
    }
    if not set(key_seeds).issubset(available_keys):
        raise ValueError("Multi-key reference manifest is missing a requested key")

    iterations = (
        args.iterations
        if args.iterations is not None
        else int(config["attack"]["num_iterations"])
    )
    lambda_pixel = (
        args.lambda_pixel
        if args.lambda_pixel is not None
        else float(config["attack"]["lambda_pixel"])
    )
    detection_every = (
        args.detection_every
        if args.detection_every is not None
        else int(config["attack"]["detection_every"])
    )
    threshold = float(config["evaluation"]["p_value_threshold"])
    inversion_steps = int(config["evaluation"]["inversion_steps"])
    if iterations <= 0 or detection_every <= 0:
        raise ValueError("iterations and detection_every must be positive")
    if lambda_pixel < 0:
        raise ValueError("lambda_pixel must be non-negative")

    output_root = project_path(config, config["data"]["output_dir"]) / "attacks"
    run_name = args.run_name or datetime.now().strftime("multikey_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    manifest_path = run_dir / "manifest.json"
    config_path = Path(config["_config_path"])
    config_sha256 = sha256_file(config_path)
    existing_manifest = None
    if manifest_path.exists():
        if not args.skip_existing:
            raise FileExistsError(
                f"Run already exists: {run_dir}; use a new name or --skip-existing"
            )
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "methods": list(METHODS),
            "pair_count": pair_count,
            "iterations": iterations,
            "lambda_pixel": lambda_pixel,
            "detection_every": detection_every,
        }
        recorded = {
            "methods": existing_manifest.get("methods"),
            "pair_count": existing_manifest.get("pair_count"),
            "iterations": existing_manifest.get("attack", {}).get("iterations"),
            "lambda_pixel": existing_manifest.get("attack", {}).get("lambda_pixel"),
            "detection_every": existing_manifest.get("attack", {}).get("detection_every"),
        }
        if recorded != expected:
            raise ValueError("Cannot resume with different paired-design settings")
        if existing_manifest.get("config", {}).get("sha256") != config_sha256:
            raise ValueError("Cannot resume with a different configuration")

    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = run_dir / "config_snapshot.yaml"
    shutil.copyfile(config_path, config_snapshot)
    reference_manifest_snapshot = run_dir / "reference_multikey_manifest_snapshot.json"
    shutil.copyfile(multikey_reference_manifest, reference_manifest_snapshot)
    cover_manifest_path = run_dir / "cover_manifest.json"
    cover_manifest = {
        "dataset_name": config["data"].get("dataset_name", "unspecified"),
        "cover_root": str(cover_dir.resolve()),
        "selection": "first N images in recursive deterministic directory/name order",
        "pairing": "key_seeds[i] is paired only with covers[i]",
        "count": pair_count,
        "files": [file_record(path, cover_dir) for path in cover_paths],
    }
    cover_manifest_path.write_text(
        json.dumps(cover_manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if existing_manifest is None:
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "design": "one-to-one key-cover pairing",
            "pair_count": pair_count,
            "total_attacks": pair_count * len(METHODS),
            "methods": list(METHODS),
            "config": {
                "snapshot": "config_snapshot.yaml",
                "sha256": config_sha256,
            },
            "models": {
                "target": {
                    "id": config["model"]["target_model_id"],
                    "revision": config["model"].get("target_model_revision"),
                    "variant": config["model"].get("target_model_variant"),
                },
                "proxy_vae": {
                    "id": config["model"]["proxy_vae_model_id"],
                    "revision": config["model"].get("proxy_vae_revision"),
                    "variant": config["model"].get("proxy_vae_variant"),
                },
                "dtype": config["model"]["dtype"],
                "device": config["model"]["device"],
            },
            "prototype": {
                "baseline": "latent of reference index 0",
                "simple_average": "fp32 mean of all five reference latents",
                "reference_count": int(config["prototype"]["reference_count"]),
                "reference_filtering_applied": False,
            },
            "attack": {
                "lambda_pixel": lambda_pixel,
                "alpha": float(config["attack"]["alpha"]),
                "iterations": iterations,
                "detection_every": detection_every,
                "detection_threshold": threshold,
                "early_stop_on_success": True,
            },
            "reference_banks": {
                "snapshot": reference_manifest_snapshot.name,
                "sha256": sha256_file(multikey_reference_manifest),
            },
            "cover_manifest": {
                "path": cover_manifest_path.name,
                "sha256": sha256_file(cover_manifest_path),
            },
            "pairings": [
                {
                    "pair_index": index,
                    "w_seed": w_seed,
                    "cover": file_record(cover_path, cover_dir),
                }
                for index, (w_seed, cover_path) in enumerate(pairs)
            ],
            "git": git_provenance(config["_project_root"]),
            "runtime": runtime_provenance(),
            "commands": [list(sys.argv)],
            "entries": [],
        }
    else:
        manifest = existing_manifest
        manifest.setdefault("commands", []).append(list(sys.argv))
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()

    existing_entries = {
        (int(entry["w_seed"]), entry["method"]): entry
        for entry in manifest.get("entries", [])
    }

    print("loading shared SD1.4 proxy VAE and SD2-base target detector", flush=True)
    proxy_vae = load_proxy_vae(config)
    detector_pipe = load_target_pipeline(config)
    img_size = int(config["watermark"]["img_size"])
    n_refs = int(config["prototype"]["reference_count"])

    for pair_index, (w_seed, cover_path) in enumerate(
        tqdm(pairs, desc="key-cover pairs")
    ):
        key_config = deepcopy(config)
        key_config["watermark"]["w_seed"] = w_seed
        reference_dir = reference_root / key_directory_name(w_seed)
        reference_paths, metadata_path, metadata = load_reference_bank(
            reference_dir, n_refs, key_config
        )
        reference_latents = []
        for path in reference_paths:
            reference_tensor = preprocess_pil(load_rgb(path), img_size)
            reference_latents.append(encode_vae_latent(proxy_vae, reference_tensor)[0])
        stacked = torch.stack(reference_latents, dim=0)
        targets = {
            "baseline": stacked[0].float().unsqueeze(0),
            "simple_average": simple_average_prototype(stacked).unsqueeze(0),
        }
        detector_key, detector_mask = build_key_and_mask(detector_pipe, key_config)

        sample_id = f"pair_{pair_index:02d}_key_{w_seed:03d}_{safe_stem(cover_path)}"
        cover_tensor = preprocess_pil(load_rgb(cover_path), img_size).unsqueeze(0)
        clean_export = run_dir / "covers" / f"{sample_id}.png"
        clean_export.parent.mkdir(parents=True, exist_ok=True)
        tensor_to_pil(cover_tensor).save(clean_export)
        clean_p_value = detect_p_value(
            tensor_to_pil(cover_tensor),
            detector_pipe,
            detector_key,
            detector_mask,
            img_size,
            inversion_steps,
        )

        key_artifact_dir = run_dir / "keys" / key_directory_name(w_seed)
        key_artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "w_seed": w_seed,
                "reference_latents": stacked.cpu(),
                "baseline_target": targets["baseline"].cpu(),
                "simple_average_target": targets["simple_average"].cpu(),
            },
            key_artifact_dir / "prototype_artifacts.pt",
        )
        diagnostics = {
            "w_seed": w_seed,
            "reference_paths": [str(path) for path in reference_paths],
            "reference_metadata_sha256": sha256_file(metadata_path),
            "reference_key_sha256": metadata.get("key_sha256"),
            "reference_filtering_applied": False,
            "baseline_reference_index": 0,
            "simple_average_reference_indices": list(range(n_refs)),
            "reference_latent_statistics": [
                {"index": index, **latent_statistics(stacked[index])}
                for index in range(n_refs)
            ],
            "baseline_target_statistics": latent_statistics(targets["baseline"]),
            "simple_average_target_statistics": latent_statistics(
                targets["simple_average"]
            ),
        }
        (key_artifact_dir / "prototype_diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

        for method in METHODS:
            entry_key = (w_seed, method)
            output_path = run_dir / method / f"{sample_id}.png"
            if args.skip_existing and output_path.is_file() and entry_key in existing_entries:
                print(f"skip existing key={w_seed} method={method}", flush=True)
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)

            def report_progress(record, *, _method=method, _sample=sample_id):
                print(
                    f"progress sample={_sample} method={_method} "
                    f"step={record['step']}/{iterations} "
                    f"latent={record['latent_loss']:.6g} "
                    f"pixel={record['pixel_loss']:.6g} "
                    f"total={record['total_loss']:.6g}",
                    flush=True,
                )

            def periodic_detection(
                step: int,
                tensor: torch.Tensor,
                *,
                _method=method,
                _sample=sample_id,
            ) -> float:
                p_value = detect_p_value(
                    tensor_to_pil(tensor),
                    detector_pipe,
                    detector_key,
                    detector_mask,
                    img_size,
                    inversion_steps,
                )
                print(
                    f"detect sample={_sample} method={_method} "
                    f"step={step}/{iterations} p_value={p_value:.8g} "
                    f"success={p_value <= threshold}",
                    flush=True,
                )
                return p_value

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            print(
                f"start pair={pair_index} key={w_seed} sample={sample_id} "
                f"method={method} max_iterations={iterations}",
                flush=True,
            )
            started = time.perf_counter()
            result = optimize_to_target_latent(
                cover=cover_tensor,
                target_latent=targets[method],
                vae=proxy_vae,
                lambda_pixel=lambda_pixel,
                alpha=float(config["attack"]["alpha"]),
                num_iterations=iterations,
                log_every=int(config["attack"]["log_every"]),
                progress_callback=report_progress,
                detection_every=detection_every,
                detection_callback=periodic_detection,
                detection_threshold=threshold,
                stop_on_detection=True,
            )
            runtime_seconds = time.perf_counter() - started
            peak_memory_mb = (
                torch.cuda.max_memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else None
            )
            tensor_to_pil(result.adversarial).save(output_path)
            history_path = run_dir / "logs" / method / f"{sample_id}.csv"
            detection_path = run_dir / "detections" / method / f"{sample_id}.csv"
            write_history(history_path, result.history)
            write_detection_history(detection_path, result.detection_history)
            entry = {
                "pair_index": pair_index,
                "w_seed": w_seed,
                "sample_id": sample_id,
                "method": method,
                "source_path": str(cover_path),
                "clean_export": str(clean_export.relative_to(run_dir)),
                "clean_p_value_during_attack": clean_p_value,
                "output": str(output_path.relative_to(run_dir)),
                "history": str(history_path.relative_to(run_dir)),
                "detection_history": str(detection_path.relative_to(run_dir)),
                "reference_metadata": str(metadata_path),
                "reference_key_sha256": metadata.get("key_sha256"),
                "requested_iterations": iterations,
                "executed_iterations": result.executed_iterations,
                "early_stopped": result.executed_iterations < iterations,
                "first_success_step": result.first_success_step,
                "first_success_p_value": result.first_success_p_value,
                "runtime_seconds": runtime_seconds,
                "peak_memory_mb": peak_memory_mb,
            }
            if entry_key in existing_entries:
                manifest["entries"] = [
                    entry
                    if (int(old["w_seed"]), old["method"]) == entry_key
                    else old
                    for old in manifest["entries"]
                ]
            else:
                manifest["entries"].append(entry)
            existing_entries[entry_key] = entry
            write_manifest(manifest_path, manifest)

        del stacked, reference_latents, targets, detector_key, detector_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_manifest(manifest_path, manifest)
    print(f"paired multi-key attack ready: {run_dir}")


if __name__ == "__main__":
    main()
