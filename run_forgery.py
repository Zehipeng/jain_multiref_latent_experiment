from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from rmlp.attack import optimize_to_target_latent
from rmlp.config import load_config, project_path
from rmlp.data import list_images
from rmlp.image_io import load_rgb, preprocess_pil, safe_stem, tensor_to_pil
from rmlp.models import load_proxy_vae, seed_everything
from rmlp.prototype import (
    encode_vae_latent,
    latent_statistics,
    robust_latent_prototype,
    simple_average_prototype,
)
from rmlp.reproducibility import (
    file_record,
    git_provenance,
    runtime_provenance,
    sha256_file,
)


MODE_METHODS = {
    "baseline": ["baseline"],
    "simple_average": ["simple_average"],
    "full": ["full"],
    "both": ["baseline", "full"],
    "all": ["baseline", "simple_average", "full"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-reference, simple-average, and robust forgery methods."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--cover-dir", default=None)
    parser.add_argument("--mode", choices=sorted(MODE_METHODS), default="both")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_reference_bank(
    reference_dir: Path, n_refs: int, config: dict
) -> tuple[list[Path], Path, dict]:
    metadata_path = reference_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing reference metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata.get("references", [])[:n_refs]
    paths = [reference_dir / record["filename"] for record in records]
    if len(paths) != n_refs or not all(path.is_file() for path in paths):
        raise FileNotFoundError("Reference bank is incomplete")
    expected_target = config["model"]["target_model_id"]
    if metadata.get("target_model_id") != expected_target:
        raise ValueError("Reference bank target model does not match the config")
    expected_target_revision = config["model"].get("target_model_revision")
    if (
        expected_target_revision
        and metadata.get("target_model_revision") != expected_target_revision
    ):
        raise ValueError("Reference bank target revision does not match the config")
    recorded_proxy = metadata.get("models", {}).get("proxy_vae_for_attack", {})
    if recorded_proxy and (
        recorded_proxy.get("id") != config["model"]["proxy_vae_model_id"]
        or recorded_proxy.get("revision")
        != config["model"].get("proxy_vae_revision")
    ):
        raise ValueError("Reference bank proxy-VAE provenance does not match the config")
    expected_w_seed = int(config["watermark"]["w_seed"])
    if {int(record["w_seed"]) for record in records} != {expected_w_seed}:
        raise ValueError("Reference images do not share the configured watermark key")
    if config.get("reference_selection", {}).get("require_detected", False):
        threshold = float(config["evaluation"]["p_value_threshold"])
        if any(
            record.get("p_value") is None or float(record["p_value"]) > threshold
            for record in records
        ):
            raise ValueError("Reference bank contains an undetected reference")
    for record, path in zip(records, paths):
        if record.get("sha256") and record["sha256"] != sha256_file(path):
            raise ValueError(f"Reference image checksum mismatch: {path}")
    return paths, metadata_path, metadata


def write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "latent_loss", "pixel_loss", "total_loss"])
        writer.writeheader()
        writer.writerows(history)


def write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    img_size = int(config["watermark"]["img_size"])
    n_refs = int(config["prototype"]["reference_count"])
    retain_count = int(config["prototype"]["retain_count"])
    reference_dir = project_path(config, config["data"]["references_dir"])
    reference_paths, reference_metadata_path, reference_metadata = load_reference_bank(
        reference_dir, n_refs, config
    )

    vae = load_proxy_vae(config)
    reference_latents = []
    for path in reference_paths:
        tensor = preprocess_pil(load_rgb(path), img_size)
        reference_latents.append(encode_vae_latent(vae, tensor)[0])
    stacked = torch.stack(reference_latents, dim=0)
    robust = robust_latent_prototype(stacked, retain_count)
    baseline_target = stacked[0].float().unsqueeze(0)
    simple_target = simple_average_prototype(stacked).unsqueeze(0)
    full_target = robust.prototype.unsqueeze(0)
    targets = {
        "baseline": baseline_target,
        "simple_average": simple_target,
        "full": full_target,
    }

    cover_dir = (
        Path(args.cover_dir).expanduser().resolve()
        if args.cover_dir
        else project_path(config, config["data"]["cover_dir"])
    )
    limit = args.limit if args.limit is not None else int(config["data"]["cover_limit"])
    cover_paths = list_images(cover_dir, limit)
    methods = MODE_METHODS[args.mode]
    iterations = args.iterations or int(config["attack"]["num_iterations"])
    configured_steps = {int(step) for step in config["attack"].get("save_steps", [])}
    snapshot_steps = {step for step in configured_steps if step <= iterations}
    snapshot_cover_limit = int(config["attack"].get("snapshot_cover_limit", limit))

    output_root = project_path(config, config["data"]["output_dir"]) / "attacks"
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    manifest_path = run_dir / "manifest.json"
    existing_manifest = None
    config_path = Path(config["_config_path"])
    config_sha256 = sha256_file(config_path)
    if manifest_path.exists():
        if not args.skip_existing:
            raise FileExistsError(
                f"Run already exists: {run_dir}; choose another --run-name or pass --skip-existing"
            )
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(existing_manifest["iterations"]) != iterations:
            raise ValueError("Cannot resume a run with a different iteration count")
        if existing_manifest.get("methods") != methods:
            raise ValueError("Cannot resume a run with a different method set")
        if existing_manifest.get("config", {}).get("sha256") != config_sha256:
            raise ValueError("Cannot resume a run with a different configuration")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = run_dir / "config_snapshot.yaml"
    reference_snapshot = run_dir / "reference_metadata_snapshot.json"
    shutil.copyfile(config_path, config_snapshot)
    shutil.copyfile(reference_metadata_path, reference_snapshot)

    torch.save(
        {
            "baseline_target": baseline_target.cpu(),
            "simple_average_target": simple_target.cpu(),
            "full_target": full_target.cpu(),
            "reference_latents": stacked.cpu(),
            "median_center": robust.median_center.cpu(),
            "distances": robust.distances.cpu(),
            "retained_indices": robust.retained_indices.cpu(),
            "rejected_indices": robust.rejected_indices.cpu(),
        },
        run_dir / "prototype_artifacts.pt",
    )
    reference_stats = [
        {
            "index": index,
            "path": str(path),
            **latent_statistics(stacked[index]),
        }
        for index, path in enumerate(reference_paths)
    ]
    diagnostics = {
        "reference_paths": [str(path) for path in reference_paths],
        "reference_latent_statistics": reference_stats,
        "distances": [float(v) for v in robust.distances.cpu()],
        "retained_indices": robust.retained_indices.cpu().tolist(),
        "rejected_indices": robust.rejected_indices.cpu().tolist(),
        "baseline_target_statistics": latent_statistics(baseline_target),
        "simple_average_target_statistics": latent_statistics(simple_target),
        "full_target_statistics": latent_statistics(full_target),
    }
    (run_dir / "prototype_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    cover_manifest_path = run_dir / "cover_manifest.json"
    cover_manifest = {
        "dataset_name": config["data"].get("dataset_name", "unspecified"),
        "cover_root": str(cover_dir.resolve()),
        "selection": "recursive deterministic directory/name order",
        "limit": limit,
        "count": len(cover_paths),
        "files": [file_record(path, cover_dir) for path in cover_paths],
    }
    cover_manifest_path.write_text(
        json.dumps(cover_manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if existing_manifest is not None:
        manifest = existing_manifest
        manifest.setdefault("commands", []).append(list(sys.argv))
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["cover_manifest"] = {
            "path": str(cover_manifest_path.relative_to(run_dir)),
            "sha256": sha256_file(cover_manifest_path),
            "count": len(cover_paths),
        }
    else:
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_path": config["_config_path"],
            "run_dir": str(run_dir),
            "mode": args.mode,
            "methods": methods,
            "iterations": iterations,
            "experiment_seed": int(config["experiment"]["seed"]),
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
            "watermark": config["watermark"],
            "prototype": config["prototype"],
            "attack": {
                "lambda_pixel": float(config["attack"]["lambda_pixel"]),
                "alpha": float(config["attack"]["alpha"]),
                "iterations": iterations,
                "snapshot_steps": sorted(snapshot_steps),
                "snapshot_cover_limit": snapshot_cover_limit,
            },
            "config": {
                "snapshot": str(config_snapshot.relative_to(run_dir)),
                "sha256": config_sha256,
            },
            "reference_bank": {
                "metadata_snapshot": str(reference_snapshot.relative_to(run_dir)),
                "metadata_sha256": sha256_file(reference_metadata_path),
                "key_sha256": reference_metadata.get("key_sha256"),
                "accepted_count": len(reference_paths),
            },
            "cover_manifest": {
                "path": str(cover_manifest_path.relative_to(run_dir)),
                "sha256": sha256_file(cover_manifest_path),
                "count": len(cover_paths),
            },
            "git": git_provenance(config["_project_root"]),
            "runtime": runtime_provenance(),
            "commands": [list(sys.argv)],
            "entries": [],
        }
    existing_entries = {
        (entry["sample_id"], entry["method"]): entry
        for entry in manifest.get("entries", [])
    }

    for cover_index, cover_path in enumerate(tqdm(cover_paths, desc="covers")):
        cover_pil = load_rgb(cover_path)
        cover_tensor = preprocess_pil(cover_pil, img_size).unsqueeze(0)
        sample_id = f"{cover_index:04d}_{safe_stem(cover_path)}"
        clean_export = run_dir / "covers" / f"{sample_id}.png"
        clean_export.parent.mkdir(parents=True, exist_ok=True)
        tensor_to_pil(cover_tensor).save(clean_export)

        for method in methods:
            output_path = run_dir / method / f"{sample_id}.png"
            entry_key = (sample_id, method)
            if args.skip_existing and output_path.exists() and entry_key in existing_entries:
                print(f"skip existing {output_path}")
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            target = targets[method]

            def save_snapshot(step: int, tensor: torch.Tensor, *, _method=method, _sample=sample_id) -> None:
                snapshot = run_dir / "snapshots" / _method / f"{_sample}_step_{step}.png"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_pil(tensor).save(snapshot)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            print(
                f"start sample={sample_id} method={method} iterations={iterations}",
                flush=True,
            )

            def report_progress(
                record: dict[str, float | int],
                *,
                _method=method,
                _sample=sample_id,
            ) -> None:
                print(
                    f"progress sample={_sample} method={_method} "
                    f"step={record['step']}/{iterations} "
                    f"latent={record['latent_loss']:.6g} "
                    f"pixel={record['pixel_loss']:.6g} "
                    f"total={record['total_loss']:.6g}",
                    flush=True,
                )

            started = time.perf_counter()
            result = optimize_to_target_latent(
                cover=cover_tensor,
                target_latent=target,
                vae=vae,
                lambda_pixel=float(config["attack"]["lambda_pixel"]),
                alpha=float(config["attack"]["alpha"]),
                num_iterations=iterations,
                log_every=int(config["attack"]["log_every"]),
                snapshot_steps=(
                    snapshot_steps if cover_index < snapshot_cover_limit else set()
                ),
                snapshot_callback=(
                    save_snapshot if cover_index < snapshot_cover_limit else None
                ),
                progress_callback=report_progress,
            )
            runtime_seconds = time.perf_counter() - started
            peak_memory_mb = (
                torch.cuda.max_memory_allocated() / (1024**2)
                if torch.cuda.is_available()
                else None
            )
            tensor_to_pil(result.adversarial).save(output_path)
            history_path = run_dir / "logs" / method / f"{sample_id}.csv"
            write_history(history_path, result.history)
            new_entry = {
                "sample_id": sample_id,
                "method": method,
                "source_path": str(cover_path),
                "clean_export": str(clean_export.relative_to(run_dir)),
                "output": str(output_path.relative_to(run_dir)),
                "history": str(history_path.relative_to(run_dir)),
                "runtime_seconds": runtime_seconds,
                "peak_memory_mb": peak_memory_mb,
            }
            if entry_key in existing_entries:
                manifest["entries"] = [
                    new_entry if (entry["sample_id"], entry["method"]) == entry_key else entry
                    for entry in manifest["entries"]
                ]
            else:
                manifest["entries"].append(new_entry)
            existing_entries[entry_key] = new_entry
            write_manifest(manifest_path, manifest)

    write_manifest(manifest_path, manifest)
    print(f"attack run ready: {run_dir}")


if __name__ == "__main__":
    main()
