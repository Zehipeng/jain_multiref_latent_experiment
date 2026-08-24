from __future__ import annotations

import argparse
import csv
import json
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
from rmlp.prototype import encode_vae_latent, robust_latent_prototype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Jain baseline and robust multi-reference forgery.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cover-dir", default=None)
    parser.add_argument("--mode", choices=["baseline", "full", "both"], default="both")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_reference_paths(reference_dir: Path, n_refs: int) -> list[Path]:
    metadata_path = reference_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing reference metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata.get("references", [])[:n_refs]
    paths = [reference_dir / record["filename"] for record in records]
    if len(paths) != n_refs or not all(path.is_file() for path in paths):
        raise FileNotFoundError("Reference bank is incomplete")
    return paths


def write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "latent_loss", "pixel_loss", "total_loss"])
        writer.writeheader()
        writer.writerows(history)


def write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
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
    reference_paths = load_reference_paths(reference_dir, n_refs)

    vae = load_proxy_vae(config)
    reference_latents = []
    for path in reference_paths:
        tensor = preprocess_pil(load_rgb(path), img_size)
        reference_latents.append(encode_vae_latent(vae, tensor)[0])
    stacked = torch.stack(reference_latents, dim=0)
    robust = robust_latent_prototype(stacked, retain_count)
    baseline_target = stacked[0].unsqueeze(0)
    full_target = robust.prototype.unsqueeze(0)

    output_root = project_path(config, config["data"]["output_dir"]) / "attacks"
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and not args.skip_existing:
        raise FileExistsError(
            f"Run already exists: {run_dir}; choose another --run-name or pass --skip-existing"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "baseline_target": baseline_target.cpu(),
            "full_target": full_target.cpu(),
            "reference_latents": stacked.cpu(),
            "median_center": robust.median_center.cpu(),
            "distances": robust.distances.cpu(),
            "retained_indices": robust.retained_indices.cpu(),
            "rejected_indices": robust.rejected_indices.cpu(),
        },
        run_dir / "prototype_artifacts.pt",
    )
    diagnostics = {
        "reference_paths": [str(path) for path in reference_paths],
        "distances": [float(v) for v in robust.distances.cpu()],
        "retained_indices": robust.retained_indices.cpu().tolist(),
        "rejected_indices": robust.rejected_indices.cpu().tolist(),
    }
    (run_dir / "prototype_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cover_dir = Path(args.cover_dir).expanduser().resolve() if args.cover_dir else project_path(config, config["data"]["cover_dir"])
    limit = args.limit if args.limit is not None else int(config["data"]["cover_limit"])
    cover_paths = list_images(cover_dir, limit)
    methods = [args.mode] if args.mode != "both" else ["baseline", "full"]
    iterations = args.iterations or int(config["attack"]["num_iterations"])
    configured_steps = {int(step) for step in config["attack"].get("save_steps", [])}
    snapshot_steps = {step for step in configured_steps if step <= iterations}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config["_config_path"],
        "run_dir": str(run_dir),
        "mode": args.mode,
        "iterations": iterations,
        "lambda_pixel": float(config["attack"]["lambda_pixel"]),
        "alpha": float(config["attack"]["alpha"]),
        "entries": [],
    }
    if manifest_path.exists() and args.skip_existing:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["iterations"]) != iterations:
            raise ValueError("Cannot resume a run with a different iteration count")
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
            target = baseline_target if method == "baseline" else full_target

            def save_snapshot(step: int, tensor: torch.Tensor, *, _method=method, _sample=sample_id) -> None:
                snapshot = run_dir / "snapshots" / _method / f"{_sample}_step_{step}.png"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                tensor_to_pil(tensor).save(snapshot)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result = optimize_to_target_latent(
                cover=cover_tensor,
                target_latent=target,
                vae=vae,
                lambda_pixel=float(config["attack"]["lambda_pixel"]),
                alpha=float(config["attack"]["alpha"]),
                num_iterations=iterations,
                log_every=int(config["attack"]["log_every"]),
                snapshot_steps=snapshot_steps,
                snapshot_callback=save_snapshot,
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
