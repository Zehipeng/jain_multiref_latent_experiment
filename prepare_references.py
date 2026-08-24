from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from rmlp.config import load_config, project_path
from rmlp.models import load_target_pipeline, seed_everything
from rmlp.reference_bank import accepts_reference, build_candidate_schedule
from rmlp.tree_ring import (
    build_key_and_mask,
    detect_p_value,
    generate_watermarked_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a same-key Tree-Ring reference bank.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify", action="store_true", help="Run expensive DDIM detection on each reference.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def tensor_digest(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def clear_generated_reference_files(output_dir: Path) -> None:
    """Remove only artifacts owned by this script after explicit --overwrite."""
    for path in output_dir.glob("ref_*.png"):
        if path.is_file():
            path.unlink()
    for filename in ("metadata.json", "metadata.partial.json", "watermark_key.pt"):
        path = output_dir / filename
        if path.is_file():
            path.unlink()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    prompts_path = project_path(config, config["data"]["reference_prompts_file"])
    prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_refs = int(config["prototype"]["reference_count"])
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_path}")

    wm = config["watermark"]
    selection = config.get("reference_selection", {})
    require_detected = bool(selection.get("require_detected", False))
    threshold = float(config["evaluation"]["p_value_threshold"])
    max_candidates = int(
        selection.get("max_candidates", len(wm["generation_seeds"]))
    )
    candidates = build_candidate_schedule(
        prompts,
        [int(seed) for seed in wm["generation_seeds"]],
        max_candidates,
    )
    if len(candidates) < n_refs:
        raise ValueError(
            f"Need at least {n_refs} reference candidates, found {len(candidates)}"
        )

    output_dir = project_path(config, config["data"]["references_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Reference directory is not empty: {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clear_generated_reference_files(output_dir)

    pipe = load_target_pipeline(config)
    w_key, w_mask = build_key_and_mask(pipe, config)
    torch.save({"w_key": w_key.cpu(), "w_mask": w_mask.cpu()}, output_dir / "watermark_key.pt")

    records = []
    rejected_candidates = []
    must_detect = args.verify or require_detected
    for candidate in candidates:
        image = generate_watermarked_image(
            pipe=pipe,
            prompt=candidate.prompt,
            w_key=w_key,
            w_mask=w_mask,
            img_size=int(wm["img_size"]),
            generation_seed=candidate.generation_seed,
            num_inference_steps=int(wm["generation_steps"]),
        )
        p_value = None
        if must_detect:
            p_value = detect_p_value(
                image,
                pipe,
                w_key,
                w_mask,
                int(wm["img_size"]),
                int(config["evaluation"]["inversion_steps"]),
            )
        candidate_record = {
            "candidate_index": candidate.candidate_index,
            "prompt_index": candidate.prompt_index,
            "prompt": candidate.prompt,
            "generation_seed": candidate.generation_seed,
            "w_seed": int(wm["w_seed"]),
            "p_value": p_value,
        }
        if not accepts_reference(p_value, threshold, require_detected):
            rejected_candidates.append(candidate_record)
            print(
                f"rejected candidate={candidate.candidate_index} "
                f"gseed={candidate.generation_seed} p_value={p_value}",
                flush=True,
            )
            continue

        index = len(records)
        filename = f"ref_{index:02d}_gseed_{candidate.generation_seed}.png"
        image.save(output_dir / filename)
        records.append(
            {
                "index": index,
                "filename": filename,
                **candidate_record,
            }
        )
        print(f"accepted {filename} p_value={p_value}", flush=True)
        if p_value is not None and p_value > threshold:
            print(
                f"WARNING: {filename} is not detected at the configured threshold; "
                "keep the record and inspect it before attacking.",
                flush=True,
            )
        if len(records) == n_refs:
            break

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config["_config_path"],
        "target_model_id": config["model"]["target_model_id"],
        "watermark": wm,
        "reference_selection": {
            "require_detected": require_detected,
            "p_value_threshold": threshold,
            "max_candidates": max_candidates,
            "attempted_candidates": len(records) + len(rejected_candidates),
            "accepted_count": len(records),
            "rejected_count": len(rejected_candidates),
        },
        "key_sha256": tensor_digest(w_key),
        "mask_true_count": int(w_mask.sum().item()),
        "references": records,
        "rejected_candidates": rejected_candidates,
    }
    metadata_name = "metadata.json" if len(records) == n_refs else "metadata.partial.json"
    (output_dir / metadata_name).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if len(records) != n_refs:
        raise RuntimeError(
            f"Only {len(records)}/{n_refs} detector-positive references were found "
            f"after {len(candidates)} candidates; add seeds or raise max_candidates"
        )
    print(f"reference bank ready: {output_dir}")


if __name__ == "__main__":
    main()
