from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from rmlp.config import load_config, project_path
from rmlp.models import load_target_pipeline, seed_everything
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    prompts_path = project_path(config, config["data"]["reference_prompts_file"])
    prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_refs = int(config["prototype"]["reference_count"])
    if len(prompts) < n_refs:
        raise ValueError(f"Need {n_refs} prompts, found {len(prompts)} in {prompts_path}")

    output_dir = project_path(config, config["data"]["references_dir"])
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Reference directory is not empty: {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_target_pipeline(config)
    w_key, w_mask = build_key_and_mask(pipe, config)
    torch.save({"w_key": w_key.cpu(), "w_mask": w_mask.cpu()}, output_dir / "watermark_key.pt")

    wm = config["watermark"]
    records = []
    for index in range(n_refs):
        generation_seed = int(wm["generation_seeds"][index])
        image = generate_watermarked_image(
            pipe=pipe,
            prompt=prompts[index],
            w_key=w_key,
            w_mask=w_mask,
            img_size=int(wm["img_size"]),
            generation_seed=generation_seed,
            num_inference_steps=int(wm["generation_steps"]),
        )
        filename = f"ref_{index:02d}_gseed_{generation_seed}.png"
        image.save(output_dir / filename)
        p_value = None
        if args.verify:
            p_value = detect_p_value(
                image,
                pipe,
                w_key,
                w_mask,
                int(wm["img_size"]),
                int(config["evaluation"]["inversion_steps"]),
            )
        records.append(
            {
                "index": index,
                "filename": filename,
                "prompt": prompts[index],
                "generation_seed": generation_seed,
                "w_seed": int(wm["w_seed"]),
                "p_value": p_value,
            }
        )
        print(f"generated {filename} p_value={p_value}", flush=True)
        if p_value is not None and p_value > float(config["evaluation"]["p_value_threshold"]):
            print(
                f"WARNING: {filename} is not detected at the configured threshold; "
                "keep the record and inspect it before attacking.",
                flush=True,
            )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config["_config_path"],
        "target_model_id": config["model"]["target_model_id"],
        "watermark": wm,
        "key_sha256": tensor_digest(w_key),
        "mask_true_count": int(w_mask.sum().item()),
        "references": records,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"reference bank ready: {output_dir}")


if __name__ == "__main__":
    main()
