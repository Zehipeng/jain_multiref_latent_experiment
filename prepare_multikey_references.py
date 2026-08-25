from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from prepare_references import prepare_reference_bank
from rmlp.config import load_config, project_path
from rmlp.models import load_target_pipeline, seed_everything
from rmlp.multikey import key_directory_name
from rmlp.reproducibility import git_provenance, runtime_provenance, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare five detector-positive references for every key."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def complete_existing_bank(
    path: Path, w_seed: int, count: int, config: dict
) -> dict | None:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata.get("references", [])
    if int(metadata.get("watermark", {}).get("w_seed", -1)) != w_seed:
        return None
    recorded_wm = metadata.get("watermark", {})
    for field in ("w_channel", "w_radius", "img_size", "generation_steps"):
        if recorded_wm.get(field) != config["watermark"].get(field):
            return None
    if metadata.get("target_model_id") != config["model"]["target_model_id"]:
        return None
    if metadata.get("target_model_revision") != config["model"].get(
        "target_model_revision"
    ):
        return None
    if len(records) != count:
        return None
    if not all(
        (path / record["filename"]).is_file()
        and record.get("sha256") == sha256_file(path / record["filename"])
        for record in records
    ):
        return None
    return metadata


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["experiment"]["seed"]))
    key_seeds = [int(seed) for seed in config["multikey"]["key_seeds"]]
    if len(set(key_seeds)) != len(key_seeds):
        raise ValueError("multikey.key_seeds must be unique")
    n_refs = int(config["prototype"]["reference_count"])
    root = project_path(config, config["data"]["references_dir"])
    root.mkdir(parents=True, exist_ok=True)
    config_path = Path(config["_config_path"])
    shutil.copyfile(config_path, root / "config_snapshot.yaml")

    pipe = load_target_pipeline(config)
    key_records = []
    for w_seed in key_seeds:
        key_dir = root / key_directory_name(w_seed)
        existing = complete_existing_bank(key_dir, w_seed, n_refs, config)
        if existing is not None and args.skip_existing and not args.overwrite:
            print(f"skip complete reference bank: {key_dir}", flush=True)
            metadata = existing
        else:
            key_config = deepcopy(config)
            key_config["watermark"]["w_seed"] = w_seed
            metadata = prepare_reference_bank(
                key_config,
                pipe,
                key_dir,
                verify=args.verify,
                overwrite=args.overwrite,
                command=list(sys.argv),
            )
        key_records.append(
            {
                "w_seed": w_seed,
                "directory": key_directory_name(w_seed),
                "metadata_sha256": sha256_file(key_dir / "metadata.json"),
                "key_sha256": metadata["key_sha256"],
                "accepted_count": len(metadata["references"]),
            }
        )

    root_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": "one key paired with one cover; five references per key",
        "config_path": config["_config_path"],
        "config_sha256": sha256_file(config_path),
        "key_count": len(key_seeds),
        "reference_count_per_key": n_refs,
        "keys": key_records,
        "git": git_provenance(config["_project_root"]),
        "runtime": runtime_provenance(),
        "command": list(sys.argv),
    }
    (root / "multikey_manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"multi-key reference banks ready: {root}")


if __name__ == "__main__":
    main()
