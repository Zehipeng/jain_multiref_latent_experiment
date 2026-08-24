from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_provenance(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    status = _git(root, "status", "--short")
    tracked_status = _git(root, "status", "--short", "--untracked-files=no")
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "status_short": status,
        "tracked_status_short": tracked_status,
        "tracked_clean": tracked_status == "" if tracked_status is not None else None,
    }


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_provenance() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            [
                "torch",
                "diffusers",
                "transformers",
                "accelerate",
                "huggingface-hub",
                "lpips",
            ]
        ),
    }


def file_record(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    relative_path = None
    if root is not None:
        try:
            relative_path = resolved.relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            relative_path = None
    return {
        "path": str(resolved),
        "relative_path": relative_path,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
