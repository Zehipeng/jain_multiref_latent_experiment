from __future__ import annotations

import os
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(directory: str | Path, limit: int | None = None) -> list[Path]:
    """Recursively list images in deterministic order, stopping at ``limit``."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    if limit is not None and limit < 1:
        raise ValueError("Image limit must be positive")

    paths: list[Path] = []
    for current_root, directory_names, filenames in os.walk(root):
        directory_names.sort()
        for filename in sorted(filenames):
            path = Path(current_root) / filename
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            paths.append(path)
            if limit is not None and len(paths) == limit:
                return paths

    if not paths:
        raise FileNotFoundError(
            f"No supported images found recursively in: {root}"
        )
    return paths
