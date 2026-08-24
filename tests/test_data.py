from pathlib import Path

import pytest

from rmlp.data import list_images


def test_list_images_recurses_sorts_and_limits(tmp_path: Path) -> None:
    (tmp_path / "train2017").mkdir()
    (tmp_path / "val2017").mkdir()
    (tmp_path / "train2017" / "0002.jpg").touch()
    (tmp_path / "train2017" / "0001.png").touch()
    (tmp_path / "val2017" / "0000.jpeg").touch()
    (tmp_path / "annotations.json").touch()

    paths = list_images(tmp_path, limit=2)

    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        "train2017/0001.png",
        "train2017/0002.jpg",
    ]


def test_list_images_reports_empty_tree(tmp_path: Path) -> None:
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "instances.json").touch()

    with pytest.raises(FileNotFoundError, match="recursively"):
        list_images(tmp_path)


def test_list_images_rejects_nonpositive_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        list_images(tmp_path, limit=0)
