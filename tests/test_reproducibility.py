from pathlib import Path

from rmlp.reproducibility import file_record, package_versions, sha256_file


def test_sha256_and_file_record(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    image = root / "sample.jpg"
    image.write_bytes(b"cover-bytes")

    record = file_record(image, root)

    assert record["relative_path"] == "sample.jpg"
    assert record["size_bytes"] == 11
    assert record["sha256"] == sha256_file(image)


def test_missing_package_version_is_none() -> None:
    versions = package_versions(["package-that-does-not-exist-rmlp-test"])
    assert versions["package-that-does-not-exist-rmlp-test"] is None
