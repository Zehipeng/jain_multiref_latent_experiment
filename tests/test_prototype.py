import pytest
import torch

from rmlp.prototype import (
    latent_statistics,
    robust_latent_prototype,
    simple_average_prototype,
)


def test_robust_prototype_rejects_obvious_outlier() -> None:
    normal = torch.tensor([0.0, 0.1, -0.1, 0.05]).view(4, 1, 1, 1)
    outlier = torch.tensor([10.0]).view(1, 1, 1, 1)
    latents = torch.cat((normal, outlier), dim=0)
    result = robust_latent_prototype(latents, retain_count=4)
    assert result.rejected_indices.tolist() == [4]
    assert torch.allclose(result.prototype, normal.mean(dim=0))


def test_retain_count_validation() -> None:
    latents = torch.zeros(2, 1, 1, 1)
    try:
        robust_latent_prototype(latents, retain_count=3)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_distances_are_float32_and_finite_for_large_fp16_values() -> None:
    latents = torch.tensor([0.0, 100.0, 200.0, 600.0], dtype=torch.float16).view(
        4, 1, 1, 1
    )
    result = robust_latent_prototype(latents, retain_count=3)
    assert result.distances.dtype == torch.float32
    assert torch.isfinite(result.distances).all()
    assert result.rejected_indices.tolist() == [3]


def test_simple_average_uses_all_references_in_float32() -> None:
    latents = torch.tensor([1.0, 3.0, 8.0], dtype=torch.float16).view(3, 1, 1, 1)
    prototype = simple_average_prototype(latents)
    assert prototype.dtype == torch.float32
    assert torch.allclose(prototype, torch.tensor([4.0]).view(1, 1, 1))


def test_nonfinite_latents_fail_closed() -> None:
    latents = torch.tensor([0.0, float("inf")]).view(2, 1, 1, 1)
    with pytest.raises(ValueError, match="non-finite"):
        robust_latent_prototype(latents, retain_count=1)
    with pytest.raises(ValueError, match="non-finite"):
        simple_average_prototype(latents)
    assert latent_statistics(latents[1])["finite"] is False
