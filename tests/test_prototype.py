import torch

from rmlp.prototype import robust_latent_prototype


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

