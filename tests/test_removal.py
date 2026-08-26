import torch

from rmlp.removal import mean_image_target, mean_shift_target, removal_mode


def test_mean_image_target_is_constant_and_preserves_shape() -> None:
    image = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    target = mean_image_target(image)
    assert target.shape == image.shape
    assert torch.all(target == 4.0)


def test_mean_shift_target_matches_defined_direction() -> None:
    target = torch.tensor([[[[5.0]]]])
    watermarked = torch.tensor([[[[4.0]]]])
    clean = torch.tensor([[[[1.0]]]])
    assert torch.equal(mean_shift_target(target, watermarked, clean, 1.0), torch.tensor([[[[2.0]]]]))


def test_removal_modes() -> None:
    assert removal_mode("latent_repulsion") == "repel"
    assert removal_mode("mean_shift") == "attract"
