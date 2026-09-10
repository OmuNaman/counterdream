import pytest
import torch

from counterdream.demo_sampling import SamplingOptions, sample
from counterdream.model import ModelConfig, WorldModel


def test_default_sampler_preserves_existing_generation():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = WorldModel(ModelConfig(base=16)).eval()
    context = torch.rand(1, 4, 3, 64, 112) * 2 - 1
    actions = torch.zeros(1, 4, 51)
    expected = model.sample(context, actions, steps=4, seed=123)
    actual = sample(model, context, actions, SamplingOptions(steps=4), seed=123)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "options",
    [
        SamplingOptions(solver="invalid"),
        SamplingOptions(steps=1),
        SamplingOptions(context_noise=0.6),
        SamplingOptions(initial_noise_scale=0),
    ],
)
def test_invalid_profiles_rejected_before_inference(options):
    with pytest.raises(ValueError):
        sample(None, torch.zeros(1, 4, 3, 64, 112), None, options, seed=0)


def test_noisy_context_repeatable_and_bounded():
    class Denoiser:
        def __call__(self, x, sigma, context, actions, context_sigma):
            return context[:, -1] * 0.8 + actions[:, -1, :3, None, None] * 0.1

    context = torch.zeros(1, 4, 3, 8, 8)
    actions = torch.ones(1, 4, 51)
    options = SamplingOptions(steps=8, context_noise=0.03)
    first = sample(Denoiser(), context, actions, options, 33)
    second = sample(Denoiser(), context, actions, options, 33)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.isfinite(first).all() and first.abs().max() <= 1
    changed = sample(Denoiser(), context, torch.zeros_like(actions), options, 33)
    assert (first - changed).abs().mean() > 0.05
