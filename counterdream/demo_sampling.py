"""Inference experiments for existing checkpoints; no external frames or weights."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingOptions:
    steps: int = 8
    sigma_max: float = 20.0
    solver: str = "euler"
    context_noise: float = 0.0
    warm_start: bool = False
    initial_noise_scale: float = 1.0


@torch.inference_mode()
def sample(model, context, actions, options, seed, initial_image=None):
    if options.solver not in ("euler", "heun"):
        raise ValueError("Unknown sampler")
    if not 2 <= options.steps <= 32 or not 0.05 <= options.sigma_max <= 80:
        raise ValueError("Invalid denoising schedule")
    if not 0 <= options.context_noise <= 0.5:
        raise ValueError("Context noise outside trained range")
    if not 0 < options.initial_noise_scale <= 2:
        raise ValueError("Invalid initial noise scale")
    b, _, c, h, w = context.shape
    gen = torch.Generator(device=context.device).manual_seed(seed)
    noise = torch.randn((b, c, h, w), device=context.device, generator=gen)
    x = noise * options.sigma_max * options.initial_noise_scale
    if options.warm_start:
        initial = context[:, -1] if initial_image is None else initial_image
        if initial.shape != x.shape:
            raise ValueError("Initial image does not match the model output")
        x = x + initial
    sc = torch.full((b,), options.context_noise, device=context.device)
    conditioned = context
    if options.context_noise:
        conditioned = (
            context
            + torch.randn(context.shape, device=context.device, generator=gen)
            * options.context_noise
        )
    ramp = torch.linspace(0, 1, options.steps, device=context.device)
    sigmas = (
        options.sigma_max ** (1 / 7)
        + ramp * (0.002 ** (1 / 7) - options.sigma_max ** (1 / 7))
    ) ** 7
    sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
    for i in range(options.steps):
        current, following = sigmas[i], sigmas[i + 1]
        clean = model(x, current.expand(b), conditioned, actions, sc).clamp(-1, 1)
        derivative = (x - clean) / current
        # Preserve the original Euler operation order for baseline compatibility.
        proposal = x + (following - current) * (x - clean) / current
        if options.solver == "heun" and i < options.steps - 1:
            clean_next = model(
                proposal, following.expand(b), conditioned, actions, sc
            ).clamp(-1, 1)
            derivative_next = (proposal - clean_next) / following
            proposal = x + (following - current) * (derivative + derivative_next) / 2
        x = proposal
    return x.clamp(-1, 1)
