"""Experimental blend of two model predictions; never uses a weapon sprite."""

from dataclasses import replace
import torch

from .demo_sampling import sample


@torch.inference_mode()
def refine_firing(model, context, actions, predicted, options, seed, strength):
    if not 0 <= strength <= 1:
        raise ValueError("Weapon refinement strength must be in [0,1]")
    if strength == 0 or not bool((actions[:, -1, 11] > 0.5).any()):
        return predicted
    proposal = sample(
        model,
        context,
        actions,
        replace(options, sigma_max=20.0, warm_start=False, fire_guidance=1.0),
        seed=seed + 500000,
    )
    height, width = predicted.shape[-2:]
    y = torch.linspace(0, 1, height, device=predicted.device)[None, None, :, None]
    x = torch.linspace(0, 1, width, device=predicted.device)[None, None, None, :]
    # Fixed soft lower-right region, not a segmentation or a physics model.
    mask = torch.sigmoid((x - 0.4) * 25) * torch.sigmoid((y - 0.45) * 30) * strength
    return predicted.lerp(proposal, mask).clamp(-1, 1)
