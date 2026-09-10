"""A small learned motion model; actions predict image flow, not game rules."""

import torch
from torch import nn
from torch.nn import functional as F


def warp(frame, flow):
    b, c, h, w = frame.shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=frame.device),
        torch.linspace(-1, 1, w, device=frame.device),
        indexing="ij",
    )
    grid = torch.stack((x, y), -1)[None].expand(b, -1, -1, -1)
    delta = torch.stack((flow[:, 0] * 2 / (w - 1), flow[:, 1] * 2 / (h - 1)), -1)
    return F.grid_sample(
        frame, grid + delta, mode="bilinear", padding_mode="border", align_corners=True
    )


class MotionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 5, 2, 2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 96, 3, 2, 1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
        )
        self.action = nn.Sequential(nn.Linear(102, 128), nn.SiLU(), nn.Linear(128, 192))
        self.middle = nn.Sequential(
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 64, 3, 1, 1),
            nn.SiLU(),
        )
        self.flow = nn.Conv2d(64, 2, 3, 1, 1)
        self.residual = nn.Conv2d(64, 3, 3, 1, 1)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, context, actions):
        features = self.encoder(context[:, -2:].flatten(1, 2))
        scale, bias = self.action(actions[:, -2:].flatten(1)).chunk(2, 1)
        features = self.middle(
            features * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        )
        flow = (
            F.interpolate(
                self.flow(features),
                size=context.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).tanh()
            * 24
        )
        residual = (
            F.interpolate(
                self.residual(features),
                size=context.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).tanh()
            * 0.25
        )
        prediction = (warp(context[:, -1], flow) + residual).clamp(-1, 1)
        return prediction, flow, residual


def guided_motion(model, context, actions, scale=1.0, center=False):
    """Optional action contrast removes motion also predicted for an idle input."""
    if not 0 < scale <= 2:
        raise ValueError("Motion scale must be in (0,2]")
    prediction, flow, residual = model(context, actions)
    if center:
        from .actions import encode

        neutral = actions.clone()
        neutral[:, -1] = torch.as_tensor(encode(), device=actions.device)
        _, base_flow, base_residual = model(context, neutral)
        flow, residual = flow - base_flow, residual - base_residual
    if center or scale != 1.0:
        prediction = (warp(context[:, -1], flow * scale) + residual * scale).clamp(
            -1, 1
        )
    return prediction
