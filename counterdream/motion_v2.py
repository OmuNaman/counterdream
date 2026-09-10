"""Experimental action-conditioned motion with a four-frame training unroll."""

import torch
from torch import nn
from torch.nn import functional as F
from .motion_model import warp


class MotionV2(nn.Module):
    def __init__(self):
        super().__init__()

        def block(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride, 1),
                nn.GroupNorm(8, cout),
                nn.SiLU(),
                nn.Conv2d(cout, cout, 3, 1, 1),
                nn.GroupNorm(8, cout),
                nn.SiLU(),
            )

        self.action = nn.Sequential(nn.Linear(102, 128), nn.SiLU(), nn.Linear(128, 64))
        self.e0 = block(6 + 2 + 64, 48)
        self.e1 = block(48, 96, 2)
        self.e2 = block(96, 128, 2)
        self.middle = block(128, 128)
        self.u1 = block(128 + 96, 96)
        self.u0 = block(96 + 48, 48)
        self.head = nn.Conv2d(48, 5, 3, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, context, actions):
        b, _, _, h, w = context.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=context.device),
            torch.linspace(-1, 1, w, device=context.device),
            indexing="ij",
        )
        coords = torch.stack((xx, yy))[None].expand(b, -1, -1, -1)
        action = self.action(actions[:, -2:].flatten(1))[:, :, None, None].expand(
            -1, -1, h, w
        )
        x0 = self.e0(torch.cat((context[:, -2:].flatten(1, 2), coords, action), 1))
        x1 = self.e1(x0)
        x2 = self.middle(self.e2(x1))
        x = self.u1(
            torch.cat(
                (
                    F.interpolate(
                        x2, size=x1.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    x1,
                ),
                1,
            )
        )
        x = self.u0(
            torch.cat(
                (
                    F.interpolate(
                        x, size=x0.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    x0,
                ),
                1,
            )
        )
        flow, residual = self.head(x).split((2, 3), 1)
        flow, residual = flow.tanh() * 24, residual.tanh() * 0.3
        return (warp(context[:, -1], flow) + residual).clamp(-1, 1), flow, residual


def load_motion(path, device="cuda"):
    from .motion_model import MotionPredictor

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture", "MotionPredictor-v1")
    if architecture not in ("MotionPredictor-v1", "MotionV2"):
        raise ValueError("Unknown motion architecture")
    model = (
        (MotionV2() if architecture == "MotionV2" else MotionPredictor())
        .to(device)
        .eval()
    )
    model.load_state_dict(checkpoint["model"])
    return model
