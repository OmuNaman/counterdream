"""Compact pixel-space EDM world model, initialized entirely from random weights.

Algorithmic references: Karras et al. (EDM, 2022), Alonso et al. (DIAMOND, 2024).
This is a standalone implementation; no pretrained weights are loaded.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    height: int = 64
    width: int = 112
    context: int = 4
    action_dim: int = 51
    base: int = 64
    cond_dim: int = 256
    sigma_data: float = 0.5
    version: int = 2


class Residual(nn.Module):
    def __init__(self, cin, cout, cond):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.affine = nn.Linear(cond, cout * 2)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, bias = self.affine(F.silu(cond)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        return (self.skip(x) + self.conv2(F.silu(h))) / math.sqrt(2)


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, 4, c // 4, h * w).unbind(1)
        a = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        return x + self.out(a.transpose(-1, -2).reshape(b, c, h, w))


class WorldModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c, d = self.cfg.base, self.cfg.cond_dim
        frequencies = (
            torch.exp(torch.linspace(math.log(1), math.log(1000), 32))
            if self.cfg.version == 1
            else torch.randn(32) * (2 * math.pi)
        )
        self.register_buffer("frequencies", frequencies)
        self.noise_emb = nn.Sequential(nn.Linear(128, d), nn.SiLU(), nn.Linear(d, d))
        self.action_emb = nn.Sequential(
            nn.Linear(self.cfg.context * self.cfg.action_dim, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.stem = nn.Conv2d(3 * (self.cfg.context + 1), c, 3, padding=1)
        self.e0 = nn.ModuleList([Residual(c, c, d), Residual(c, c, d)])
        self.down0 = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)
        self.e1 = nn.ModuleList([Residual(c * 2, c * 2, d), Residual(c * 2, c * 2, d)])
        self.down1 = nn.Conv2d(c * 2, c * 3, 3, stride=2, padding=1)
        self.e2 = nn.ModuleList([Residual(c * 3, c * 3, d), Residual(c * 3, c * 3, d)])
        self.down2 = nn.Conv2d(c * 3, c * 4, 3, stride=2, padding=1)
        self.middle1 = Residual(c * 4, c * 4, d)
        self.attention = Attention(c * 4)
        self.middle2 = Residual(c * 4, c * 4, d)
        self.u2 = nn.ModuleList([Residual(c * 7, c * 3, d), Residual(c * 3, c * 3, d)])
        self.u1 = nn.ModuleList([Residual(c * 5, c * 2, d), Residual(c * 2, c * 2, d)])
        self.u0 = nn.ModuleList([Residual(c * 3, c, d), Residual(c, c, d)])
        self.out = nn.Sequential(
            nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1)
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def embed_noise(self, sigma):
        level = sigma.clamp_min(1e-5).log() / 4
        if self.cfg.version >= 2:
            level = torch.where(sigma == 0, torch.zeros_like(level), level)
        features = level[:, None] * self.frequencies[None, :]
        return torch.cat((features.sin(), features.cos()), dim=1)

    def forward(self, noisy, sigma, context, actions, context_sigma=None):
        b = noisy.shape[0]
        sigma = sigma.expand(b)
        if context_sigma is None:
            context_sigma = (
                torch.full_like(sigma, 1e-5)
                if self.cfg.version == 1
                else torch.zeros_like(sigma)
            )
        cond = self.noise_emb(
            torch.cat((self.embed_noise(sigma), self.embed_noise(context_sigma)), dim=1)
        )
        cond = cond + self.action_emb(actions.flatten(1))
        sd = self.cfg.sigma_data
        s = sigma[:, None, None, None]
        cin = (s * s + sd * sd).rsqrt()
        cskip = sd * sd / (s * s + sd * sd)
        cout = s * sd * (s * s + sd * sd).rsqrt()
        x = self.stem(torch.cat((noisy * cin, context.flatten(1, 2) / sd), dim=1))
        for layer in self.e0:
            x = layer(x, cond)
        s0 = x
        x = self.down0(x)
        for layer in self.e1:
            x = layer(x, cond)
        s1 = x
        x = self.down1(x)
        for layer in self.e2:
            x = layer(x, cond)
        s2 = x
        x = self.middle2(self.attention(self.middle1(self.down2(x), cond)), cond)
        for layers, skip in ((self.u2, s2), (self.u1, s1), (self.u0, s0)):
            x = torch.cat(
                (F.interpolate(x, size=skip.shape[-2:], mode="nearest"), skip), dim=1
            )
            for layer in layers:
                x = layer(x, cond)
        return cskip * noisy + cout * self.out(x)

    def loss(self, context, actions, target, sigma=None, noise=None):
        b = target.shape[0]
        if sigma is None:
            sigma = (
                (torch.randn(b, device=target.device) * 1.2 - 1.2)
                .exp()
                .clamp(0.002, 20)
            )
        if noise is None:
            noise = torch.randn_like(target)
        # Noise augmentation prepares the context for imperfect generated frames.
        sc = (torch.randn(b, device=target.device) * 1.0 - 3.5).exp().clamp(0.001, 0.5)
        if self.cfg.version >= 2:
            # Clean context is explicitly trained, including the inference condition.
            sc = torch.where(
                torch.rand(b, device=target.device) < 0.2, torch.zeros_like(sc), sc
            )
        noised_context = (
            context + torch.randn_like(context) * sc[:, None, None, None, None]
        )
        noisy = target + noise * sigma[:, None, None, None]
        pred = self(noisy, sigma, noised_context, actions, sc)
        weights = (sigma.square() + self.cfg.sigma_data**2) / (
            sigma * self.cfg.sigma_data
        ).square()
        loss = (
            (pred.float() - target.float()).square().mean((1, 2, 3)) * weights
        ).mean()
        return loss, pred.clamp(-1, 1)

    @torch.no_grad()
    def sample(self, context, actions, steps=8, seed=None, sigma_max=None):
        if not 2 <= steps <= 32:
            raise ValueError("Use 2–32 denoising steps")
        b, _, c, h, w = context.shape
        if sigma_max is None:
            sigma_max = 5.0 if self.cfg.version == 1 else 20.0
        if not 0.05 <= sigma_max <= 80:
            raise ValueError("sigma_max must be between 0.05 and 80")
        generator = (
            None
            if seed is None
            else torch.Generator(device=context.device).manual_seed(seed)
        )
        x = (
            torch.randn((b, c, h, w), device=context.device, generator=generator)
            * sigma_max
        )
        ramp = torch.linspace(0, 1, steps, device=context.device)
        sigmas = (
            sigma_max ** (1 / 7) + ramp * (0.002 ** (1 / 7) - sigma_max ** (1 / 7))
        ) ** 7
        sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
        for i in range(steps):
            s = sigmas[i]
            denoised = self(x, s.expand(b), context, actions).clamp(-1, 1)
            x = x + (sigmas[i + 1] - s) * (x - denoised) / s
        return x.clamp(-1, 1)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    config.setdefault("version", 1)
    model = WorldModel(ModelConfig(**config)).to(device)
    model.load_state_dict(
        checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
    )
    return model.eval(), checkpoint
