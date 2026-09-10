"""Optional Real-ESRGAN display enhancement, separate from world-model generation.

Compatible compact architecture adapted from Xintao Wang's Real-ESRGAN
SRVGGNetCompact (BSD-3-Clause). See assets/REALESRGAN-LICENSE.txt.
https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/archs/srvgg_arch.py
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
WEIGHTS_SHA256 = "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"


def download_weights(destination):
    """Fetch the explicitly credited, pinned display model, never world weights."""
    import hashlib
    import requests

    destination = Path(destination)
    if (
        destination.is_file()
        and hashlib.sha256(destination.read_bytes()).hexdigest() == WEIGHTS_SHA256
    ):
        return destination
    if destination.exists():
        raise ValueError("Destination exists with different weights")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    digest = hashlib.sha256()
    with requests.get(WEIGHTS_URL, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                digest.update(block)
                stream.write(block)
    if digest.hexdigest() != WEIGHTS_SHA256:
        temporary.unlink()
        raise ValueError("Display-model checksum mismatch")
    temporary.replace(destination)
    return destination


class DisplayUpscaler(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(64)]
        for _ in range(32):
            layers.extend((nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(64)))
        layers.append(nn.Conv2d(64, 48, 3, 1, 1))
        self.body = nn.ModuleList(layers)
        self.upsampler = nn.PixelShuffle(4)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB BCHW image in [0, 1]")
        result = image
        for layer in self.body:
            result = layer(result)
        return (
            self.upsampler(result)
            + F.interpolate(image, scale_factor=4, mode="nearest")
        ).clamp(0, 1)

    @classmethod
    def from_weights(cls, path, device="cuda"):
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
        model = cls()
        model.load_state_dict(state, strict=True)
        return model.to(device).eval().requires_grad_(False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--download", required=True)
    print(download_weights(parser.parse_args().download))
