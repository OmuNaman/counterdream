"""Render actual saved model frames, with matched controls and explicit labels."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .demo_record import controls, sha256
from .upscale import DisplayUpscaler


@torch.inference_mode()
def render(baseline, candidate, upscaler, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(Path(baseline) / "native-frames.npz", allow_pickle=False) as data:
        before = data["frames"].copy()
        actions = data["actions"].copy()
    with np.load(Path(candidate) / "native-frames.npz", allow_pickle=False) as data:
        after = data["frames"].copy()
        if not np.array_equal(actions, data["actions"]):
            raise ValueError("Comparison controls do not match")
    if len(before) != 480 or before.shape != after.shape:
        raise ValueError("Expected matching30-second clips")
    torch.set_num_threads(4)
    model = DisplayUpscaler.from_weights(upscaler)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)
    with (
        imageio.get_writer(
            output / "comparison.mp4",
            fps=16,
            codec="libx264",
            quality=9,
            macro_block_size=2,
        ) as compare,
        imageio.get_writer(
            output / "showcase.mp4",
            fps=16,
            codec="libx264",
            quality=9,
            macro_block_size=2,
        ) as solo,
    ):
        for i in range(480):
            pixels = (
                torch.tensor(after[i], device="cuda").permute(2, 0, 1)[None].float()
                / 255
            )
            with torch.autocast("cuda", dtype=torch.float16):
                enlarged = model(pixels)
            enhanced = Image.fromarray(
                enlarged[0]
                .float()
                .mul(255)
                .round()
                .clamp(0, 255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            picture = Image.new("RGB", (1280, 480), "#10151e")
            draw = ImageDraw.Draw(picture)
            draw.text((16, 12), "BEFORE | Original model", font=font, fill="white")
            draw.text(
                (656, 12), "AFTER | Motion-stabilized model", font=font, fill="#c8edb2"
            )
            picture.paste(
                Image.fromarray(before[i]).resize((640, 352), Image.Resampling.BICUBIC),
                (0, 48),
            )
            picture.paste(enhanced, (640, 48))
            draw.text(
                (16, 416),
                controls(i)[0] + f"  |  {(i+1)/16:.1f}s / 30s",
                font=font,
                fill="white",
            )
            draw.text(
                (656, 416),
                "Slower movement (35%) · Real-ESRGAN 4x display",
                font=small,
                fill="#c8edb2",
            )
            draw.text(
                (16, 448),
                "Same starting view and controls. 480 generated frames. No resets or recorded future frames.",
                font=small,
                fill="#a4adbb",
            )
            compare.append_data(np.asarray(picture))
            single = Image.new("RGB", (960, 624), "#10151e")
            draw = ImageDraw.Draw(single)
            draw.text(
                (18, 14),
                "COUNTERDREAM | Stabilized world-model demo",
                font=font,
                fill="white",
            )
            single.paste(enhanced.resize((960, 528), Image.Resampling.BICUBIC), (0, 48))
            draw.text(
                (18, 586),
                controls(i)[0] + f"  |  {(i+1)/16:.1f}s",
                font=font,
                fill="#c8edb2",
            )
            draw.text(
                (485, 589),
                "Slower motion · 4x display · no resets",
                font=small,
                fill="#a4adbb",
            )
            solo.append_data(np.asarray(single))
            if i in (31, 95, 239, 479):
                picture.save(output / f"comparison-{i+1}.png")
    reports = {
        name: json.loads((Path(folder) / "report.json").read_text())
        for name, folder in [("baseline", baseline), ("candidate", candidate)]
    }
    report = dict(
        **reports,
        display_upscaler_sha256=sha256(upscaler),
        frames=480,
        fps=16,
        seconds=30,
        identical_controls=True,
        display_only_processing=True,
        files={
            name: sha256(output / name) for name in ("comparison.mp4", "showcase.mp4")
        },
    )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["files"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("baseline", "candidate", "upscaler", "output"):
        p.add_argument("--" + name, required=True)
    render(**vars(p.parse_args()))
