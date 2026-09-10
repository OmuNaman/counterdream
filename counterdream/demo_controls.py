"""Matched-noise control check for the motion-guided demonstration."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .actions import encode
from .demo_sampling import SamplingOptions, sample
from .demo_record import sha256
from .model import load_model
from .motion_model import MotionPredictor, guided_motion


@torch.inference_mode()
def check(checkpoint, seeds, profile, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    model, _ = load_model(checkpoint, "cuda")
    settings = json.loads(Path(profile).read_text())
    options = SamplingOptions(**settings["sampling"])
    motion = MotionPredictor().cuda().eval()
    motion.load_state_dict(
        torch.load(
            settings["motion_checkpoint"], map_location="cpu", weights_only=True
        )["model"]
    )
    with np.load(seeds, allow_pickle=False) as data:
        original = (
            torch.tensor(data["frames"][0:1], device="cuda")
            .permute(0, 1, 4, 2, 3)
            .float()
            / 127.5
            - 1
        )
        original_actions = torch.tensor(data["actions"][0:1], device="cuda")
    controls = {
        "IDLE": {},
        "FORWARD": dict(keys=["w"]),
        "BACKWARD": dict(keys=["s"]),
        "LEFT": dict(dx=-10),
        "RIGHT": dict(dx=10),
        "FIRE": dict(fire=True),
    }
    clips = {}
    for label, action in controls.items():
        context, history = original.clone(), original_actions.clone()
        frames = []
        for i in range(32):
            actions = torch.cat(
                (history, torch.tensor(encode(**action), device="cuda")[None, None]), 1
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                initial = guided_motion(
                    motion,
                    context[:, -2:],
                    actions[:, -2:],
                    settings.get("motion_scale", 1.0),
                    settings.get("motion_center", False),
                )
                predicted = sample(
                    model,
                    context,
                    actions,
                    options,
                    seed=1000 + i,
                    initial_image=initial,
                )
            frames.append(
                predicted[0]
                .float()
                .add(1)
                .mul(127.5)
                .round()
                .clamp(0, 255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            context = torch.cat((context[:, 1:], predicted[:, None]), 1)
            history = actions[:, 1:]
        clips[label] = np.stack(frames)
        print(json.dumps(dict(control=label, frames=32)), flush=True)
    sheet = Image.new("RGB", (960, 4 * 112), "#10151e")
    for col, (label, frames) in enumerate(clips.items()):
        for row, index in enumerate((7, 15, 23, 31)):
            tile = Image.fromarray(frames[index])
            x = col * 160
            y = row * 112
            sheet.paste(tile, (x, y + 24))
            ImageDraw.Draw(sheet).text(
                (x + 4, y + 5), f"{label} | {(index+1)/16:.1f}s", fill="white"
            )
    sheet.save(output / "controls.png")
    np.savez_compressed(output / "frames.npz", **clips)
    metrics = {
        name: float(np.mean(((frames.astype(np.float32) - clips["IDLE"]) / 255) ** 2))
        for name, frames in clips.items()
    }
    report = dict(
        profile=settings,
        checkpoint_sha256=sha256(checkpoint),
        motion_sha256=sha256(settings["motion_checkpoint"]),
        seeds_sha256=sha256(seeds),
        frame_count=32,
        spawn=0,
        matched_noise=True,
        versus_idle_mse=metrics,
        interpretation="Image differences demonstrate action sensitivity, not correct game physics or hit detection.",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("checkpoint", "seeds", "profile", "output"):
        p.add_argument("--" + name, required=True)
    check(**vars(p.parse_args()))
