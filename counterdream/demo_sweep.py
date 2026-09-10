"""Save every inference trial so selection can be visually audited."""

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from .actions import encode
from .demo_sampling import SamplingOptions, sample
from .model import load_model


ROOT = Path("artifacts/dust2-v3-full")


def trial_controls(frame):
    segments = [
        ("FORWARD", dict(keys=["w"])),
        ("BACKWARD", dict(keys=["s"])),
        ("LEFT", dict(dx=-4)),
        ("RIGHT", dict(dx=4)),
        ("FIRE", dict(fire=True)),
        ("FORWARD + FIRE", dict(keys=["w"], fire=True)),
        ("BACKWARD", dict(keys=["s"])),
        ("IDLE", {}),
    ]
    return segments[min(frame // 16, 7)]


@torch.inference_mode()
def main(phase=1):
    output = Path(f"artifacts/demo-upgrade/sweep-{phase:02d}")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    seeds_path = ROOT / "final-step-60000/evaluation-val-4-step-60000/seeds.npz"
    with np.load(seeds_path, allow_pickle=False) as data:
        seeds = (
            torch.from_numpy(data["frames"][:1].copy())
            .to("cuda")
            .permute(0, 1, 4, 2, 3)
            .float()
            / 127.5
            - 1
        )
        seed_actions = torch.from_numpy(data["actions"][:1].copy()).to("cuda")
    configurations = [
        ("10k-euler4", 10000, SamplingOptions(steps=4)),
        ("20k-euler8", 20000, SamplingOptions()),
        ("20k-heun4", 20000, SamplingOptions(steps=4, solver="heun")),
        ("20k-context003", 20000, SamplingOptions(context_noise=0.03)),
        ("20k-context010", 20000, SamplingOptions(context_noise=0.1)),
        (
            "20k-warm05",
            20000,
            SamplingOptions(steps=4, sigma_max=0.5, solver="heun", warm_start=True),
        ),
        (
            "20k-warm15",
            20000,
            SamplingOptions(steps=4, sigma_max=1.5, solver="heun", warm_start=True),
        ),
        (
            "20k-warm3",
            20000,
            SamplingOptions(steps=4, sigma_max=3.0, solver="heun", warm_start=True),
        ),
        ("60k-heun4", 60000, SamplingOptions(steps=4, solver="heun")),
    ]
    if phase == 2:
        configurations = [
            (
                "20k-warm03-euler",
                20000,
                SamplingOptions(steps=4, sigma_max=0.3, warm_start=True),
            ),
            (
                "20k-warm05-euler",
                20000,
                SamplingOptions(steps=4, sigma_max=0.5, warm_start=True),
            ),
            (
                "20k-warm15-euler",
                20000,
                SamplingOptions(steps=8, sigma_max=1.5, warm_start=True),
            ),
            ("20k-unitnoise", 20000, SamplingOptions(initial_noise_scale=0.05)),
            (
                "20k-unitnoise-context",
                20000,
                SamplingOptions(initial_noise_scale=0.05, context_noise=0.03),
            ),
            ("20k-lowtemp", 20000, SamplingOptions(initial_noise_scale=0.25)),
            ("20k-heun16", 20000, SamplingOptions(steps=16, solver="heun")),
            ("10k-context003", 10000, SamplingOptions(steps=8, context_noise=0.03)),
        ]
    if phase == 3:
        configurations = [
            ("20k-detail035", 20000, SamplingOptions(context_noise=0.03)),
            ("20k-detail065", 20000, SamplingOptions(context_noise=0.03)),
            (
                "20k-detail035-sigma5",
                20000,
                SamplingOptions(sigma_max=5, context_noise=0.03),
            ),
            (
                "20k-euler16-context003",
                20000,
                SamplingOptions(steps=16, context_noise=0.03),
            ),
        ]
    model = loaded_step = None
    reports = []
    for name, step, options in configurations:
        if step != loaded_step:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            checkpoint = (
                ROOT / "early-step-10000/evaluation-val-4-step-10000/model.pt"
                if step == 10000
                else (
                    ROOT / "early-step-20000/evaluation-val-8-step-20000/model.pt"
                    if step == 20000
                    else ROOT / "final-step-60000/evaluation-val-4-step-60000/model.pt"
                )
            )
            model, _ = load_model(checkpoint, "cuda")
            loaded_step = step
        context, history = seeds.clone(), seed_actions.clone()
        frames, durations = [], []
        for i in range(128):
            _, command = trial_controls(i)
            act = torch.from_numpy(encode(**command)).to("cuda")[None, None]
            full_history = torch.cat((history, act), 1)
            tick = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = sample(model, context, full_history, options, 1000 + i)
            if "detail" in name:
                from .temporal_detail import restore_detail

                strength = 0.65 if "065" in name else 0.35
                predicted = restore_detail(context[:, -1], predicted, strength)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - tick)
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
            history = full_history[:, 1:]
        arrays = np.stack(frames)
        np.savez_compressed(output / (name + ".npz"), frames=arrays)
        with imageio.get_writer(
            output / (name + ".mp4"),
            fps=16,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        ) as writer:
            for i, frame in enumerate(frames):
                cell = Image.new("RGB", (640, 392), "#10151e")
                cell.paste(Image.fromarray(frame).resize((640, 352)), (0, 40))
                ImageDraw.Draw(cell).text(
                    (12, 12),
                    name + " | " + trial_controls(i)[0] + " | frame " + str(i + 1),
                    fill="white",
                )
                writer.append_data(np.asarray(cell))
        sheet = Image.new("RGB", (1280, 416), "#10151e")
        for j, index in enumerate([0, 15, 31, 47, 63, 79, 95, 127]):
            cell = Image.new("RGB", (320, 208), "#10151e")
            cell.paste(Image.fromarray(frames[index]).resize((320, 176)), (0, 26))
            ImageDraw.Draw(cell).text(
                (6, 6), name + " | " + str(index + 1), fill="white"
            )
            sheet.paste(cell, ((j % 4) * 320, (j // 4) * 208))
        sheet.save(output / (name + ".png"))
        values = arrays.astype(np.float32) / 255
        gradient = np.mean(np.abs(np.diff(values, axis=2)), axis=(1, 2, 3)) + np.mean(
            np.abs(np.diff(values, axis=1)), axis=(1, 2, 3)
        )
        report = dict(
            name=name,
            checkpoint_step=step,
            options=asdict(options),
            mean_frame_change=float(np.abs(np.diff(values, axis=0)).mean()),
            texture_first16=float(gradient[:16].mean()),
            texture_last16=float(gradient[-16:].mean()),
            median_ms=float(np.median(durations) * 1000),
            hidden_resets=0,
            future_recorded_frames=0,
        )
        reports.append(report)
        (output / "reports.json").write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), default=1)
    main(**vars(parser.parse_args()))
