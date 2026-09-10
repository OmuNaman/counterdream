"""Record a repeatable, uninterrupted world-model control sequence."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from .actions import encode
from .model import load_model


def controls(frame):
    # Thirty seconds at sixteen generated frames/second. No hidden resets.
    schedule = [
        (32, "LOOK AROUND", dict(dx=4)),
        (96, "FORWARD · W", dict(keys=["w"])),
        (144, "BACKWARD · S", dict(keys=["s"])),
        (176, "LOOK LEFT", dict(dx=-4)),
        (208, "LOOK RIGHT", dict(dx=4)),
        (240, "FIRE", dict(fire=True)),
        (288, "FORWARD + FIRE", dict(keys=["w"], fire=True)),
        (336, "BACKWARD · S", dict(keys=["s"])),
        (368, "STRAFE LEFT · A", dict(keys=["a"])),
        (400, "STRAFE RIGHT · D", dict(keys=["d"])),
        (416, "RELOAD · R", dict(keys=["r"])),
        (432, "JUMP", dict(keys=["space"] if frame == 416 else [])),
        (480, "FORWARD · W", dict(keys=["w"])),
    ]
    for end, label, action in schedule:
        if frame < end:
            return label, action
    return "IDLE", {}


def showcase_controls(frame):
    """Gentle, explicitly curated control sequence; still one continuous rollout."""
    cycle = frame % 96
    if cycle < 24:
        return "FORWARD · W", dict(keys=["w"])
    if cycle < 48:
        return "BACKWARD · S", dict(keys=["s"])
    if cycle < 56:
        return "LOOK LEFT", dict(dx=-2)
    if cycle < 72:
        return "LOOK RIGHT", dict(dx=2)
    if cycle < 80:
        return "LOOK LEFT", dict(dx=-2)
    if cycle < 84:
        return "FIRE · short burst", dict(fire=True)
    if cycle < 88:
        return "RELOAD · R", dict(keys=["r"])
    return "IDLE", {}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_frame(frame, label, index, title, total=480):
    image = Image.new("RGB", (960, 624), "#10151e")
    image.paste(
        Image.fromarray(frame).resize((960, 528), Image.Resampling.BICUBIC), (0, 48)
    )
    draw = ImageDraw.Draw(image)
    draw.text((20, 17), title, fill="#ffffff")
    draw.text((20, 590), label, fill="#9cdbcb")
    draw.text((680, 590), f"{index + 1}/{total} frames | no resets", fill="#c1c8d2")
    draw.rectangle((0, 620, round(960 * (index + 1) / total), 623), fill="#9cdbcb")
    return np.asarray(image)


@torch.inference_mode()
def record(
    checkpoint,
    seeds,
    output,
    steps=8,
    spawn=0,
    frames=480,
    profile=None,
    sequence="stress",
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    control_sequence = showcase_controls if sequence == "showcase" else controls
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    model, ckpt = load_model(checkpoint, "cuda")
    options = None
    settings = json.loads(Path(profile).read_text()) if profile else {}
    if settings:
        from .demo_sampling import SamplingOptions

        options = SamplingOptions(**settings["sampling"])
    upscaler = None
    if settings.get("upscaler"):
        from .upscale import DisplayUpscaler

        upscaler = DisplayUpscaler.from_weights(settings["upscaler"])
    motion = None
    motion_state = None
    if settings.get("motion_checkpoint"):
        from .motion_model import MotionPredictor

        motion = MotionPredictor().to("cuda").eval()
        motion_state = torch.load(
            settings["motion_checkpoint"], map_location="cpu", weights_only=True
        )
        motion.load_state_dict(motion_state["model"])
    with np.load(seeds, allow_pickle=False) as data:
        context = (
            torch.from_numpy(data["frames"][spawn : spawn + 1].copy())
            .to("cuda")
            .permute(0, 1, 4, 2, 3)
            .float()
            / 127.5
            - 1
        )
        history = torch.from_numpy(data["actions"][spawn : spawn + 1].copy()).to("cuda")
        spawn_name = str(data["names"][spawn])
    clip, actions, elapsed = [], [], []
    title = f'UNMODIFIED BASELINE | CounterDream {ckpt["step"]:,} steps | {steps} sampling passes'
    if options:
        title = f'INFERENCE UPGRADE | CounterDream {ckpt["step"]:,} steps | {options.steps} passes'
        if settings.get("motion_only"):
            title = f'MOTION EXPERIMENT | {motion_state["step"]:,} training steps | no diffusion denoising'
        if upscaler is not None:
            title += " | Real-ESRGAN 4x display"
    video_name = "demo.mp4" if options else "baseline.mp4"
    start = time.monotonic()
    with imageio.get_writer(
        output / video_name, fps=16, codec="libx264", quality=9, macro_block_size=2
    ) as writer:
        for i in range(frames):
            label, action = control_sequence(i)
            encoded = encode(**action)
            actions.append(encoded)
            current = torch.from_numpy(encoded).to("cuda")[None, None]
            full_history = torch.cat((history, current), 1)
            tick = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                initial_image = None
                if motion is not None:
                    from .motion_model import guided_motion

                    initial_image = guided_motion(
                        motion,
                        context[:, -2:],
                        full_history[:, -2:],
                        scale=settings.get("motion_scale", 1.0),
                        center=settings.get("motion_center", False),
                    )
                if options:
                    from .demo_sampling import sample

                    if settings.get("motion_only"):
                        predicted = initial_image
                    else:
                        predicted = sample(
                            model,
                            context,
                            full_history,
                            options,
                            seed=1000 + i,
                            initial_image=initial_image,
                        )
                else:
                    predicted = model.sample(
                        context, full_history, steps=steps, seed=1000 + i
                    )
            if settings.get("temporal_detail", 0):
                from .temporal_detail import restore_detail

                predicted = restore_detail(
                    context[:, -1], predicted, settings["temporal_detail"]
                )
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - tick)
            frame = (
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
            clip.append(frame)
            display = frame
            if upscaler is not None:
                with torch.autocast("cuda", dtype=torch.float16):
                    enlarged = upscaler(predicted.float().add(1).mul(0.5))
                display = (
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
            writer.append_data(display_frame(display, label, i, title, frames))
            context = torch.cat((context[:, 1:], predicted[:, None]), 1)
            history = full_history[:, 1:]
            if (i + 1) % 64 == 0:
                print(
                    json.dumps(
                        dict(
                            frame=i + 1,
                            total=frames,
                            seconds=round(time.monotonic() - start, 1),
                        )
                    ),
                    flush=True,
                )
    np.savez_compressed(
        output / "native-frames.npz", frames=np.stack(clip), actions=np.stack(actions)
    )
    # Contact sheet preserves consecutive-run timestamps; no image enhancement.
    moments = [min(i, frames - 1) for i in [0, 31, 95, 143, 239, 287, 399, frames - 1]]
    sheet = Image.new("RGB", (640, 4 * 208), "#10151e")
    for j, i in enumerate(moments):
        cell = Image.new("RGB", (320, 208), "#10151e")
        cell.paste(Image.fromarray(clip[i]).resize((320, 176)), (0, 24))
        ImageDraw.Draw(cell).text(
            (6, 6), f"{(i+1)/16:.1f}s | {control_sequence(i)[0]}", fill="white"
        )
        sheet.paste(cell, ((j % 2) * 320, (j // 2) * 208))
    sheet.save(output / "contact-sheet.png")
    report = dict(
        checkpoint_step=ckpt["step"],
        checkpoint_sha256=sha256(checkpoint),
        seeds_sha256=sha256(seeds),
        spawn=spawn,
        spawn_name=spawn_name,
        frames=frames,
        playback_fps=16,
        duration_seconds=frames / 16,
        sampling_steps=steps,
        sigma_max=20.0,
        random_seed_start=1000,
        generation_ms_median=float(np.median(elapsed) * 1000),
        gpu=torch.cuda.get_device_name(),
        model_source_sha256=sha256(Path(__file__).with_name("model.py")),
        resets_after_initial_seed=0,
        recorded_frames_injected_after_seed=0,
        display="Bicubic enlargement only; native frames also saved",
        action_script_sha256=sha256(__file__),
        video_sha256=sha256(output / video_name),
        profile=settings,
        native_frames_sha256=sha256(output / "native-frames.npz"),
        control_sequence=sequence,
        controls_sha256=hashlib.sha256(np.stack(actions).tobytes()).hexdigest(),
    )
    if options:
        report.update(
            sampling_steps=options.steps,
            sigma_max=options.sigma_max,
            display=(
                "Real-ESRGAN 4x pretrained display upscaler"
                if upscaler is not None
                else "Bicubic enlargement"
            ),
            world_model_pretrained=False,
            display_upscaler_pretrained=upscaler is not None,
            upscaler_fed_to_model=False,
        )
        if motion is not None:
            report.update(
                motion_checkpoint_sha256=sha256(settings["motion_checkpoint"]),
                motion_checkpoint_step=motion_state["step"],
                diffusion_used=not settings.get("motion_only", False),
                generation=(
                    "Learned action-conditioned motion"
                    if settings.get("motion_only")
                    else "Learned motion guiding diffusion denoising"
                ),
            )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=8, choices=(4, 8, 16))
    parser.add_argument("--spawn", type=int, default=0)
    parser.add_argument("--frames", type=int, default=480)
    parser.add_argument("--profile")
    parser.add_argument("--sequence", choices=("stress", "showcase"), default="stress")
    args = parser.parse_args()
    record(**vars(args))
