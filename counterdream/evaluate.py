"""Held-out evaluation and honest side-by-side autoregressive rollout artifacts."""

from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
from PIL import Image, ImageDraw
import torch
import imageio.v2 as imageio

from .actions import encode
from .data import Replay
from .model import load_model
from .train import DeviceReplay, atomic_json, validate


def uint8(x):
    return (
        x.detach()
        .float()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def panel(images, labels, scale=3):
    h, w = images[0].shape[:2]
    canvas = Image.new("RGB", (w * scale * len(images), (h * scale) + 40), "#111722")
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(images, labels)):
        canvas.paste(
            Image.fromarray(im).resize(
                (w * scale, h * scale), Image.Resampling.NEAREST
            ),
            (w * scale * i, 40),
        )
        draw.text((12 + w * scale * i, 13), label, fill="#ecedf0")
    return np.asarray(canvas)


@torch.no_grad()
def evaluate(checkpoint, data_root, output, steps=8, clips=4, frames=64):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    model, ckpt = load_model(checkpoint, device)
    torch.save({k: ckpt[k] for k in ("config", "ema", "step", "run")}, out / "model.pt")
    replay = Replay(data_root, "val")
    val = DeviceReplay(replay, device)
    result = validate(model, val, device, batches=16, batch_size=16, steps=steps)
    result.update(
        checkpoint_step=ckpt["step"],
        sampler_steps=steps,
        sampler_sigma_max=5.0 if model.cfg.version == 1 else 20.0,
        initialization=ckpt.get("run", {}).get("initialization"),
        pretrained_weights=False,
        checkpoint_sha256=hashlib.sha256((out / "model.pt").read_bytes()).hexdigest(),
        torch_version=str(torch.__version__),
        evaluation_source_sha256={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("model.py", "train.py", "evaluate.py", "data.py")
        },
    )
    horizons = {i: [] for i in (1, 4, 8, 16, 32, 64) if i <= frames}
    repeat = {i: [] for i in horizons}
    seed_frames, seed_actions, seed_names = [], [], []
    durations = []
    for clip in range(clips):
        ep = clip % len(replay.frames)
        start = 120 + clip * 71
        obs = (
            replay.frames[ep][start : start + 4 + frames].to(device).float() / 127.5 - 1
        )
        acts = replay.actions[ep][start : start + 3 + frames].to(device)
        seed_frames.append(
            replay.frames[ep][start : start + 4].permute(0, 2, 3, 1).numpy()
        )
        seed_actions.append(replay.actions[ep][start : start + 3].numpy())
        seed_names.append(f"Holdout {ep + 1} / frame {start}")
        context = obs[:4][None]
        sequence = []
        contact = []
        for t in range(frames):
            now = time.perf_counter()
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
            ):
                prediction = model.sample(
                    context,
                    acts[t : t + 4][None],
                    steps=steps,
                    seed=10000 + clip * 1000 + t,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - now)
            gt = obs[t + 4]
            image = panel(
                [uint8(gt), uint8(prediction[0]), uint8(obs[3])],
                [
                    "HELD-OUT GAMEPLAY",
                    "MODEL: OWN FRAMES FED BACK",
                    "REPEAT LAST SEED FRAME",
                ],
            )
            sequence.append(image)
            if t in (0, 3, 7, 15, 31, 63):
                contact.append(Image.fromarray(image))
            if t + 1 in horizons:
                horizons[t + 1].append(
                    float(((prediction[0] - gt) / 2).square().mean())
                )
                repeat[t + 1].append(float(((obs[3] - gt) / 2).square().mean()))
            context = torch.cat((context[:, 1:], prediction[:, None]), dim=1)
        imageio.mimwrite(
            out / f"rollout-{clip + 1}.mp4",
            sequence,
            fps=8,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        )
        sheet = Image.new("RGB", (contact[0].width, contact[0].height * len(contact)))
        for i, row in enumerate(contact):
            sheet.paste(row, (0, i * row.height))
        sheet.save(out / f"rollout-{clip + 1}.png")
    np.savez_compressed(
        out / "seeds.npz",
        frames=np.stack(seed_frames),
        actions=np.stack(seed_actions),
        names=np.array(seed_names),
    )
    result["rollout"] = {
        str(h): dict(
            mse=float(np.mean(v)),
            psnr=-10 * math.log10(max(np.mean(v), 1e-12)),
            repeat_frame_mse=float(np.mean(repeat[h])),
            clips=len(v),
        )
        for h, v in horizons.items()
    }
    result["inference_ms_median"] = float(np.median(durations) * 1000)
    result["inference_fps_median"] = float(1 / np.median(durations))
    result["gpu"] = torch.cuda.get_device_name() if device == "cuda" else "CPU"
    # Counterfactual action test: identical visual history and identical random noise.
    context = (
        torch.as_tensor(np.stack(seed_frames[:1]), device=device)
        .permute(0, 1, 4, 2, 3)
        .float()
        / 127.5
        - 1
    )
    past = torch.as_tensor(np.stack(seed_actions[:1]), device=device)
    branches = []
    labels = []
    for name, dx in [("TURN LEFT", -60), ("IDLE", 0), ("TURN RIGHT", 60)]:
        c = context.clone()
        a = past.clone()
        for t in range(12):
            action = torch.from_numpy(encode(dx=dx)).to(device)[None, None]
            full = torch.cat((a, action), dim=1)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
            ):
                p = model.sample(c, full, steps=steps, seed=777 + t)
            c = torch.cat((c[:, 1:], p[:, None]), dim=1)
            a = full[:, 1:]
        branches.append(uint8(p[0]))
        labels.append(name + " / 12 STEPS")
    Image.fromarray(panel(branches, labels, scale=4)).save(out / "counterfactual.png")
    result["counterfactual_pixel_difference_left_right"] = float(
        np.abs(branches[0].astype(float) - branches[2].astype(float)).mean() / 255
    )
    atomic_json(out / "evaluation.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    evaluate(a.checkpoint, a.data, a.output)
