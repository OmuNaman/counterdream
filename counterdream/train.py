"""Reproducible training with EMA, checkpoints, holdout metrics, and a wall-time limit."""

import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .data import Replay
from .model import ModelConfig, WorldModel


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


class DeviceReplay:
    def __init__(self, replay, device):
        self.context = replay.context
        self.lengths = replay.lengths
        self.offsets = np.concatenate(([0], np.cumsum(self.lengths)[:-1]))
        self.frames = torch.cat(replay.frames).to(device)
        self.actions = torch.cat(replay.actions).to(device)
        self.device = device

    def batch(self, batch_size, rng, horizon=1):
        ep = rng.integers(len(self.lengths), size=batch_size)
        starts = (
            rng.random(batch_size) * (self.lengths[ep] - self.context - horizon + 1)
        ).astype(np.int64)
        starts = torch.as_tensor(starts + self.offsets[ep], device=self.device)
        indices = (
            starts[:, None]
            + torch.arange(self.context + horizon, device=self.device)[None, :]
        )
        return self.frames[indices].float() / 127.5 - 1, self.actions[indices[:, :-1]]


@torch.no_grad()
def validate(model, replay, device, batches=4, batch_size=16):
    rng = np.random.default_rng(90210)
    total_mse = 0.0
    repeat_mse = 0.0
    shuffled_mse = 0.0
    denoise_loss = 0.0
    model.eval()
    for i in range(batches):
        obs, acts = replay.batch(batch_size, rng)
        context, target = obs[:, :-1], obs[:, -1]
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=str(device).startswith("cuda"),
        ):
            pred = model.sample(context, acts, steps=8, seed=1234 + i)
            shuffled = model.sample(context, acts.roll(1, 0), steps=8, seed=1234 + i)
            # fixed denoising corruption for comparable validation checkpoints
            gen = torch.Generator(device=device).manual_seed(850 + i)
            noise = torch.randn(target.shape, device=device, generator=gen)
            denoised = model(
                target + noise * 0.5,
                torch.full((batch_size,), 0.5, device=device),
                context,
                acts,
            )
        total_mse += float(((pred - target) / 2).square().mean())
        repeat_mse += float(((context[:, -1] - target) / 2).square().mean())
        shuffled_mse += float(((shuffled - target) / 2).square().mean())
        denoise_loss += float(((denoised - target) / 2).square().mean())
    mse = total_mse / batches
    return dict(
        next_frame_mse=mse,
        next_frame_psnr=-10 * math.log10(max(mse, 1e-12)),
        repeat_frame_mse=repeat_mse / batches,
        repeat_frame_psnr=-10 * math.log10(max(repeat_mse / batches, 1e-12)),
        shuffled_action_mse=shuffled_mse / batches,
        action_advantage_mse=(shuffled_mse - total_mse) / batches,
        fixed_noise_mse=denoise_loss / batches,
        validation_frames=batches * batch_size,
    )


def train(
    data_root,
    output,
    steps=30000,
    batch_size=48,
    max_seconds=7200,
    seed=42,
    resume=None,
    base=64,
    commit=None,
    source=None,
):
    if not 1 <= max_seconds <= 14400:
        raise ValueError("Run limit must be 1–14400 seconds")
    if steps < 1 or batch_size < 1:
        raise ValueError("Steps and batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Use a CUDA GPU for training")
    started = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    rng = np.random.default_rng(seed)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig(base=base)
    model = WorldModel(cfg).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95), fused=True
    )
    begin = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        if ckpt["config"] != asdict(cfg):
            raise ValueError("Resume configuration differs")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        begin = ckpt["step"]
        rng.bit_generator.state = ckpt["numpy_rng"]
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())
    if begin >= steps:
        raise ValueError(
            f"Checkpoint is already at step {begin}; target must be larger"
        )
    dataset = DeviceReplay(Replay(data_root, "train"), device)
    valset = DeviceReplay(Replay(data_root, "val"), device)
    details = dict(
        config=asdict(cfg),
        seed=seed,
        parameters=sum(p.numel() for p in model.parameters()),
        initialization="random",
        resumed_from_step=begin,
        pretrained_weights=False,
        gpu=torch.cuda.get_device_name(),
        torch_version=str(torch.__version__),
        train_frames=int(dataset.lengths.sum()),
        val_frames=int(valset.lengths.sum()),
        planned_steps=steps,
        max_seconds=max_seconds,
        batch_size=batch_size,
        source=source or {},
    )
    manifest_path = Path(data_root) / "manifest.json"
    if manifest_path.exists():
        shutil.copyfile(manifest_path, root / "data_manifest.json")
    atomic_json(root / "run.json", details)
    print(json.dumps(details), flush=True)
    best = float("inf")
    if resume and (root / "metrics.jsonl").exists():
        old_metrics = [
            json.loads(line)
            for line in (root / "metrics.jsonl").read_text().splitlines()
            if line
        ]
        if old_metrics and (root / "best.pt").exists():
            best = min(m["next_frame_mse"] for m in old_metrics)

    def save(step, name):
        checkpoint = dict(
            config=asdict(cfg),
            model=model.state_dict(),
            ema=ema.state_dict(),
            optimizer=optimizer.state_dict(),
            step=step,
            seed=seed,
            numpy_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),
            run=details,
        )
        temp = root / (name + ".tmp")
        torch.save(checkpoint, temp)
        temp.replace(root / name)

    # Record an honest random-initialization baseline before any optimizer steps.
    if not resume:
        initial = validate(ema, valset, device, batches=2)
        atomic_json(root / "initial_metrics.json", initial)
    save(begin, "latest.pt")
    if commit:
        commit()
    smooth = None
    for step in range(begin + 1, steps + 1):
        model.train()
        lr = (
            2e-4
            * min(step / 200, 1)
            * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * min(step / steps, 1))))
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        # Short unrolling exposes the model to its own denoised context.
        horizon = 2 if step > 2000 and step % 2 == 0 else 1
        obs, acts = dataset.batch(batch_size, rng, horizon=horizon)
        context = obs[:, : cfg.context]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, pred = model.loss(
                context, acts[:, : cfg.context], obs[:, cfg.context]
            )
            if horizon == 2:
                context2 = torch.cat((context[:, 1:], pred.detach()[:, None]), dim=1)
                loss2, _ = model.loss(
                    context2, acts[:, 1 : cfg.context + 1], obs[:, cfg.context + 1]
                )
                loss = (loss + loss2) / 2
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at {step}")
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        decay = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            torch._foreach_lerp_(
                list(ema.parameters()), list(model.parameters()), 1 - decay
            )
        value = float(loss.detach())
        smooth = value if smooth is None else 0.98 * smooth + 0.02 * value
        elapsed = time.time() - started
        if step % 100 == 0 or step == 1:
            state = dict(
                step=step,
                loss=value,
                smoothed_loss=smooth,
                grad_norm=grad,
                seconds=elapsed,
                steps_per_second=(step - begin) / elapsed,
                gpu_allocated_gb=torch.cuda.memory_allocated() / 1e9,
            )
            atomic_json(root / "progress.json", state)
            print(json.dumps(state), flush=True)
        stopping = elapsed > max_seconds - 90 or step == steps
        if step % 1000 == 0 or stopping:
            metrics = validate(ema, valset, device)
            metrics.update(step=step, seconds=time.time() - started, train_loss=smooth)
            with (root / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            atomic_json(root / "metrics.json", metrics)
            save(step, "latest.pt")
            if metrics["next_frame_mse"] < best:
                best = metrics["next_frame_mse"]
                save(step, "best.pt")
            if commit:
                commit()
            print("VALIDATION " + json.dumps(metrics), flush=True)
        if stopping:
            break
    # Publish a lightweight inference-only checkpoint separately from optimizer state.
    best_ckpt = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    torch.save(
        {k: best_ckpt[k] for k in ("config", "ema", "step", "run")}, root / "model.pt"
    )
    atomic_json(
        root / "complete.json",
        dict(
            step=step,
            seconds=time.time() - started,
            best_step=best_ckpt["step"],
            best_mse=best,
            stopped_by_time=step < steps,
        ),
    )
    if commit:
        commit()
    return json.loads((root / "complete.json").read_text())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--max-seconds", type=int, default=7200)
    p.add_argument("--resume")
    a = p.parse_args()
    train(a.data, a.output, steps=a.steps, max_seconds=a.max_seconds, resume=a.resume)
