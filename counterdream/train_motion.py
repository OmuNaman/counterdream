"""Bounded local-GPU motion-predictor experiment on training-only transitions."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .motion_model import MotionPredictor


def train(data, output, seconds=600, steps=5000):
    if not 60 <= seconds <= 900 or not 1 <= steps <= 8000:
        raise ValueError("Motion experiment exceeds local bounds")
    torch.set_num_threads(4)
    torch.manual_seed(72)
    torch.backends.cudnn.benchmark = True
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((data / "report.json").read_text())
    assert report["split"] == "train" and report["samples"] == 4096
    for name, spec in report["files"].items():
        digest = hashlib.sha256((data / name).read_bytes()).hexdigest()
        if digest != spec["sha256"]:
            raise ValueError("Motion data checksum mismatch: " + name)
    observations = torch.from_numpy(
        np.load(data / "observations.npy", allow_pickle=False)
    ).to("cuda")
    actions = torch.from_numpy(np.load(data / "actions.npy", allow_pickle=False)).to(
        "cuda"
    )
    flows = torch.from_numpy(np.load(data / "flows.npy", allow_pickle=False)).to("cuda")
    confidence = (
        torch.from_numpy(np.load(data / "confidence.npy", allow_pickle=False))
        .to("cuda")
        .float()
        / 255
    )
    sources = json.loads((data / "samples.json").read_text())
    held = [
        int(hashlib.sha256(s["source"].encode()).hexdigest()[:8], 16) % 10 == 0
        for s in sources
    ]
    train_ids = torch.tensor([i for i, x in enumerate(held) if not x], device="cuda")
    val_ids = torch.tensor([i for i, x in enumerate(held) if x], device="cuda")
    assert len(train_ids) > 3000 and len(val_ids) > 100
    model = MotionPredictor().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.001)
    started = time.monotonic()
    best = float("inf")
    history = []
    for step in range(1, steps + 1):
        model.train()
        ids = train_ids[torch.randint(len(train_ids), (48,), device="cuda")]
        obs = observations[ids].float() / 127.5 - 1
        act = actions[ids]
        target_flow = flows[ids].float().clamp(-24, 24)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted, flow, residual = model(obs[:, :2], act)
            flow_error = F.smooth_l1_loss(flow.float(), target_flow, reduction="none")
            flow_loss = (flow_error * confidence[ids]).mean()
            photo = (predicted.float() - obs[:, 2]).abs().mean()
            smooth = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean() + (
                flow[:, :, 1:] - flow[:, :, :-1]
            ).abs().mean()
            detail = (
                (
                    (predicted[:, :, :, 1:] - predicted[:, :, :, :-1])
                    - (obs[:, 2, :, :, 1:] - obs[:, 2, :, :, :-1])
                )
                .abs()
                .mean()
            )
            loss = (
                photo
                + 0.07 * flow_loss
                + 0.002 * smooth
                + 0.08 * detail
                + 0.03 * residual.abs().mean()
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == steps or time.monotonic() - started > seconds:
            model.eval()
            metrics = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for ids in val_ids.split(48):
                    obs = observations[ids].float() / 127.5 - 1
                    pred, flow, _ = model(obs[:, :2], actions[ids])
                    shuffled, _, _ = model(obs[:, :2], actions[ids].roll(1, 0))
                    metrics.append(
                        torch.stack(
                            (
                                ((pred - obs[:, 2]) / 2).square().mean(),
                                ((obs[:, 1] - obs[:, 2]) / 2).square().mean(),
                                ((shuffled - obs[:, 2]) / 2).square().mean(),
                            )
                        )
                    )
            values = torch.stack(metrics).mean(0).float().cpu().tolist()
            row = dict(
                step=step,
                seconds=round(time.monotonic() - started, 1),
                loss=float(loss),
                validation_mse=values[0],
                repeat_mse=values[1],
                shuffled_action_mse=values[2],
            )
            history.append(row)
            if values[0] < best:
                best = values[0]
                torch.save(
                    dict(
                        model=model.state_dict(),
                        step=step,
                        architecture="MotionPredictor-v1",
                        data=report,
                        initialization="random",
                        pretrained_weights=False,
                    ),
                    output / "model.pt",
                )
            (output / "metrics.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            print(json.dumps(row), flush=True)
        if time.monotonic() - started > seconds:
            break
    summary = dict(
        step=step,
        seconds=time.monotonic() - started,
        best_validation_mse=best,
        train_samples=len(train_ids),
        internal_holdout_samples=len(val_ids),
        split="Only original training records; internal holdout grouped by source episode",
        parameters=sum(p.numel() for p in model.parameters()),
        gpu=torch.cuda.get_device_name(),
        peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    (output / "complete.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--steps", type=int, default=5000)
    train(**vars(parser.parse_args()))
