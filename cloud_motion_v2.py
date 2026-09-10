"""Bounded expert-action, multi-step motion experiment. No test split is read."""

import json
from pathlib import Path
import modal
from counterdream.cloud_config import gpu_base_image, volume, DATA, RUN

app = modal.App("counterdream-motion-v2")
image = gpu_base_image.pip_install(
    "opencv-python-headless==4.10.0.84"
).add_local_python_source("counterdream")
ROOT = Path(RUN, "motion-v2-experiment")


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=1000,
    retries=0,
    volumes={"/artifacts": volume},
    max_containers=1,
    scaledown_window=2,
)
def prepare():
    import hashlib
    import time
    import numpy as np
    import cv2
    from counterdream.scaled_data import DiskReplay

    volume.reload()
    root = ROOT / "data"
    if (root / "report.json").exists():
        return json.loads((root / "report.json").read_text())
    root.mkdir(parents=True, exist_ok=True)
    replay = DiskReplay(DATA, split="train", context=2)
    expert = [i for i, r in enumerate(replay.records) if r.get("expert")]
    other = [i for i, r in enumerate(replay.records) if not r.get("expert")]
    rng = np.random.default_rng(9102026)
    episodes = expert + rng.choice(other, min(64, len(other)), replace=False).tolist()
    if len(expert) < 100:
        raise ValueError("Expected prepared clean expert training episodes")
    count, length = len(episodes) * 24, 10
    observations = np.lib.format.open_memmap(
        root / "observations.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(count, length, 3, 88, 160),
    )
    actions = np.lib.format.open_memmap(
        root / "actions.npy", mode="w+", dtype=np.float32, shape=(count, length - 1, 51)
    )
    flows = np.lib.format.open_memmap(
        root / "flows.npy", mode="w+", dtype=np.float16, shape=(count, 2, 88, 160)
    )
    confidence = np.lib.format.open_memmap(
        root / "confidence.npy", mode="w+", dtype=np.uint8, shape=(count, 1, 88, 160)
    )
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    cv2.setNumThreads(2)
    yy, xx = np.mgrid[:88, :160].astype(np.float32)
    sources = []
    start_time = time.monotonic()
    for ei, episode in enumerate(episodes):
        if time.monotonic() - start_time > 850:
            raise TimeoutError("Data preparation exceeded its bound")
        frames, acts = replay.episode(episode)
        n = len(frames) - length + 1
        # Windows sample movement, fire, jump and reload transitions more often.
        weights = (
            1
            + acts[1 : n + 1, 2] * 2
            + acts[1 : n + 1, 11] * 2
            + acts[1 : n + 1, 4] * 3
            + acts[1 : n + 1, 10] * 3
        )
        starts = rng.choice(n, 24, replace=False, p=weights / weights.sum())
        for wi, start in enumerate(starts):
            i = ei * 24 + wi
            obs = frames[start : start + length].copy()
            act = acts[start : start + length - 1].copy()
            before, after = obs[1].transpose(1, 2, 0), obs[2].transpose(1, 2, 0)
            flow = estimator.calc(
                cv2.cvtColor(after, cv2.COLOR_RGB2GRAY),
                cv2.cvtColor(before, cv2.COLOR_RGB2GRAY),
                None,
            )
            warped = cv2.remap(
                before,
                xx + flow[:, :, 0],
                yy + flow[:, :, 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            error = np.abs(warped.astype(np.float32) - after).mean(2) / 255
            valid = (
                (xx + flow[:, :, 0] >= 0)
                & (xx + flow[:, :, 0] < 160)
                & (yy + flow[:, :, 1] >= 0)
                & (yy + flow[:, :, 1] < 88)
            )
            observations[i], actions[i], flows[i] = obs, act, flow.transpose(2, 0, 1)
            confidence[i, 0] = (
                (np.exp(-error / 0.1) * valid * 255).round().astype(np.uint8)
            )
            sources.append(
                dict(
                    source=replay.records[episode]["source"],
                    start=int(start),
                    expert=episode in expert,
                )
            )
        if ei % 16 == 0:
            print(
                json.dumps(
                    dict(
                        prepared=(ei + 1) * 24,
                        total=count,
                        seconds=time.monotonic() - start_time,
                    )
                ),
                flush=True,
            )
    for array in (observations, actions, flows, confidence):
        array.flush()
    (root / "samples.json").write_text(json.dumps(sources))
    report = dict(
        samples=count,
        frames_per_window=length,
        expert_episodes=len(expert),
        other_episodes=len(episodes) - len(expert),
        split="train",
        seed=9102026,
        dataset_index_sha256=hashlib.sha256(
            Path(DATA, "index.json").read_bytes()
        ).hexdigest(),
        seconds=time.monotonic() - start_time,
    )
    (root / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=1050,
    retries=0,
    volumes={"/artifacts": volume},
    max_containers=1,
    scaledown_window=2,
)
def fit():
    import hashlib
    import time
    import numpy as np
    import torch
    from torch.nn import functional as F
    from counterdream.motion_v2 import MotionV2

    volume.reload()
    output = ROOT / "fit"
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(93)
    torch.backends.cudnn.benchmark = True
    root = ROOT / "data"
    obs = torch.from_numpy(np.load(root / "observations.npy")).to("cuda")
    actions = torch.from_numpy(np.load(root / "actions.npy")).to("cuda")
    flows = torch.from_numpy(np.load(root / "flows.npy")).to("cuda")
    confidence = (
        torch.from_numpy(np.load(root / "confidence.npy")).to("cuda").float() / 255
    )
    sources = json.loads((root / "samples.json").read_text())
    held = np.array(
        [
            int(hashlib.sha256(s["source"].encode()).hexdigest()[:8], 16) % 10 == 0
            for s in sources
        ]
    )
    train_ids = torch.as_tensor(np.flatnonzero(~held), device="cuda")
    val_ids = torch.as_tensor(np.flatnonzero(held), device="cuda")
    # Fixed, episode-disjoint validation windows; final test remains untouched.
    val_ids = val_ids[
        torch.linspace(
            0, len(val_ids) - 1, min(128, len(val_ids)), device="cuda"
        ).long()
    ]
    model = MotionV2().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.001)
    started = time.monotonic()
    history = []
    best = float("inf")
    for step in range(1, 10001):
        model.train()
        ids = train_ids[torch.randint(len(train_ids), (12,), device="cuda")]
        observations = obs[ids].float() / 127.5 - 1
        act = actions[ids]
        context = observations[:, :2]
        optimizer.zero_grad(set_to_none=True)
        horizon = 1 if step < 200 else 4
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = 0
            for t in range(horizon):
                prediction, flow, residual = model(context, act[:, t : t + 2])
                target = observations[:, t + 2]
                photo = (prediction.float() - target).abs().mean()
                coarse = (
                    (F.avg_pool2d(prediction.float(), 4) - F.avg_pool2d(target, 4))
                    .abs()
                    .mean()
                )
                detail = (
                    (
                        torch.diff(prediction.float(), dim=-1)
                        - torch.diff(target, dim=-1)
                    )
                    .abs()
                    .mean()
                )
                detail = (
                    detail
                    + (
                        torch.diff(prediction.float(), dim=-2)
                        - torch.diff(target, dim=-2)
                    )
                    .abs()
                    .mean()
                )
                smooth = (
                    torch.diff(flow, dim=-1).abs().mean()
                    + torch.diff(flow, dim=-2).abs().mean()
                )
                loss = (
                    loss
                    + photo
                    + 0.3 * coarse
                    + 0.12 * detail
                    + 0.004 * smooth
                    + 0.015 * residual.abs().mean()
                )
                if t == 0:
                    loss = (
                        loss
                        + 0.05
                        * (
                            F.smooth_l1_loss(
                                flow.float(),
                                flows[ids].float().clamp(-24, 24),
                                reduction="none",
                            )
                            * confidence[ids]
                        ).mean()
                    )
                context = torch.cat((context[:, 1:], prediction[:, None]), 1)
            loss = loss / horizon
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or time.monotonic() - started > 880:
            model.eval()
            scores = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for ids in val_ids.split(16):
                    observations = obs[ids].float() / 127.5 - 1
                    context = observations[:, :2]
                    mses = []
                    for t in range(8):
                        pred, _, _ = model(context, actions[ids, t : t + 2])
                        mses.append(
                            ((pred.float() - observations[:, t + 2]) / 2)
                            .square()
                            .mean((1, 2, 3))
                        )
                        context = torch.cat((context[:, 1:], pred[:, None]), 1)
                    scores.append(torch.stack(mses, 1))
            values = torch.cat(scores).mean(0).cpu().tolist()
            row = dict(
                step=step,
                seconds=round(time.monotonic() - started, 1),
                loss=float(loss),
                mse_by_horizon=values,
            )
            history.append(row)
            score = float(np.mean(values[3:]))
            if score < best:
                best = score
                torch.save(
                    dict(
                        model=model.state_dict(),
                        architecture="MotionV2",
                        step=step,
                        initialization="random",
                        pretrained_weights=False,
                    ),
                    output / "model.pt",
                )
            (output / "metrics.json").write_text(json.dumps(history, indent=2))
            print(json.dumps(row), flush=True)
        if time.monotonic() - started > 880:
            break
    report = dict(
        step=step,
        seconds=time.monotonic() - started,
        best_rollout_mse=best,
        parameters=sum(p.numel() for p in model.parameters()),
        train_windows=len(train_ids),
        validation_windows=len(val_ids),
        gpu=torch.cuda.get_device_name(),
        selection="mean internal holdout MSE at horizons 4 through 8; grouped by source episode",
        model_sha256=hashlib.sha256((output / "model.pt").read_bytes()).hexdigest(),
    )
    (output / "complete.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def run():
    print(json.dumps(prepare.remote()), flush=True)
    print(json.dumps(fit.remote()), flush=True)
