"""Bounded comparison of sampler settings on a saved preview checkpoint."""

from pathlib import Path
import modal

volume = modal.Volume.from_name("counterdream-artifacts-v1")
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "numpy==1.26.4",
        "Pillow==11.1.0",
        "h5py==3.12.1",
        "requests==2.32.3",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "fastapi==0.115.8",
        "uvicorn==0.34.0",
    )
    .add_local_python_source("counterdream")
)

app = modal.App("counterdream-sampling-check")


@app.function(
    image=gpu_image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=600,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def sweep():
    import json
    import numpy as np
    import torch
    from PIL import Image
    from counterdream.data import Replay
    from counterdream.model import load_model
    from counterdream.train import DeviceReplay, validate
    from counterdream.evaluate import panel, uint8

    torch.set_num_threads(4)
    model, checkpoint = load_model("/artifacts/runs/dust2-v2/preview/model.pt", "cuda")
    replay = Replay("/artifacts/data", "val")
    device_replay = DeviceReplay(replay, "cuda")
    original = model.forward
    reports = []
    pictures = [[] for _ in range(4)]
    labels = []
    for steps, context_noise in [(8, 0), (16, 0), (8, 0.03), (8, 0.1)]:

        def forward(noisy, sigma, context, actions, context_sigma=None):
            return original(
                noisy,
                sigma,
                context,
                actions,
                torch.full((len(noisy),), context_noise, device="cuda"),
            )

        model.forward = forward
        metrics = validate(
            model, device_replay, "cuda", batches=4, batch_size=16, steps=steps
        )
        errors = {16: [], 32: [], 64: []}
        label = f"{steps} STEPS / CONTEXT NOISE {context_noise}"
        labels.append(label)
        for clip in range(4):
            start = 120 + clip * 71
            obs = replay.frames[clip][start : start + 68].to("cuda").float() / 127.5 - 1
            actions = replay.actions[clip][start : start + 67].to("cuda")
            context = obs[:4][None]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for t in range(64):
                    pred = model.sample(
                        context,
                        actions[t : t + 4][None],
                        steps=steps,
                        seed=10000 + clip * 1000 + t,
                    )
                    if t + 1 in errors:
                        errors[t + 1].append(
                            float(((pred[0] - obs[t + 4]) / 2).square().mean())
                        )
                    context = torch.cat((context[:, 1:], pred[:, None]), dim=1)
            if not pictures[clip]:
                pictures[clip].append(uint8(obs[-1]))
            pictures[clip].append(uint8(pred[0]))
        metrics.update(
            context_noise=context_noise,
            checkpoint_step=checkpoint["step"],
            rollout_mse={str(k): float(np.mean(v)) for k, v in errors.items()},
        )
        reports.append(metrics)
        print(json.dumps(metrics), flush=True)
    root = Path("/artifacts/runs/dust2-v2/stability")
    root.mkdir(exist_ok=True)
    for clip, pictures_for_clip in enumerate(pictures):
        Image.fromarray(
            panel(pictures_for_clip, ["HELD-OUT FRAME 64", *labels], scale=3)
        ).save(root / f"clip-{clip + 1}.png")
    (root / "comparison.json").write_text(json.dumps(reports, indent=2))
    volume.commit()
    return reports


@app.local_entrypoint()
def check():
    sweep.remote()
