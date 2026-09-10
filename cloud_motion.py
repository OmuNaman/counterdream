"""Bounded training-data preparation for an action-conditioned motion experiment."""

import json
from pathlib import Path

import modal

from counterdream.cloud_config import DATA, RUN, volume, gpu_base_image

app = modal.App("counterdream-motion-experiment")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "requests==2.32.3",
        "Pillow==11.1.0",
        "h5py==3.12.1",
    )
    .add_local_python_source("counterdream")
)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=1200,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/artifacts": volume},
)
def prepare_pairs():
    import hashlib
    import time
    import cv2
    import numpy as np
    from counterdream.scaled_data import DiskReplay

    destination = Path(RUN, "motion-data-v2")
    marker = destination / "report.json"
    if marker.exists():
        return json.loads(marker.read_text())
    output = Path("/tmp/motion-data-v2")
    output.mkdir(exist_ok=True)
    cv2.setNumThreads(2)
    replay = DiskReplay(DATA, split="train", context=2)
    rng = np.random.default_rng(7102026)
    count = 4096
    observations = np.lib.format.open_memmap(
        output / "observations.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(count, 3, 3, 88, 160),
    )
    actions = np.lib.format.open_memmap(
        output / "actions.npy", mode="w+", dtype=np.float32, shape=(count, 2, 51)
    )
    flows = np.lib.format.open_memmap(
        output / "flows.npy", mode="w+", dtype=np.float16, shape=(count, 2, 88, 160)
    )
    confidences = np.lib.format.open_memmap(
        output / "confidence.npy", mode="w+", dtype=np.uint8, shape=(count, 1, 88, 160)
    )
    flow_estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    yy, xx = np.mgrid[:88, :160].astype(np.float32)
    sampled = []
    started = time.monotonic()
    # Read multiple windows per episode: opening thousands of different remote
    # memory maps otherwise spends most of this bounded job waiting for storage.
    episodes = rng.choice(
        len(replay.records), count // 32, replace=False, p=replay.weights
    )
    for i in range(count):
        if time.monotonic() - started > 1050:
            raise TimeoutError("Motion data preparation exceeded its bound")
        episode = int(episodes[i // 32])
        frames, acts = replay.episode(episode)
        start = int(rng.integers(0, len(frames) - 2))
        obs = frames[start : start + 3].copy()
        act = acts[start : start + 2].copy()
        before, after = obs[1].transpose(1, 2, 0), obs[2].transpose(1, 2, 0)
        gray_before = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        flow = flow_estimator.calc(gray_after, gray_before, None)
        warped = cv2.remap(
            before,
            xx + flow[:, :, 0],
            yy + flow[:, :, 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        difference = np.abs(warped.astype(np.float32) - after).mean(2) / 255
        valid = (
            (xx + flow[:, :, 0] >= 0)
            & (xx + flow[:, :, 0] <= 159)
            & (yy + flow[:, :, 1] >= 0)
            & (yy + flow[:, :, 1] <= 87)
        )
        confidence = (np.exp(-difference / 0.1) * valid * 255).round().astype(np.uint8)
        observations[i], actions[i], flows[i], confidences[i, 0] = (
            obs,
            act,
            flow.transpose(2, 0, 1),
            confidence,
        )
        sampled.append(
            dict(source=replay.records[episode]["source"], start=start, split="train")
        )
        if (i + 1) % 64 == 0:
            print(
                json.dumps(
                    dict(
                        prepared=i + 1,
                        total=count,
                        seconds=round(time.monotonic() - started, 1),
                    )
                ),
                flush=True,
            )
    for a in (observations, actions, flows, confidences):
        a.flush()
    (output / "samples.json").write_text(json.dumps(sampled), encoding="utf-8")

    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    report = dict(
        samples=count,
        split="train",
        seed=7102026,
        flow_method="OpenCV DIS medium, next-to-previous",
        dataset_index_sha256=digest(Path(DATA, "index.json")),
        files={
            p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
            for p in output.iterdir()
            if p.is_file()
        },
        seconds=time.monotonic() - started,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import shutil

    shutil.copytree(output, destination, dirs_exist_ok=True)
    volume.commit()
    return report


@app.local_entrypoint()
def prepare():
    print(json.dumps(prepare_pairs.remote()), flush=True)


@app.function(
    image=gpu_base_image.add_local_python_source("counterdream"),
    gpu="A100",
    cpu=4,
    memory=16384,
    timeout=720,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/artifacts": volume},
)
def fit_motion():
    from counterdream.train_motion import train

    volume.reload()
    output = Path(RUN, "motion-v1")
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    try:
        train(Path(RUN, "motion-data-v2"), output, seconds=600, steps=5000)
    finally:
        volume.commit()
    return json.loads((output / "complete.json").read_text())


@app.local_entrypoint()
def fit():
    print(json.dumps(fit_motion.remote()), flush=True)
