"""Compare the experimental motion model against the deployed one on equal inputs."""

import json
from pathlib import Path
import modal
from counterdream.cloud_config import gpu_base_image, volume, RUN

app = modal.App("counterdream-motion-v2-eval")
image = gpu_base_image.add_local_python_source("counterdream")
ROOT = Path(RUN, "motion-v2-experiment")
EVALUATION = "evaluation-2"


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=600,
    retries=0,
    volumes={"/artifacts": volume},
    max_containers=1,
    scaledown_window=2,
)
def evaluate():
    import hashlib
    import io
    import time
    import statistics
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from counterdream.motion_v2 import load_motion
    from counterdream.motion_model import guided_motion
    from counterdream.remote_runtime import SessionEngine
    from counterdream.play_sequence import play_controls

    volume.reload()
    output = ROOT / EVALUATION
    if (output / "complete.json").exists():
        return json.loads((output / "complete.json").read_text())
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    data = ROOT / "data"
    observations = np.load(data / "observations.npy", mmap_mode="r")
    actions = np.load(data / "actions.npy", mmap_mode="r")
    sources = json.loads((data / "samples.json").read_text())
    held = [
        i
        for i, s in enumerate(sources)
        if int(hashlib.sha256(s["source"].encode()).hexdigest()[:8], 16) % 10 == 0
    ]
    ids = np.asarray(held)[
        np.linspace(0, len(held) - 1, min(128, len(held))).astype(int)
    ]
    obs = torch.from_numpy(observations[ids].copy()).cuda().float() / 127.5 - 1
    act = torch.from_numpy(actions[ids].copy()).cuda()
    paths = {"v1": Path(RUN, "motion-v1/model.pt"), "v2": ROOT / "fit/model.pt"}
    metrics = {}
    started = time.monotonic()
    for name, path in paths.items():
        model = load_motion(path)
        for centered in (False, True):
            scores = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for offset in range(0, len(obs), 16):
                    context = obs[offset : offset + 16, :2]
                    values = []
                    for t in range(8):
                        prediction = guided_motion(
                            model,
                            context,
                            act[offset : offset + 16, t : t + 2],
                            center=centered,
                        )
                        values.append(
                            (
                                (prediction.float() - obs[offset : offset + 16, t + 2])
                                / 2
                            )
                            .square()
                            .mean((1, 2, 3))
                        )
                        context = torch.cat((context[:, 1:], prediction[:, None]), 1)
                    scores.append(torch.stack(values, 1))
            metrics[f"{name}-" + ("centered" if centered else "raw")] = (
                torch.cat(scores).mean(0).cpu().tolist()
            )
        del model
    metrics["repeat"] = (
        ((obs[:, 1, None] - obs[:, 2:]) / 2).square().mean((0, 2, 3, 4)).cpu().tolist()
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(dict(holdout_metrics=metrics)), flush=True)
    del obs, act
    source = Path(RUN, "demo-responsive-v1")
    engine = SessionEngine(
        source / "model.pt", source / "seeds.npz", source / "profile.json"
    )
    reports = []
    for name, centered in [("v1", True), ("v2", False), ("v2", True)]:
        engine.motion = load_motion(paths[name])
        engine.settings["motion_center"] = centered
        for spawn in (0, 4):
            label = f"{name}-" + ("centered" if centered else "raw") + f"-spawn{spawn}"
            folder = output / label
            folder.mkdir()
            engine.sessions.clear()
            engine.frame("1" * 32, dict(type="reset", spawn=spawn), 0)
            timings = []
            shots = []
            native = []
            with imageio.get_writer(
                folder / "rollout.mp4",
                fps=16,
                codec="libx264",
                quality=8,
                macro_block_size=2,
            ) as writer:
                for i in range(320):
                    if time.monotonic() - started > 520:
                        raise TimeoutError("Evaluation bound reached")
                    action_name, command = play_controls(i)
                    result = engine.frame(
                        "1" * 32, dict(type="step", steps=4, **command), i + 1
                    )
                    if result.get("error"):
                        raise RuntimeError(result["error"])
                    pixels = Image.open(io.BytesIO(result["png"])).convert("RGB")
                    timings.append(result["gpu_ms"])
                    native.append(
                        engine.sessions["1" * 32]["context"][0, -1]
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
                    canvas = Image.new("RGB", (640, 400), "#10151e")
                    canvas.paste(pixels, (0, 24))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((8, 6), f"{label} / frame {i+1}", fill="white")
                    draw.text(
                        (8, 383), action_name + " / OFFLINE 16 FPS", fill="#d9ee8b"
                    )
                    writer.append_data(np.asarray(canvas))
                    if i + 1 in (1, 32, 48, 80, 88, 112, 160, 192, 224, 256, 288, 320):
                        shots.append(canvas.copy())
            sheet = Image.new("RGB", (960, 800), "#10151e")
            for i, shot in enumerate(shots):
                sheet.paste(shot.resize((320, 200)), ((i % 3) * 320, (i // 3) * 200))
            sheet.save(folder / "contact-sheet.jpg", quality=94)
            np.savez_compressed(folder / "native-frames.npz", frames=np.asarray(native))
            row = dict(
                name=label,
                frames=320,
                gpu_ms_median=statistics.median(timings),
                motion_sha256=hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            )
            (folder / "report.json").write_text(json.dumps(row, indent=2))
            reports.append(row)
            print(json.dumps(row), flush=True)
    files = {
        p.relative_to(output).as_posix(): dict(
            bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()
        )
        for p in output.rglob("*")
        if p.is_file()
    }
    report = dict(
        metrics=metrics,
        rollouts=reports,
        seconds=time.monotonic() - started,
        files=files,
        validation="128 fixed episode-disjoint windows from original training split; no test data read",
        recording="offline playback at 16fps, not a latency benchmark; no resets or recorded future frames",
    )
    (output / "complete.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def run():
    print(json.dumps(evaluate.remote()), flush=True)


@app.local_entrypoint()
def download():
    import hashlib

    folder = Path("artifacts/motion-v2-experiment")
    folder.mkdir(exist_ok=True, parents=True)
    remote = ROOT.relative_to("/artifacts").as_posix()
    for part in (
        "data/report.json",
        "fit/complete.json",
        "fit/metrics.json",
        "fit/model.pt",
        EVALUATION + "/complete.json",
    ):
        target = folder / part
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes(b"".join(volume.read_file(remote + "/" + part)))
    report = json.loads((folder / EVALUATION / "complete.json").read_text())
    for name, spec in report["files"].items():
        if name.endswith("native-frames.npz"):
            continue
        target = folder / EVALUATION / name
        target.parent.mkdir(exist_ok=True, parents=True)
        if (
            target.is_file()
            and hashlib.sha256(target.read_bytes()).hexdigest() == spec["sha256"]
        ):
            continue
        target.write_bytes(
            b"".join(volume.read_file(remote + "/" + EVALUATION + "/" + name))
        )
        assert hashlib.sha256(target.read_bytes()).hexdigest() == spec["sha256"]
        print(json.dumps(dict(downloaded=name, verified=True)), flush=True)
