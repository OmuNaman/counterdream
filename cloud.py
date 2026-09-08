"""Bounded Modal jobs. Authentication comes from Modal's environment/profile only."""

from pathlib import Path
import json
import modal

APP_NAME = "counterdream-research"
app = modal.App(APP_NAME)
volume = modal.Volume.from_name("counterdream-artifacts-v1", create_if_missing=True)
data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.26.4", "Pillow==11.1.0", "h5py==3.12.1", "requests==2.32.3")
    .add_local_python_source("counterdream")
)

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


@app.function(
    image=gpu_image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=15000,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/artifacts": volume},
)
def train_model(
    run: str = "pilot", steps: int = 100, max_seconds: int = 600, resume: bool = False
):
    from counterdream.train import train

    if run not in ("pilot", "dust2-v1", "dust2-v2"):
        raise ValueError("Unknown run name")
    if steps > 80000 or max_seconds > 14400:
        raise ValueError("This allocation is limited to 80000 steps / four H100 hours")
    if run != "pilot" and not Path("/artifacts/data/manifest.json").exists():
        raise ValueError("Complete and verify the dataset before a full run")
    output = f"/artifacts/runs/{run}"
    latest = Path(output) / "latest.pt"
    if latest.exists() and not resume:
        raise ValueError("Run already exists; choose resume or another run")
    return train(
        "/artifacts/data",
        output,
        steps=steps,
        max_seconds=max_seconds,
        resume=str(latest) if resume else None,
        commit=volume.commit,
    )


@app.function(
    image=gpu_image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=1200,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/artifacts": volume},
)
def evaluate_model(run: str = "dust2-v1"):
    from counterdream.evaluate import evaluate

    if run not in ("pilot", "dust2-v1", "dust2-v2"):
        raise ValueError("Unknown run")
    root = Path("/artifacts/runs") / run
    result = evaluate(root / "model.pt", "/artifacts/data", root / "evaluation")
    volume.commit()
    return result


@app.cls(
    image=gpu_image,
    gpu="L4",
    cpu=2,
    memory=8192,
    timeout=300,
    retries=0,
    max_containers=1,
    min_containers=0,
    scaledown_window=15,
    volumes={"/artifacts": volume},
)
class Dreamer:
    @modal.enter()
    def load(self):
        import torch
        from counterdream.model import load_model

        torch.set_num_threads(2)
        self.model, self.checkpoint = load_model(
            "/artifacts/runs/dust2-v1/model.pt", "cuda"
        )

    @modal.method()
    def step(self, context, actions, steps: int, seed: int):
        import numpy as np
        import torch

        if context.shape != (4, 64, 112, 3) or context.dtype != np.uint8:
            raise ValueError("Expected four uint8 RGB context frames")
        if actions.shape != (4, 51) or not np.isfinite(actions).all():
            raise ValueError("Expected four finite action vectors")
        if not 2 <= steps <= 16:
            raise ValueError("Use 2–16 steps for interactive inference")
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            c = (
                torch.from_numpy(context.copy())
                .to("cuda")
                .permute(0, 3, 1, 2)[None]
                .float()
                / 127.5
                - 1
            )
            a = torch.from_numpy(actions.copy()).to("cuda")[None].float()
            output = self.model.sample(c, a, steps=steps, seed=seed)[0]
            return (
                output.float()
                .add(1)
                .mul(127.5)
                .round()
                .clamp(0, 255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )


@app.function(
    image=data_image,
    cpu=8,
    memory=16384,
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def prepare_data(episodes: int = 100):
    from counterdream.data import prepare

    if not 10 <= episodes <= 200:
        raise ValueError("The bounded data job supports 10–200 episodes")
    manifest = Path("/artifacts/data/manifest.json")
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if len(data["episodes"]) == episodes:
            return {"cached": True, "episodes": episodes}
        raise ValueError(
            "Dataset already exists; use a new volume for a different split"
        )
    result = prepare("/artifacts/data", episodes=episodes, progress=volume.commit)
    volume.commit()
    return {"episodes": len(result["episodes"]), "seconds": result["seconds"]}


@app.local_entrypoint()
def prepare(episodes: int = 100):
    print(prepare_data.remote(episodes))


@app.local_entrypoint()
def fit(
    run: str = "pilot", steps: int = 100, max_seconds: int = 600, resume: bool = False
):
    print(train_model.remote(run, steps, max_seconds, resume))


@app.local_entrypoint()
def assess(run: str = "dust2-v1"):
    print(evaluate_model.remote(run))


@app.local_entrypoint()
def play(seeds: str = "artifacts/dust2-v1/evaluation/seeds.npz"):
    import uvicorn
    from counterdream.serve import make_app

    dreamer = Dreamer()

    async def predict(context, actions, steps, seed):
        return await dreamer.step.remote.aio(context, actions, steps, seed)

    print("Open http://127.0.0.1:7860 — Modal credentials stay in this process.")
    uvicorn.run(
        make_app(seeds, predict, {"device": "Modal L4", "pretrained_weights": False}),
        host="127.0.0.1",
        port=7860,
    )
