"""Five-H100 experiment; explicit invocations, bounded runtime, no auto retries."""
import json
from pathlib import Path
import modal

VOLUME_NAME = "counterdream-artifacts-v1"

app = modal.App("counterdream-scale-v3")
volume = modal.Volume.from_name("counterdream-artifacts-v1")
DATA = "/artifacts/data-v3"
PILOT = "/artifacts/runs/dust2-v3"
RUN = "/artifacts/runs/dust2-v3-full"
data_image = (modal.Image.debian_slim(python_version="3.11")
              .pip_install("numpy==1.26.4", "Pillow==11.1.0", "h5py==3.12.1", "requests==2.32.3")
              .add_local_python_source("counterdream")
              .add_local_file("counterdream/assets/diamond-test-split.txt", "/root/diamond-test-split.txt"))
gpu_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch==2.6.0", "numpy==1.26.4", "Pillow==11.1.0",
                          "h5py==3.12.1", "requests==2.32.3", "imageio==2.37.0",
                          "imageio-ffmpeg==0.6.0", "fastapi==0.115.8", "uvicorn==0.34.0")
             .add_local_python_source("counterdream"))


def _launch(seconds, batch, source, resume, output, gpu_replay=False, target_steps=60000):
    import os
    import subprocess
    import sys
    import torch
    if torch.cuda.device_count() != 5:
        raise RuntimeError("This experiment requires exactly five allocated GPUs")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "invocation-source.json"
    source_path.write_text(json.dumps(source, indent=2))
    latest = root / "latest.pt"
    if latest.exists() != resume:
        raise ValueError("Resume must match whether a checkpoint already exists")
    args = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            "--nproc-per-node=5", "-m", "counterdream.train_distributed", "--data", DATA,
            "--output", output, "--steps", str(target_steps), "--max-seconds", str(seconds),
            "--batch", str(batch), "--source", str(source_path),
            "--commit-volume", VOLUME_NAME]
    if resume:
        args += ["--resume", str(latest)]
    if gpu_replay:
        args += ["--gpu-replay"]
    env = dict(os.environ, OMP_NUM_THREADS="3", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    subprocess.run(args, check=True, env=env, timeout=seconds+90)
    volume.reload()
    return json.loads((root / "complete.json").read_text())


@app.function(image=gpu_image, gpu="H100:5", cpu=20, memory=65536,
              timeout=900, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": volume})
def benchmark_five(source: dict, batch: int = 12):
    if not 1 <= batch <= 24:
        raise ValueError("Batch outside tested bounds")
    return _launch(600, batch, source, resume=False, output=PILOT)


@app.function(image=gpu_image,gpu="H100:5",cpu=20,memory=65536,timeout=600,
              retries=0,max_containers=1,scaledown_window=2,volumes={"/artifacts":volume})
def verify_gpu_shards(source: dict):
    return _launch(300,12,source,False,"/artifacts/runs/dust2-v3-shard-check",True,target_steps=2)


@app.local_entrypoint()
def verify():
    print(json.dumps(verify_gpu_shards.remote(source_manifest())),flush=True)


@app.function(image=gpu_image, gpu="H100:5", cpu=20, memory=65536,
              timeout=21600, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": volume})
def train_five(source: dict, batch: int = 12, seconds: int = 19800):
    if not 600 <= seconds <= 19800 or not 1 <= batch <= 24:
        raise ValueError("Full run is capped at 5.5 training hours")
    index = json.loads(Path(DATA, "index.json").read_text())
    if index["counts"]["train"] < 1_000_000:
        raise ValueError("Prepare at least one million training frames first")
    # A committed launch marker prevents an accidental second full paid run.
    marker = Path(RUN, "full-allocation.json")
    marker.parent.mkdir(parents=True,exist_ok=True)
    if marker.exists():
        raise ValueError("Full allocation already used; inspect progress and budget first")
    marker.write_text(json.dumps(dict(seconds=seconds, maximum_function_seconds=21600,
                                     gpu_count=5, source=source), indent=2))
    volume.commit()
    return _launch(seconds, batch, source, resume=False, output=RUN, gpu_replay=True)


def source_manifest():
    import hashlib
    import subprocess
    paths = sorted(Path("counterdream").glob("*.py")) + [Path("cloud_scale.py")]
    return dict(git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
                sha256={p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


@app.local_entrypoint()
def benchmark(batch: int = 12):
    print(json.dumps(benchmark_five.remote(source_manifest(), batch)), flush=True)


@app.local_entrypoint()
def train(batch: int = 12, seconds: int = 19800):
    print(json.dumps(train_five.remote(source_manifest(), batch, seconds)), flush=True)


@app.function(image=gpu_image, gpu="H100", cpu=4, memory=16384,
              timeout=1800, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": volume})
def evaluate_run(split: str = "val", steps: int = 4, pilot: bool = False):
    from counterdream.evaluate_scaled import evaluate
    if split not in ("val", "test") or steps not in (4,8):
        raise ValueError("Choose val/test and 4/8 sampling steps")
    if pilot and split != "val":
        raise ValueError("Pilot evaluation is validation-only")
    folder = PILOT if pilot else RUN
    result = evaluate(Path(folder,"best.pt"),DATA,Path(folder,f"evaluation-{split}-{steps}"),split=split,steps=steps)
    volume.commit()
    return result


@app.local_entrypoint()
def assess(split: str = "val", steps: int = 4, pilot: bool = False):
    print(json.dumps(evaluate_run.remote(split,steps,pilot)),flush=True)


@app.cls(image=gpu_image, gpu="H100", cpu=4, memory=16384,
         timeout=120, retries=0, min_containers=0, max_containers=1,
         scaledown_window=15, region=["ap-south","ap-southeast"],routing_region="ap-south",
         volumes={"/artifacts": volume})
class CloudDreamer:
    variant: str = modal.parameter(default="full")

    @modal.enter()
    def load(self):
        from counterdream.remote_runtime import SessionEngine
        if self.variant not in ("full","pilot"):
            raise ValueError("Unknown inference variant")
        folder = Path(RUN,"evaluation-test-4") if self.variant=="full" else Path(PILOT,"evaluation-val-4")
        self.engine = SessionEngine(folder/"model.pt",folder/"seeds.npz")

    @modal.method()
    def frame(self, session_id: str, command: dict, sequence: int):
        return self.engine.frame(session_id,command,sequence)


@app.function(image=data_image, cpu=1, memory=2048, timeout=120,
              retries=0, volumes={"/artifacts": volume})
def viewer_metadata():
    import numpy as np
    folder = Path(RUN,"evaluation-test-4")
    report = json.loads((folder/"evaluation.json").read_text())
    with np.load(folder/"seeds.npz",allow_pickle=False) as seeds:
        names = seeds["names"].tolist()
    return dict(model="CounterDream v3 / Dust II", device="Cloud H100",
                checkpoint_step=report["checkpoint_step"], pretrained_weights=False,
                context_frames=report["config"]["context"],spawns=names,
                resolution=[report["config"]["width"],report["config"]["height"]])


@app.local_entrypoint()
def play():
    import uvicorn
    from counterdream.cloud_viewer import make_cloud_app
    metadata = viewer_metadata.remote()
    remote = CloudDreamer()
    async def frame(session,command,sequence):
        return await remote.frame.remote.aio(session,command,sequence)
    # Credentials stay in this process. This creates no public endpoint.
    uvicorn.run(make_cloud_app(metadata,frame),host="127.0.0.1",port=7860)


@app.local_entrypoint()
def network_check(variant: str = "pilot"):
    import time
    import uuid
    import statistics
    if variant not in ("pilot","full"):
        raise ValueError("Unknown network-check variant")
    remote = CloudDreamer(variant=variant)
    session = uuid.uuid4().hex
    tick = time.perf_counter()
    first = remote.frame.remote(session,{"type":"reset","spawn":0},0)
    cold = time.perf_counter()-tick
    if not first["png"].startswith(b"\x89PNG"):
        raise ValueError("Invalid cloud frame")
    totals,gpu = [],[]
    for seq in range(1,33):
        tick = time.perf_counter()
        response = remote.frame.remote(session,{"type":"step","dx":10,"steps":4},seq)
        if response.get("error"):
            raise RuntimeError(response["error"])
        totals.append((time.perf_counter()-tick)*1000)
        gpu.append(response["gpu_ms"])
    report = dict(variant=variant,frames=len(totals),cold_start_seconds=cold,
                  total_ms_median=statistics.median(totals[3:]),gpu_ms_median=statistics.median(gpu[3:]),
                  total_ms_samples=totals,gpu_ms_samples=gpu)
    root = Path("artifacts/dust2-v3")
    root.mkdir(parents=True,exist_ok=True)
    (root/f"network-{variant}.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


@app.local_entrypoint()
def fetch(destination: str = "artifacts/dust2-v3", weights: bool = False):
    root = Path(destination)
    root.mkdir(parents=True,exist_ok=True)
    volume_folder = Path(RUN).relative_to("/artifacts").as_posix()
    for entry in volume.iterdir(volume_folder,recursive=True):
        if not entry.path.endswith((".json",".jsonl",".png",".mp4",".npz")) and not (
            weights and entry.path.endswith("/model.pt")):
            continue
        relative = Path(entry.path).relative_to(volume_folder)
        target = root / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("wb") as stream:
            for block in volume.read_file(entry.path):
                stream.write(block)
        print(str(target),flush=True)


@app.function(image=data_image, cpu=2, memory=4096, timeout=3600, retries=0,
              max_containers=8, scaledown_window=2, volumes={"/artifacts": volume})
def prepare_part(shard: str, limit: int = 0):
    from counterdream.scaled_data import prepare_shard
    test_names = set(Path("/root/diamond-test-split.txt").read_text().splitlines())
    result = prepare_shard(DATA, shard, test_names, limit=limit or None, commit=volume.commit)
    return dict(shard=shard, episodes=len(result["episodes"]), complete=result["complete"])


@app.function(image=data_image, cpu=1, memory=2048, timeout=300, retries=0,
              volumes={"/artifacts": volume})
def index_data(partial: bool = False):
    from counterdream.scaled_data import build_index
    result = build_index(DATA, allow_partial=partial)
    volume.commit()
    return {k: v for k, v in result.items() if k != "episodes"}


@app.function(image=data_image,cpu=2,memory=4096,timeout=3600,retries=0,
              max_containers=1,scaledown_window=2,volumes={"/artifacts":volume})
def prepare_expert_data():
    from counterdream.scaled_data import prepare_expert
    result = prepare_expert(DATA,commit=volume.commit)
    return dict(episodes=len(result["episodes"]),complete=result["complete"])


@app.local_entrypoint()
def expert():
    print(json.dumps(prepare_expert_data.remote()),flush=True)


@app.local_entrypoint()
def index(partial: bool = False):
    print(json.dumps(index_data.remote(partial)),flush=True)


@app.local_entrypoint()
def prepare(shards: int = 28, limit: int = 0):
    import requests
    import re
    if not 1 <= shards <= 28 or not 0 <= limit <= 200:
        raise ValueError("Limit corpus to 28 shards, 200 episodes each")
    response = requests.get('https://huggingface.co/api/datasets/TeaPearce/CounterStrike_Deathmatch/tree/265c6e5ac7aa335f58a2f2e864aad176fecfedde',
                            params={'recursive': 'false', 'limit': 1000}, timeout=60)
    response.raise_for_status()
    selected = sorted([x for x in response.json() if re.fullmatch(
        r'hdf5_dm_july2021_\d+_to_\d+\.tar', x['path'])],
        key=lambda x: int(x['path'].split('_')[3]))[:shards]
    print(json.dumps(dict(archives=len(selected), source_bytes=sum(x["size"] for x in selected))), flush=True)
    results = list(prepare_part.starmap([(x["path"], limit) for x in selected]))
    print(json.dumps(results), flush=True)
    print(json.dumps(index_data.remote(partial=any(not x["complete"] for x in results))), flush=True)
