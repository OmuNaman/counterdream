"""Bounded direct TLS stream. Authentication material stays in process memory."""

import asyncio
import json
import os
from pathlib import Path
import modal

from counterdream.cloud_config import gpu_base_image, volume, PILOT, RUN, DATA
from counterdream.live_checkpoint import snapshot_folder

app = modal.App("counterdream-stream")
GPU_TYPE = os.getenv("COUNTERDREAM_GPU", "H100")
if GPU_TYPE not in ("A100", "H100", "H200"):
    raise ValueError("COUNTERDREAM_GPU must be A100, H100 or H200")
image = gpu_base_image.pip_install("websockets==15.0.1").add_local_python_source(
    "counterdream"
)


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=300,
    retries=0,
    volumes={"/artifacts": volume},
)
def prepare_latest(snapshot: str):
    from counterdream.live_checkpoint import snapshot_latest

    volume.reload()
    report = snapshot_latest(RUN, DATA, snapshot)
    volume.commit()
    return report


@app.function(
    image=image,
    cpu=2,
    memory=2048,
    timeout=180,
    retries=0,
    volumes={"/artifacts": volume},
)
def prepare_demo(variant: str = "demo"):
    """Reuse existing validated20k weights; uploaded bundle fixes every checksum."""
    import shutil
    from counterdream.live_checkpoint import sha256

    volume.reload()
    if variant not in ("demo", "responsive"):
        raise ValueError("Expected a demo bundle variant")
    folder = artifact_folder(variant)
    manifest = json.loads((folder / "bundle.json").read_text())
    source = Path(RUN, "evaluation-val-8-step-20000")
    for name in ("model.pt", "seeds.npz"):
        if sha256(source / name) != manifest["files"][name]:
            raise ValueError("Source artifact differs from demo manifest")
        if not (folder / name).exists():
            shutil.copyfile(source / name, folder / name)
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha256(folder / name) != digest:
            raise ValueError("Demo artifact integrity check failed")
    volume.commit()
    return manifest


@app.local_entrypoint()
def demo_ready(variant: str = "demo"):
    print(json.dumps(prepare_demo.remote(variant)), flush=True)


def artifact_folder(variant, snapshot=""):
    if variant == "latest":
        return snapshot_folder(RUN, snapshot)
    if variant == "full":
        return Path(RUN, "evaluation-test-4")
    if variant == "pilot":
        return Path(PILOT, "evaluation-val-4")
    if variant == "demo":
        return Path(RUN, "demo-upgrade-v1")
    if variant == "responsive":
        return Path(RUN, "demo-responsive-v1")
    raise ValueError("Choose latest, full, pilot, demo or responsive")


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=16384,
    timeout=1900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/artifacts": volume},
)
def stream_server(
    queue,
    token: str,
    variant: str = "full",
    snapshot: str = "",
    deadline_unix: float = 0.0,
    launch_id: str = "",
):
    import os
    import socket
    import time
    import torch
    import uvicorn
    from counterdream.remote_runtime import SessionEngine
    from counterdream.streaming import make_gpu_stream

    if len(token) < 32:
        raise ValueError("Invalid stream configuration")
    deadline = min(deadline_unix or time.time() + 1800, time.time() + 1800)
    if time.time() >= deadline:
        raise RuntimeError("Viewer allocation expired before GPU startup")
    folder = artifact_folder(variant, snapshot)
    volume.reload()
    if variant == "latest":
        from counterdream.live_checkpoint import sha256

        report = json.loads((folder / "preview.json").read_text())
        if (
            sha256(folder / "model.pt") != report["export_sha256"]
            or sha256(folder / "seeds.npz") != report["seeds_sha256"]
        ):
            raise ValueError("Live snapshot integrity check failed")
    if variant in ("demo", "responsive"):
        from counterdream.live_checkpoint import sha256

        manifest = json.loads((folder / "bundle.json").read_text())
        for name, digest in manifest["files"].items():
            if Path(name).name != name or sha256(folder / name) != digest:
                raise ValueError("Demo bundle integrity check failed")
    engine = SessionEngine(
        folder / "model.pt",
        folder / "seeds.npz",
        folder / "profile.json" if variant in ("demo", "responsive") else None,
    )
    if variant == "latest" and engine.checkpoint["step"] != report["checkpoint_step"]:
        raise ValueError("Live snapshot step mismatch")
    # Warm kernels before the first playable frame, then discard the warmup state.
    engine.frame("0" * 32, {"type": "reset", "spawn": 0}, 0)
    engine.frame(
        "0" * 32,
        {"type": "step", "steps": 8 if variant in ("latest", "demo") else 4},
        1,
    )
    engine.sessions.clear()
    activity = {"last": time.monotonic()}
    application = make_gpu_stream(engine, token, activity)
    server = uvicorn.Server(
        uvicorn.Config(
            application,
            host="0.0.0.0",
            port=8000,
            access_log=False,
            log_level="warning",
            ws="websockets",
        )
    )
    # An owned ephemeral socket is released even if Modal cancels the input.
    with socket.socket() as listener:
        listener.bind(("0.0.0.0", 0))
        with modal.forward(listener.getsockname()[1]) as tunnel:

            async def run():
                task = asyncio.create_task(server.serve(sockets=[listener]))
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("Stream server did not start")
                    await asyncio.sleep(0.1)
                await queue.put.aio(
                    dict(
                        url=tunnel.url,
                        region=os.getenv("MODAL_REGION", "unknown"),
                        gpu=torch.cuda.get_device_name(),
                        checkpoint_step=engine.checkpoint["step"],
                        launch_id=launch_id,
                    )
                )
                print(
                    json.dumps(
                        dict(
                            event="stream_ready",
                            checkpoint_step=engine.checkpoint["step"],
                            region=os.getenv("MODAL_REGION", "unknown"),
                            gpu=torch.cuda.get_device_name(),
                            deadline_unix=deadline,
                        )
                    ),
                    flush=True,
                )
                while (
                    time.time() < deadline and time.monotonic() - activity["last"] < 90
                ):
                    if task.done():
                        break
                    await asyncio.sleep(1)
                server.should_exit = True
                await task

            asyncio.run(run())


@app.local_entrypoint()
def probe(variant: str = "pilot", region: str = "any"):
    import secrets
    from counterdream.benchmark_stream import measure

    token = secrets.token_urlsafe(32)
    with modal.Queue.ephemeral() as queue:
        if region not in ("any", "ap"):
            raise ValueError("Choose any or ap")
        provider = stream_server_asia if region == "ap" else stream_server
        call = provider.spawn(queue, token, variant)
        try:
            destination = queue.get(timeout=120)
            result = dict(
                asyncio.run(
                    measure(
                        destination["url"].replace("https://", "wss://", 1) + "/ws",
                        {"Authorization": "Bearer " + token},
                    )
                ),
                region=destination["region"],
                transport="authenticated direct TLS websocket",
            )
            out = Path("artifacts/dust2-v3")
            out.mkdir(parents=True, exist_ok=True)
            (out / f"network-stream-{region}-{variant}.json").write_text(
                json.dumps(result, indent=2)
            )
            print(json.dumps(result), flush=True)
        finally:
            call.cancel()


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=16384,
    region="ap",
    timeout=1900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/artifacts": volume},
)
def stream_server_asia(
    queue,
    token: str,
    variant: str = "full",
    snapshot: str = "",
    deadline_unix: float = 0.0,
    launch_id: str = "",
):
    return stream_server.local(
        queue, token, variant, snapshot, deadline_unix, launch_id
    )


@app.local_entrypoint()
def play(
    variant: str = "full",
    snapshot_id: str = "",
    deadline_unix: float = 0.0,
    region: str = "any",
):
    import secrets
    import time
    import uuid
    import uvicorn
    from counterdream.streaming import make_stream_proxy
    from counterdream.stream_lease import StreamLease, call_is_running

    if region not in ("any", "ap"):
        raise ValueError("Choose any or ap for the inference region")
    # Read-only metadata without allocating an inference GPU yet.
    if snapshot_id and variant != "latest":
        raise ValueError("A pinned snapshot requires the latest variant")
    snapshot = (snapshot_id or uuid.uuid4().hex) if variant == "latest" else ""
    folder = artifact_folder(variant, snapshot)
    if variant == "latest":
        if snapshot_id:
            volume_folder = folder.relative_to("/artifacts").as_posix()
            report = json.loads(
                b"".join(volume.read_file(volume_folder + "/preview.json"))
            )
        else:
            report = prepare_latest.remote(snapshot)
        names = report["spawns"]
        print(json.dumps(dict(event="live_snapshot", **report)), flush=True)
    else:
        volume_folder = folder.relative_to("/artifacts").as_posix()
        report = json.loads(
            b"".join(
                volume.read_file(
                    volume_folder
                    + (
                        "/bundle.json"
                        if variant in ("demo", "responsive")
                        else "/evaluation.json"
                    )
                )
            )
        )
        import io
        import numpy as np

        with np.load(
            io.BytesIO(b"".join(volume.read_file(volume_folder + "/seeds.npz"))),
            allow_pickle=False,
        ) as seeds:
            names = seeds["names"].tolist()
    metadata = dict(
        model="CounterDream v3 / Dust II",
        device=f"Cloud {GPU_TYPE}"
        + (
            " · training preview"
            if variant == "latest"
            else " · pilot" if variant == "pilot" else ""
        ),
        spawns=names,
        checkpoint_step=report["checkpoint_step"],
        context_frames=report["config"]["context"],
        resolution=[report["config"]["width"], report["config"]["height"]],
        pretrained_weights=False,
        recommended_steps=8 if variant in ("latest", "demo") else 4,
        session_note=f"Separate {GPU_TYPE} · pauses when unfocused · GPU stops after 90 seconds idle · 30-minute allocation window",
    )
    if variant in ("demo", "responsive"):
        metadata.update(
            recommended_steps=report["sampling_steps"],
            display_resolution=report["display_resolution"],
            display_note=report["display_note"],
            quality_note=report["quality_note"],
        )
    metadata["recommended_fps"] = 24 if variant == "responsive" else 16
    token = secrets.token_urlsafe(32)
    with modal.Queue.ephemeral() as queue:

        async def start(remaining):
            launch_id = uuid.uuid4().hex
            ready_by = time.monotonic() + min(600, remaining)
            provider = stream_server_asia if region == "ap" else stream_server
            call = await provider.spawn.aio(
                queue, token, variant, snapshot, time.time() + remaining, launch_id
            )
            try:
                while True:
                    timeout = ready_by - time.monotonic()
                    if timeout <= 0:
                        raise TimeoutError("Cloud GPU startup timed out")
                    destination = await queue.get.aio(timeout=timeout)
                    if destination.get("launch_id") == launch_id:
                        break
                if destination["checkpoint_step"] != report["checkpoint_step"]:
                    raise ValueError(
                        "Cloud stream checkpoint does not match the viewer"
                    )
                return call, destination
            except BaseException:
                await call.cancel.aio()
                raise

        async def cancel(call):
            await call.cancel.aio()

        lease = StreamLease(start, call_is_running, cancel)
        if deadline_unix:
            # A software restart can retain the original paid-window deadline.
            lease.deadline = time.monotonic() + max(
                0, min(1800, deadline_unix - time.time())
            )

        async def get_remote():
            destination = await lease.get()
            metadata.update(
                device=destination.get("gpu", f"Cloud {GPU_TYPE}"),
                region=destination["region"],
            )
            return destination["url"], token

        try:
            uvicorn.run(
                make_stream_proxy(metadata, get_remote), host="127.0.0.1", port=7860
            )
        finally:
            asyncio.run(lease.close())
