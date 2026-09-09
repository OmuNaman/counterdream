"""Bounded direct TLS stream. Authentication material stays in process memory."""
import asyncio
import json
from pathlib import Path
import modal

from counterdream.cloud_config import gpu_base_image,volume,PILOT,RUN

app=modal.App("counterdream-stream")
image=gpu_base_image.pip_install("websockets==15.0.1").add_local_python_source("counterdream")


@app.function(image=image,gpu="H100",cpu=4,memory=16384,
              region=["ap-south","ap-southeast"],timeout=1900,retries=0,
              max_containers=1,scaledown_window=2,volumes={"/artifacts":volume})
def stream_server(queue,token:str,variant:str="full"):
    import os
    import time
    import uvicorn
    from counterdream.remote_runtime import SessionEngine
    from counterdream.streaming import make_gpu_stream
    if variant not in ("full","pilot") or len(token)<32:
        raise ValueError("Invalid stream configuration")
    folder=Path(RUN,"evaluation-test-4") if variant=="full" else Path(PILOT,"evaluation-val-4")
    engine=SessionEngine(folder/"model.pt",folder/"seeds.npz")
    # Warm kernels before the first playable frame, then discard the warmup state.
    engine.frame("0"*32,{"type":"reset","spawn":0},0)
    engine.frame("0"*32,{"type":"step","steps":4},1)
    engine.sessions.clear()
    activity={"last":time.monotonic()}
    application=make_gpu_stream(engine,token,activity)
    server=uvicorn.Server(uvicorn.Config(application,host="0.0.0.0",port=8000,
                                        access_log=False,log_level="warning",ws="websockets"))
    with modal.forward(8000) as tunnel:
        async def run():
            task=asyncio.create_task(server.serve())
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("Stream server did not start")
                await asyncio.sleep(.1)
            await queue.put.aio(dict(url=tunnel.url,region=os.getenv("MODAL_REGION","unknown")))
            started=time.monotonic()
            while time.monotonic()-started<1800 and time.monotonic()-activity["last"]<90:
                if task.done():
                    break
                await asyncio.sleep(1)
            server.should_exit=True
            await task
        asyncio.run(run())


@app.local_entrypoint()
def probe():
    import secrets
    from counterdream.benchmark_stream import measure
    token=secrets.token_urlsafe(32)
    with modal.Queue.ephemeral() as queue:
        call=stream_server.spawn(queue,token,"pilot")
        try:
            destination=queue.get(timeout=600)
            result=dict(asyncio.run(measure(destination["url"].replace("https://","wss://",1)+"/ws",
                                           {"Authorization":"Bearer "+token})),
                        region=destination["region"],transport="authenticated direct TLS websocket")
            out=Path("artifacts/dust2-v3")
            out.mkdir(parents=True,exist_ok=True)
            (out/"network-stream.json").write_text(json.dumps(result,indent=2))
            print(json.dumps(result),flush=True)
        finally:
            call.cancel()


@app.local_entrypoint()
def play(variant: str = "full"):
    import secrets
    import uvicorn
    from counterdream.streaming import make_stream_proxy
    # Read-only metadata without allocating an inference GPU yet.
    if variant not in ("full","pilot"):
        raise ValueError("Choose full or pilot")
    folder=Path(RUN,"evaluation-test-4") if variant=="full" else Path(PILOT,"evaluation-val-4")
    volume_folder=folder.relative_to("/artifacts").as_posix()
    report=json.loads(b"".join(volume.read_file(volume_folder+"/evaluation.json")))
    import io
    import numpy as np
    with np.load(io.BytesIO(b"".join(volume.read_file(volume_folder+"/seeds.npz"))),allow_pickle=False) as seeds:
        names=seeds["names"].tolist()
    metadata=dict(model="CounterDream v3 / Dust II",device="Cloud H100"+(" · pilot" if variant=="pilot" else ""),spawns=names,
                  checkpoint_step=report["checkpoint_step"],context_frames=report["config"]["context"],
                  resolution=[report["config"]["width"],report["config"]["height"]],pretrained_weights=False)
    token=secrets.token_urlsafe(32)
    with modal.Queue.ephemeral() as queue:
        state={}
        lock=asyncio.Lock()
        async def get_remote():
            async with lock:
                if "destination" not in state:
                    state["call"]=await stream_server.spawn.aio(queue,token,variant)
                    state["destination"]=await queue.get.aio(timeout=600)
                return state["destination"]["url"],token
        try:
            uvicorn.run(make_stream_proxy(metadata,get_remote),host="127.0.0.1",port=7860)
        finally:
            if "call" in state:
                state["call"].cancel()
