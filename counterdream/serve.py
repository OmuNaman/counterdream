"""Local browser viewer. All frames after reset come from the learned model.

The Modal GPU is private: credentials stay in this local server process, never JS.
"""

import asyncio
import io
from pathlib import Path
import time
from typing import Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from .actions import encode


class Control(BaseModel):
    type: Literal["step", "reset"] = "step"
    keys: list[
        Literal["w", "a", "s", "d", "space", "ctrl", "shift", "1", "2", "3", "r"]
    ] = Field(default_factory=list, max_length=11)
    dx: float = Field(default=0, ge=-1000, le=1000, allow_inf_nan=False)
    dy: float = Field(default=0, ge=-200, le=200, allow_inf_nan=False)
    fire: bool = False
    scope: bool = False
    spawn: int = Field(default=0, ge=0)
    steps: int = Field(default=8, ge=2, le=16)
    fps: Literal[16, 24] = 16


def png(frame):
    stream = io.BytesIO()
    Image.fromarray(frame).save(stream, format="PNG")
    return stream.getvalue()


def make_app(seed_path, predict, metadata=None, max_generated_frames=2000):
    """predict(context uint8 THWC, actions float32, steps, seed) -> uint8 HWC."""
    with np.load(seed_path, allow_pickle=False) as seeds:
        seed_frames = seeds["frames"].copy()
        seed_actions = seeds["actions"].copy()
        names = seeds["names"].tolist()
    if seed_frames.ndim != 5 or seed_frames.shape[1:] not in ((4,64,112,3),(8,88,160,3)):
        raise ValueError("Unexpected seed shape")
    context_frames,height,width,_=seed_frames.shape[1:]
    if seed_actions.shape != (len(seed_frames), context_frames-1, 51):
        raise ValueError("Unexpected seed action shape")
    if metadata and (metadata.get("context_frames",context_frames)!=context_frames or
                     metadata.get("resolution",[width,height])!=[width,height]):
        raise ValueError("Checkpoint and starting frames have different dimensions")
    app = FastAPI(title="CounterDream", docs_url=None, redoc_url=None)
    state = {"generated": 0}
    semaphore = asyncio.Semaphore(1)
    static = Path(__file__).parent / "static"

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/app.js")
    def script():
        return FileResponse(static / "app.js", media_type="text/javascript")

    @app.get("/style.css")
    def styles():
        return FileResponse(static / "style.css", media_type="text/css")

    @app.get("/api/info")
    def info():
        return {
            "model": "CounterDream / Dust II",
            "spawns": names,
            "resolution": [width, height],
            "context_frames": context_frames,
            "budget_frames": max_generated_frames,
            "generated_frames": state["generated"],
            **(metadata or {}),
        }

    @app.websocket("/ws")
    async def play(ws: WebSocket):
        # Only a loopback viewer is supported by this entrypoint.
        origin = ws.headers.get("origin", "")
        if origin and origin not in ("http://127.0.0.1:7860", "http://localhost:7860"):
            await ws.close(code=1008)
            return
        try:
            spawn = int(ws.query_params.get("spawn", "0"))
            if not 0 <= spawn < len(seed_frames):
                raise ValueError("Unknown spawn")
        except ValueError:
            await ws.close(code=1008)
            return
        await ws.accept()
        context = seed_frames[spawn].copy()
        actions = seed_actions[spawn].copy()
        frame = 0
        started = time.monotonic()
        await ws.send_bytes(png(context[-1]))
        try:
            while True:
                raw = await ws.receive_text()
                if len(raw) > 2048:
                    await ws.send_json({"error": "Control message too large"})
                    continue
                try:
                    control = Control.model_validate_json(raw)
                except ValidationError:
                    await ws.send_json({"error": "Invalid control input"})
                    continue
                if control.type == "reset":
                    if control.spawn >= len(seed_frames):
                        await ws.send_json({"error": "Unknown spawn"})
                        continue
                    spawn = control.spawn
                    context = seed_frames[spawn].copy()
                    actions = seed_actions[spawn].copy()
                    frame = 0
                    await ws.send_json({"reset": True, "frame": 0})
                    await ws.send_bytes(png(context[-1]))
                    continue
                if state["generated"] >= max_generated_frames:
                    await ws.send_json(
                        {
                            "error": "Session frame budget reached. Restart the local viewer to allocate another session."
                        }
                    )
                    continue
                if time.monotonic() - started > 1200:
                    await ws.send_json(
                        {
                            "error": "20-minute connection limit reached. Reconnect to continue."
                        }
                    )
                    break
                action = encode(
                    control.keys, control.dx, control.dy, control.fire, control.scope
                )
                full_actions = np.concatenate((actions, action[None]), axis=0)
                now = time.perf_counter()
                async with semaphore:
                    if state["generated"] >= max_generated_frames:
                        await ws.send_json({"error": "Session frame budget reached."})
                        continue
                    prediction = await predict(
                        context,
                        full_actions,
                        control.steps,
                        1000 + spawn * 100000 + frame,
                    )
                    prediction = np.asarray(prediction, dtype=np.uint8)
                    if prediction.shape != (height, width, 3):
                        raise ValueError("Invalid predicted frame shape")
                    state["generated"] += 1
                context = np.concatenate((context[1:], prediction[None]), axis=0)
                actions = full_actions[1:]
                frame += 1
                await ws.send_json(
                    {
                        "frame": frame,
                        "ms": round((time.perf_counter() - now) * 1000, 1),
                        "remaining": max_generated_frames - state["generated"],
                    }
                )
                await ws.send_bytes(png(prediction))
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            # Do not expose remote tracebacks or environment details to the browser.
            print(f"Viewer error: {type(exc).__name__}", flush=True)
            try:
                await ws.send_json(
                    {"error": "Inference failed. See the local server log."}
                )
            except Exception:
                pass

    return app


def local_predict(checkpoint):
    import torch
    from .model import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    torch.set_num_threads(4)
    model, ckpt = load_model(checkpoint, device)

    def infer(context, actions, steps, seed):
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=dtype, enabled=device == "cuda"),
        ):
            c = (
                torch.from_numpy(context.copy())
                .to(device)
                .permute(0, 3, 1, 2)[None]
                .float()
                / 127.5
                - 1
            )
            a = torch.from_numpy(actions.copy()).to(device)[None]
            frame = model.sample(c, a, steps=steps, seed=seed)[0]
            return (
                frame.float()
                .add(1)
                .mul(127.5)
                .round()
                .clamp(0, 255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

    async def predict(*args):
        return await asyncio.to_thread(infer, *args)

    return predict, {
        "device": device,
        "resolution": [model.cfg.width,model.cfg.height],
        "context_frames": model.cfg.context,
        "checkpoint_step": ckpt["step"],
        "pretrained_weights": False,
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seeds", required=True)
    args = p.parse_args()
    predict, info = local_predict(args.checkpoint)
    uvicorn.run(make_app(args.seeds, predict, info), host="127.0.0.1", port=7860)
