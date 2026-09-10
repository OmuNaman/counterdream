"""Loopback browser proxy. Only controls and encoded frames cross the network."""
import asyncio
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import ValidationError

from .serve import Control


def make_cloud_app(metadata, remote_frame, frame_budget=12000, on_session_end=None):
    app = FastAPI(title="CounterDream Cloud", docs_url=None, redoc_url=None)
    static = Path(__file__).parent / "static"
    state = {"generated": 0}
    gate = asyncio.Semaphore(1)

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/app.js")
    def script():
        return FileResponse(static / "app.js", media_type="text/javascript")

    @app.get("/style.css")
    def style():
        return FileResponse(static / "style.css", media_type="text/css")

    @app.get("/api/info")
    def info():
        return dict(metadata, budget_frames=frame_budget, generated_frames=state["generated"])

    @app.websocket("/ws")
    async def play(ws: WebSocket):
        if ws.headers.get("origin", "") not in ("", "http://localhost:7860", "http://127.0.0.1:7860"):
            await ws.close(code=1008)
            return
        try:
            spawn = int(ws.query_params.get("spawn", "0"))
            if not 0 <= spawn < len(metadata["spawns"]):
                raise ValueError("Unknown spawn")
        except ValueError:
            await ws.close(code=1008)
            return
        session = uuid.uuid4().hex
        await ws.accept()
        started = time.monotonic()
        sequence = 0
        try:
            async with gate:
                first = await remote_frame(session, {"type":"reset", "spawn":spawn}, sequence)
            await ws.send_bytes(first["png"])
            while True:
                raw = await ws.receive_text()
                if len(raw) > 2048:
                    await ws.send_json({"error":"Control message too large"})
                    continue
                try:
                    control = Control.model_validate_json(raw)
                except ValidationError:
                    await ws.send_json({"error":"Invalid controls"})
                    continue
                if control.spawn >= len(metadata["spawns"]):
                    await ws.send_json({"error":"Unknown spawn"})
                    continue
                if time.monotonic()-started > 900:
                    await ws.send_json({"error":"15-minute session finished. Reconnect to continue."})
                    break
                tick = time.perf_counter()
                async with gate:
                    if control.type == "step" and state["generated"] >= frame_budget:
                        await ws.send_json({"error":"Viewer frame budget reached."})
                        continue
                    sequence += 1
                    result = await remote_frame(session,control.model_dump(),sequence)
                    if result.get("error"):
                        await ws.send_json({"error":result["error"]})
                        continue
                    if control.type == "step":
                        state["generated"] += 1
                await ws.send_json(dict(frame=result["frame"], reset=control.type=="reset",
                                        ms=round((time.perf_counter()-tick)*1000,1),
                                        gpu_ms=result.get("gpu_ms"),remaining=frame_budget-state["generated"]))
                await ws.send_bytes(result["png"])
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            print(f"Cloud viewer error: {type(exc).__name__}",flush=True)
            try:
                await ws.send_json({"error":"Cloud inference stopped. Reconnect to start a new session."})
            except Exception:
                pass
        finally:
            if on_session_end is not None:
                on_session_end(session)
    return app
