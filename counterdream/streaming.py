"""Continuous authenticated GPU streaming, decoupled from control round trips."""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager, suppress
import hmac
import json
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import ValidationError

from .serve import Control
from .stream_lease import AllocationEnded


def make_gpu_stream(engine, token, activity, frame_budget=12000):
    if len(token) < 32:
        raise ValueError("A strong session token is required")
    app = FastAPI(docs_url=None, redoc_url=None)
    total = {"frames": 0, "connected": False}

    @app.websocket("/ws")
    async def play(ws: WebSocket):
        if not hmac.compare_digest(
            ws.headers.get("authorization", ""), "Bearer " + token
        ):
            await ws.close(code=1008)
            return
        try:
            spawn = int(ws.query_params.get("spawn", "0"))
            if not 0 <= spawn < len(engine.seeds):
                raise ValueError("Unknown spawn")
        except ValueError:
            await ws.close(code=1008)
            return
        if total["connected"]:
            await ws.close(code=1013)
            return
        total["connected"] = True
        await ws.accept()
        session = uuid.uuid4().hex
        state = dict(
            paused=True,
            reset=spawn,
            control=Control(steps=4),
            dx=0.0,
            dy=0.0,
            seq=0,
            pending_keys=set(),
            pending_fire=False,
            pending_scope=False,
            seen=time.monotonic(),
            closed=False,
        )
        activity["last"] = time.monotonic()
        started = time.monotonic()

        async def receive():
            try:
                while True:
                    raw = await ws.receive_text()
                    if len(raw) > 2048:
                        continue
                    try:
                        data = json.loads(raw)
                        if not isinstance(data, dict):
                            continue
                        if data.get("type") == "pause":
                            state["paused"] = True
                            state["dx"] = state["dy"] = 0.0
                            state["pending_keys"].clear()
                            state["pending_fire"] = state["pending_scope"] = False
                            state["control"] = Control(
                                steps=state["control"].steps, fps=state["control"].fps
                            )
                            continue
                        control = Control.model_validate(data)
                    except (ValueError, ValidationError):
                        continue
                    state["seen"] = time.monotonic()
                    activity["last"] = state["seen"]
                    state["seq"] += 1
                    if control.type == "reset":
                        if control.spawn >= len(engine.seeds):
                            continue
                        state["reset"] = control.spawn
                        state["paused"] = True
                        state["dx"] = state["dy"] = 0.0
                        state["pending_keys"].clear()
                        state["pending_fire"] = state["pending_scope"] = False
                        state["control"] = Control(steps=control.steps, fps=control.fps)
                    else:
                        previous = state["control"]
                        state["pending_keys"].update(
                            set(control.keys) - set(previous.keys)
                        )
                        state["pending_fire"] |= control.fire and not previous.fire
                        state["pending_scope"] |= control.scope and not previous.scope
                        state["control"] = control
                        state["dx"] += control.dx
                        state["dy"] += control.dy
                        state["paused"] = False
            except WebSocketDisconnect:
                pass
            finally:
                state["closed"] = True

        receiver = asyncio.create_task(receive())
        sequence = 0
        try:
            while not state["closed"] and time.monotonic() - started < 900:
                tick = time.monotonic()
                if state["reset"] is not None:
                    spawn = state["reset"]
                    state["reset"] = None
                    sequence += 1
                    result = await asyncio.to_thread(
                        engine.frame,
                        session,
                        {"type": "reset", "spawn": spawn},
                        sequence,
                    )
                    if result.get("error"):
                        await ws.send_json({"error": result["error"]})
                        break
                    await ws.send_json(
                        {
                            "reset": True,
                            "frame": 0,
                            "remaining": frame_budget - total["frames"],
                        }
                    )
                    await ws.send_bytes(result["png"])
                elif not state["paused"] and tick - state["seen"] < 0.5:
                    if total["frames"] >= frame_budget:
                        await ws.send_json(
                            {"error": "Cloud frame allocation finished."}
                        )
                        break
                    command = state["control"].model_dump()
                    command.update(
                        dx=max(-1000, min(1000, state["dx"])),
                        dy=max(-200, min(200, state["dy"])),
                    )
                    command["keys"] = sorted(
                        set(command["keys"]) | state["pending_keys"]
                    )
                    command["fire"] |= state["pending_fire"]
                    command["scope"] |= state["pending_scope"]
                    state["pending_keys"].clear()
                    state["pending_fire"] = state["pending_scope"] = False
                    state["dx"] = state["dy"] = 0.0
                    control_seq = state["seq"]
                    sequence += 1
                    result = await asyncio.to_thread(
                        engine.frame, session, command, sequence
                    )
                    if result.get("error"):
                        await ws.send_json({"error": result["error"]})
                        break
                    total["frames"] += 1
                    activity["last"] = time.monotonic()
                    applied = Control.model_validate(command)
                    mx, my = applied.mouse_delta()
                    await ws.send_json(
                        dict(
                            frame=result["frame"],
                            gpu_ms=result["gpu_ms"],
                            input_id=applied.input_id,
                            client_time_ms=applied.client_time_ms,
                            applied=dict(
                                keys=applied.keys,
                                fire=applied.fire,
                                scope=applied.scope,
                                dx=mx,
                                dy=my,
                            ),
                            control_seq=control_seq,
                            remaining=frame_budget - total["frames"],
                        )
                    )
                    await ws.send_bytes(result["png"])
                await asyncio.sleep(
                    max(0.001, 1 / state["control"].fps - (time.monotonic() - tick))
                )
        except WebSocketDisconnect:
            pass
        finally:
            total["connected"] = False
            engine.sessions.pop(session, None)
            activity["last"] = time.monotonic()
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            with suppress(Exception):
                await ws.close()

    return app


def make_stream_proxy(metadata, get_remote):
    """get_remote returns a trusted tunnel URL and token held only in this process."""
    import websockets

    app = FastAPI(docs_url=None, redoc_url=None)
    static = Path(__file__).parent / "static"
    connection = {"task": None, "startup": None}
    handoff = asyncio.Lock()

    async def destination():
        # A refresh must not cancel a GPU that is still starting, or consume
        # another paid start for the same connection attempt.
        if connection["startup"] is None or connection["startup"].done():
            connection["startup"] = asyncio.create_task(get_remote())
        return await asyncio.shield(connection["startup"])

    @asynccontextmanager
    async def remote_connection(url, token, spawn):
        from websockets.exceptions import InvalidStatus

        async with AsyncExitStack() as cleanup:
            for attempt in range(10):
                try:
                    remote = await cleanup.enter_async_context(
                        websockets.connect(
                            url.replace("https://", "wss://", 1) + f"/ws?spawn={spawn}",
                            additional_headers={"Authorization": "Bearer " + token},
                            open_timeout=45,
                            close_timeout=1,
                            max_size=2_000_000,
                            max_queue=4,
                        )
                    )
                    break
                except InvalidStatus as exc:
                    # The prior GPU session releases its slot after an in-flight
                    # prediction completes. Retry the same authenticated endpoint.
                    if exc.response.status_code not in (403, 409, 503) or attempt == 9:
                        raise
                    await asyncio.sleep(0.15)
            yield remote

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/app.js")
    def script():
        return FileResponse(static / "app.js", media_type="text/javascript")

    @app.get("/style.css")
    def style():
        return FileResponse(static / "style.css", media_type="text/css")

    @app.get("/client-core.js")
    def client_core():
        return FileResponse(static / "client-core.js", media_type="text/javascript")

    @app.get("/api/info")
    def info():
        return dict(
            metadata,
            streaming=True,
            recommended_steps=metadata.get("recommended_steps", 4),
            budget_frames=12000,
            generated_frames=0,
        )

    @app.websocket("/ws")
    async def play(ws: WebSocket):
        if ws.headers.get("origin", "") not in (
            "",
            "http://localhost:7860",
            "http://127.0.0.1:7860",
        ):
            print("Stream handshake rejected: foreign origin", flush=True)
            await ws.close(code=1008)
            return
        try:
            spawn = int(ws.query_params.get("spawn", "0"))
            if not 0 <= spawn < len(metadata["spawns"]):
                raise ValueError("Unknown spawn")
        except ValueError:
            await ws.close(code=1008)
            return
        owner = {"task": asyncio.current_task(), "closed": asyncio.Event()}
        async with handoff:
            previous = connection["task"]
            if previous is not None and not previous["closed"].is_set():
                previous["task"].cancel()
                try:
                    await asyncio.wait_for(previous["closed"].wait(), timeout=5)
                except TimeoutError:
                    await ws.accept()
                    await ws.send_json(
                        {
                            "error": "Previous connection is closing. Reconnect in a moment."
                        }
                    )
                    await ws.close(code=1013)
                    return
            connection["task"] = owner
        tasks = []
        try:
            # Include acceptance in cleanup: a browser can close during its
            # handshake, before a frame-forwarding task exists.
            await ws.accept()
            url, token = await destination()
            parsed = urlparse(url)
            if parsed.scheme != "https" or not (parsed.hostname or "").endswith(
                ".modal.host"
            ):
                raise ValueError("Unexpected tunnel destination")
            async with remote_connection(url, token, spawn) as remote:

                async def controls():
                    while True:
                        raw = await ws.receive_text()
                        if len(raw) <= 2048:
                            await remote.send(raw)

                async def frames():
                    async for data in remote:
                        if isinstance(data, bytes):
                            await ws.send_bytes(data)
                        else:
                            await ws.send_text(data)

                tasks = [asyncio.create_task(controls()), asyncio.create_task(frames())]
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # A newly accepted local connection takes over this handler.
            pass
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            print(f"Streaming connection: {type(exc).__name__}", flush=True)
            with suppress(Exception):
                await ws.send_json(
                    {
                        "error": (
                            str(exc)
                            if isinstance(exc, AllocationEnded)
                            else "Cloud stream stopped. Press Reconnect to try again."
                        )
                    }
                )
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            with suppress(Exception):
                await asyncio.wait_for(ws.close(), timeout=1)
            if connection["task"] is owner:
                connection["task"] = None
            owner["closed"].set()

    return app
