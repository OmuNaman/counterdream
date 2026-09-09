"""Continuous authenticated GPU streaming, decoupled from control round trips."""
import asyncio
from contextlib import suppress
import hmac
import json
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid

from fastapi import FastAPI,WebSocket,WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import ValidationError

from .serve import Control


def make_gpu_stream(engine,token,activity,frame_budget=12000):
    if len(token)<32:
        raise ValueError("A strong session token is required")
    app=FastAPI(docs_url=None,redoc_url=None)
    total={"frames":0,"connected":False}

    @app.websocket("/ws")
    async def play(ws:WebSocket):
        if not hmac.compare_digest(ws.headers.get("authorization",""),"Bearer "+token):
            await ws.close(code=1008)
            return
        try:
            spawn=int(ws.query_params.get("spawn","0"))
            if not 0<=spawn<len(engine.seeds):
                raise ValueError("Unknown spawn")
        except ValueError:
            await ws.close(code=1008)
            return
        if total["connected"]:
            await ws.close(code=1013)
            return
        total["connected"]=True
        await ws.accept()
        session=uuid.uuid4().hex
        state=dict(paused=True,reset=spawn,control=Control(steps=4),dx=0.,dy=0.,seq=0,
                   seen=time.monotonic(),closed=False)
        activity["last"]=time.monotonic()
        started=time.monotonic()

        async def receive():
            try:
                while True:
                    raw=await ws.receive_text()
                    if len(raw)>2048:
                        continue
                    try:
                        data=json.loads(raw)
                        if not isinstance(data,dict):
                            continue
                        if data.get("type")=="pause":
                            state["paused"]=True
                            state["dx"]=state["dy"]=0.
                            continue
                        control=Control.model_validate(data)
                    except (ValueError,ValidationError):
                        continue
                    state["seen"]=time.monotonic()
                    activity["last"]=state["seen"]
                    state["seq"]+=1
                    if control.type=="reset":
                        if control.spawn>=len(engine.seeds):
                            continue
                        state["reset"]=control.spawn
                        state["paused"]=True
                        state["dx"]=state["dy"]=0.
                    else:
                        state["control"]=control
                        state["dx"]+=control.dx
                        state["dy"]+=control.dy
                        state["paused"]=False
            except WebSocketDisconnect:
                state["closed"]=True

        receiver=asyncio.create_task(receive())
        sequence=0
        try:
            while not state["closed"] and time.monotonic()-started<900:
                tick=time.monotonic()
                if state["reset"] is not None:
                    spawn=state["reset"]
                    state["reset"]=None
                    sequence+=1
                    result=await asyncio.to_thread(engine.frame,session,{"type":"reset","spawn":spawn},sequence)
                    if result.get("error"):
                        await ws.send_json({"error":result["error"]})
                        break
                    await ws.send_json({"reset":True,"frame":0,"remaining":frame_budget-total["frames"]})
                    await ws.send_bytes(result["png"])
                elif not state["paused"] and tick-state["seen"]<.5:
                    if total["frames"]>=frame_budget:
                        await ws.send_json({"error":"Cloud frame allocation finished."})
                        break
                    command=state["control"].model_dump()
                    command.update(dx=max(-1000,min(1000,state["dx"])),dy=max(-200,min(200,state["dy"])))
                    state["dx"]=state["dy"]=0.
                    control_seq=state["seq"]
                    sequence+=1
                    result=await asyncio.to_thread(engine.frame,session,command,sequence)
                    if result.get("error"):
                        await ws.send_json({"error":result["error"]})
                        break
                    total["frames"]+=1
                    activity["last"]=time.monotonic()
                    await ws.send_json(dict(frame=result["frame"],gpu_ms=result["gpu_ms"],
                                            control_seq=control_seq,remaining=frame_budget-total["frames"]))
                    await ws.send_bytes(result["png"])
                await asyncio.sleep(max(.001,1/16-(time.monotonic()-tick)))
        except WebSocketDisconnect:
            pass
        finally:
            receiver.cancel()
            with suppress(asyncio.CancelledError,WebSocketDisconnect):
                await receiver
            total["connected"]=False
            engine.sessions.pop(session,None)
            activity["last"]=time.monotonic()
            with suppress(Exception):
                await ws.close()
    return app


def make_stream_proxy(metadata,get_remote):
    """get_remote returns a trusted tunnel URL and token held only in this process."""
    import websockets
    app=FastAPI(docs_url=None,redoc_url=None)
    static=Path(__file__).parent/"static"

    @app.get("/")
    def index():
        return FileResponse(static/"index.html")

    @app.get("/app.js")
    def script():
        return FileResponse(static/"app.js",media_type="text/javascript")

    @app.get("/style.css")
    def style():
        return FileResponse(static/"style.css",media_type="text/css")

    @app.get("/api/info")
    def info():
        return dict(metadata,streaming=True,recommended_steps=4,budget_frames=12000,generated_frames=0)

    @app.websocket("/ws")
    async def play(ws:WebSocket):
        if ws.headers.get("origin","") not in ("","http://localhost:7860","http://127.0.0.1:7860"):
            await ws.close(code=1008)
            return
        try:
            spawn=int(ws.query_params.get("spawn","0"))
            if not 0<=spawn<len(metadata["spawns"]):
                raise ValueError("Unknown spawn")
        except ValueError:
            await ws.close(code=1008)
            return
        await ws.accept()
        tasks=[]
        try:
            url,token=await get_remote()
            parsed=urlparse(url)
            if parsed.scheme!="https" or not (parsed.hostname or "").endswith(".modal.host"):
                raise ValueError("Unexpected tunnel destination")
            async with websockets.connect(url.replace("https://","wss://",1)+f"/ws?spawn={spawn}",
                                          additional_headers={"Authorization":"Bearer "+token},
                                          open_timeout=45,max_size=2_000_000,max_queue=4) as remote:
                async def controls():
                    while True:
                        raw=await ws.receive_text()
                        if len(raw)<=2048:
                            await remote.send(raw)
                async def frames():
                    async for data in remote:
                        if isinstance(data,bytes):
                            await ws.send_bytes(data)
                        else:
                            await ws.send_text(data)
                tasks=[asyncio.create_task(controls()),asyncio.create_task(frames())]
                await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            print(f"Streaming connection: {type(exc).__name__}",flush=True)
            with suppress(Exception):
                await ws.send_json({"error":"Cloud stream stopped. Restart the viewer to allocate another session."})
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks,return_exceptions=True)
            with suppress(Exception):
                await ws.close()
    return app
