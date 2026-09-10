import time

from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from counterdream.streaming import make_gpu_stream, make_stream_proxy


class Engine:
    def __init__(self):
        self.seeds = [None, None]
        self.sessions = {}
        self.calls = []

    def frame(self, session, command, sequence):
        self.calls.append((session, command, sequence))
        self.sessions[session] = {}
        return dict(png=b"example-frame", frame=sequence-1, gpu_ms=25)


@pytest.mark.parametrize('fps',[16,24])
def test_authenticated_stream_generates_without_per_frame_requests_and_stops(fps):
    engine = Engine()
    app = make_gpu_stream(engine, "x"*40, {"last": time.monotonic()}, frame_budget=3)
    with TestClient(app) as client:
        with client.websocket_connect("/ws?spawn=1", headers={"authorization": "Bearer "+"x"*40}) as ws:
            assert ws.receive_json()["reset"]
            assert ws.receive_bytes() == b"example-frame"
            # One control produces several frames without a response/request cycle.
            ws.send_json(dict(type="step", keys=["w"], dx=30, steps=4, fps=fps))
            for remaining in (2, 1, 0):
                assert ws.receive_json()["remaining"] == remaining
                assert ws.receive_bytes() == b"example-frame"
            assert "allocation" in ws.receive_json()["error"]
    assert engine.calls[0][1]["spawn"] == 1
    assert [call[2] for call in engine.calls] == [1, 2, 3, 4]
    assert [call[1]["dx"] for call in engine.calls[1:]] == [30, 0, 0]
    assert all(call[1]["keys"] == ["w"] for call in engine.calls[1:])
    assert all(call[1]['fps'] == fps for call in engine.calls[1:])
    assert not engine.sessions


def test_stream_rejects_invalid_authentication_and_spawn_before_model_use():
    engine = Engine()
    app = make_gpu_stream(engine, "x"*40, {"last": time.monotonic()})
    with TestClient(app) as client:
        for url, headers in (("/ws", {}), ("/ws?spawn=2", {"authorization": "Bearer "+"x"*40})):
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(url, headers=headers):
                    pass
    assert not engine.calls


def test_local_proxy_rejects_foreign_origin_before_gpu_allocation():
    async def remote():
        raise AssertionError("Must not allocate a GPU for a rejected connection")
    with TestClient(make_stream_proxy(dict(spawns=["One"]), remote)) as client:
        assert client.get("/api/info").json()["streaming"]
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws", headers={"origin": "https://example.com"}):
                    pass


def test_live_metadata_sets_sampler_and_expired_allocation_explains_failure():
    from counterdream.stream_lease import AllocationEnded
    async def remote():
        raise AllocationEnded("This 30-minute viewer allocation has ended.")
    with TestClient(make_stream_proxy(dict(spawns=["Val 1"], recommended_steps=8,
                                          checkpoint_step=34000), remote)) as client:
        info = client.get("/api/info").json()
        assert info["recommended_steps"] == 8 and info["checkpoint_step"] == 34000
        with client.websocket_connect("/ws") as ws:
            assert "30-minute" in ws.receive_json()["error"]


def test_refresh_replaces_old_connection_and_reuses_pending_gpu_start(monkeypatch):
    import asyncio
    import json
    import threading
    import websockets

    starting, release = threading.Event(), threading.Event()
    starts=[]
    async def get_remote():
        starts.append(1)
        starting.set()
        while not release.is_set():
            await asyncio.sleep(.005)
        return 'https://example.modal.host','x'*40

    class Remote:
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def send(self,raw):pass
        async def messages(self):
            yield json.dumps({'reset':True,'frame':0})
            yield b'frame-from-existing-gpu'
            await asyncio.Event().wait()
        def __aiter__(self):return self.messages()

    monkeypatch.setattr(websockets,'connect',lambda *args,**kwargs:Remote())
    with TestClient(make_stream_proxy(dict(spawns=['One']),get_remote)) as client:
        with client.websocket_connect('/ws') as old:
            assert starting.wait(2)
            # New browser connection arrives before the GPU has finished loading.
            with client.websocket_connect('/ws') as fresh:
                release.set()
                assert fresh.receive_json()['reset']
                assert fresh.receive_bytes()==b'frame-from-existing-gpu'
                assert starts==[1]
        # A later reconnect also works after both previous sockets have closed.
        with client.websocket_connect('/ws') as later:
            assert later.receive_json()['reset']
            assert later.receive_bytes()==b'frame-from-existing-gpu'
