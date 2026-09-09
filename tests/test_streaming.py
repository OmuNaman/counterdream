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


def test_authenticated_stream_generates_without_per_frame_requests_and_stops():
    engine = Engine()
    app = make_gpu_stream(engine, "x"*40, {"last": time.monotonic()}, frame_budget=3)
    with TestClient(app) as client:
        with client.websocket_connect("/ws?spawn=1", headers={"authorization": "Bearer "+"x"*40}) as ws:
            assert ws.receive_json()["reset"]
            assert ws.receive_bytes() == b"example-frame"
            # One control produces several frames without a response/request cycle.
            ws.send_json(dict(type="step", keys=["w"], dx=30, steps=4))
            for remaining in (2, 1, 0):
                assert ws.receive_json()["remaining"] == remaining
                assert ws.receive_bytes() == b"example-frame"
            assert "allocation" in ws.receive_json()["error"]
    assert engine.calls[0][1]["spawn"] == 1
    assert [call[2] for call in engine.calls] == [1, 2, 3, 4]
    assert [call[1]["dx"] for call in engine.calls[1:]] == [30, 0, 0]
    assert all(call[1]["keys"] == ["w"] for call in engine.calls[1:])
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
