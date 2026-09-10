from fastapi.testclient import TestClient
import numpy as np

from counterdream.cloud_viewer import make_cloud_app
from counterdream.serve import png


def test_cloud_viewer_sends_controls_without_context_and_stops_at_budget():
    received = []
    ended = []
    async def remote(session, command, sequence):
        assert "context" not in command
        received.append((session, command, sequence))
        return dict(png=png(np.zeros((88,160,3),np.uint8)),frame=sequence,gpu_ms=12)
    metadata = dict(spawns=["One"],resolution=[160,88],context_frames=8)
    app = make_cloud_app(metadata,remote,frame_budget=1,on_session_end=ended.append)
    with TestClient(app) as client:
        with client.websocket_connect("/ws?spawn=0", headers={"origin":"http://127.0.0.1:7860"}) as ws:
            assert ws.receive_bytes().startswith(b"\x89PNG")
            ws.send_json(dict(type="step",keys=["w"],dx=-30))
            result = ws.receive_json()
            assert result["remaining"] == 0
            assert result["gpu_ms"] == 12
            ws.receive_bytes()
            ws.send_json(dict(type="step"))
            assert "budget" in ws.receive_json()["error"]
            ws.send_json(dict(type="reset",spawn=0))
            assert ws.receive_json()["reset"]
            ws.receive_bytes()
        assert len(received) == 3
        assert len({x[0] for x in received}) == 1
        assert ended == [received[0][0]]
        assert [x[2] for x in received] == [0,1,2]
        assert client.get("/api/info").json()["generated_frames"] == 1
