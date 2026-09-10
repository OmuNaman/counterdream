import numpy as np
import pytest
from fastapi.testclient import TestClient
from counterdream.serve import make_app


def app_for_test(tmp_path, budget=2, shape=(4,64,112,3)):
    seeds = tmp_path / "seeds.npz"
    np.savez(
        seeds,
        frames=np.zeros((1,*shape), np.uint8),
        actions=np.zeros((1,shape[0]-1,51), np.float32),
        names=np.array(["Test seed"]),
    )
    seen = []

    async def predict(context, actions, steps, seed):
        seen.append((context.copy(), actions.copy()))
        return np.minimum(context[-1].astype(int) + 10, 255).astype(np.uint8)

    return make_app(seeds, predict, max_generated_frames=budget), seen


@pytest.mark.parametrize("shape",[(4,64,112,3),(8,88,160,3)])
def test_viewer_feeds_predictions_back_and_limits_calls(tmp_path,shape):
    app, seen = app_for_test(tmp_path,shape=shape)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/info").json()["budget_frames"] == 2
        assert client.get("/api/info").json()["resolution"] == [shape[2],shape[1]]
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_bytes().startswith(b"\x89PNG")
            for i in range(2):
                ws.send_json({"type": "step", "keys": ["w"]})
                assert ws.receive_json()["frame"] == i + 1
                ws.receive_bytes()
            ws.send_json({"type": "step"})
            assert "budget" in ws.receive_json()["error"]
    assert seen[1][0][-1].mean() == 10
    assert seen[0][1][-1, 0] == 1


def test_invalid_controls_and_reset(tmp_path):
    app, seen = app_for_test(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_bytes()
            ws.send_json({"dx": 99999})
            assert "Invalid" in ws.receive_json()["error"]
            ws.send_json({"type": "reset", "spawn": 2})
            assert "Unknown" in ws.receive_json()["error"]
            ws.send_json({"type": "reset", "spawn": 0})
            assert ws.receive_json()["reset"]
            assert ws.receive_bytes().startswith(b"\x89PNG")
    assert not seen
