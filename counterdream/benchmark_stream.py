"""Measure an already-running viewer with bounded, scripted control traffic."""
import asyncio
import json
from pathlib import Path
import statistics
import time

import websockets


async def measure(url="ws://127.0.0.1:7860/ws?spawn=0", headers=None):
    sent, times, responses, gpu = {}, [], [], []
    async with websockets.connect(url, additional_headers=headers, open_timeout=45) as ws:
        initial = json.loads(await asyncio.wait_for(ws.recv(), timeout=600))
        if not initial.get("reset"):
            raise RuntimeError(initial.get("error", "Expected starting frame"))
        await asyncio.wait_for(ws.recv(), timeout=15)
        started = time.perf_counter()

        async def controls():
            seq = 0
            while True:
                seq += 1
                sent[seq] = time.perf_counter()
                await ws.send(json.dumps(dict(type="step", keys=["w"], dx=10, steps=4)))
                await asyncio.sleep(1/16)

        task = asyncio.create_task(controls())
        try:
            while len(times) < 96:
                data = await asyncio.wait_for(ws.recv(), timeout=15)
                now = time.perf_counter()
                if isinstance(data, bytes):
                    times.append(now)
                else:
                    data = json.loads(data)
                    if data.get("error"):
                        raise RuntimeError(data["error"])
                    seq = data.get("control_seq")
                    if seq in sent:
                        responses.append((now-sent[seq])*1000)
                        gpu.append(data["gpu_ms"])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await ws.send(json.dumps(dict(type="pause")))
    gaps = [(b-a)*1000 for a,b in zip(times[8:-1], times[9:])]
    return dict(frames=len(times), frame_ms_median=statistics.median(gaps),
                delivered_fps=87/(times[-1]-times[8]),
                first_control_response_ms=(times[0]-started)*1000,
                control_response_ms_median=statistics.median(responses[8:]),
                control_response_ms_p95=sorted(responses[8:])[int(.95*len(responses[8:]))],
                gpu_ms_median=statistics.median(gpu[8:]))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/network-stream-proxy.json"))
    args = parser.parse_args()
    report = dict(asyncio.run(measure()), transport="local proxy and authenticated cloud TLS websocket")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
