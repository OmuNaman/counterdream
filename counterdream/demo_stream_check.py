"""Bounded real cloud-stream recording with actual applied-control metadata."""

import argparse
import asyncio
import io
import json
from pathlib import Path
import statistics
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import websockets

from .demo_record import controls, sha256


async def check(output, spawn=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sent = {}
    times = []
    delays = []
    gpu = []
    records = []
    shots = []
    async with websockets.connect(
        f"ws://127.0.0.1:7860/ws?spawn={spawn}", open_timeout=45, max_queue=4
    ) as ws:
        info = json.loads(await asyncio.wait_for(ws.recv(), 600))
        if not info.get("reset"):
            raise RuntimeError(info.get("error", "Expected initial view"))
        await ws.recv()
        start = time.perf_counter()

        async def sender():
            seq = 0
            while True:
                label, control = controls(
                    min(479, int((time.perf_counter() - start) * 16))
                )
                seq += 1
                sent[seq] = (time.perf_counter(), label, control)
                await ws.send(json.dumps(dict(type="step", steps=4, **control)))
                await asyncio.sleep(1 / 16)

        task = asyncio.create_task(sender())
        metadata = {}
        try:
            with imageio.get_writer(
                output / "cloud-session.mp4",
                fps=16,
                codec="libx264",
                quality=9,
                macro_block_size=2,
            ) as writer:
                while len(times) < 480:
                    packet = await asyncio.wait_for(ws.recv(), 20)
                    now = time.perf_counter()
                    if isinstance(packet, str):
                        metadata = json.loads(packet)
                        if metadata.get("error"):
                            raise RuntimeError(metadata["error"])
                        continue
                    frame = np.asarray(Image.open(io.BytesIO(packet)).convert("RGB"))
                    if frame.shape != (352, 640, 3):
                        raise ValueError("Expected 4x display image")
                    times.append(now)
                    seq = metadata["control_seq"]
                    sent_at, label, control = sent[seq]
                    delays.append((now - sent_at) * 1000)
                    gpu.append(metadata["gpu_ms"])
                    records.append(
                        dict(**metadata, control=control, received_seconds=now - start)
                    )
                    writer.append_data(frame)
                    if len(times) in (1, 32, 96, 144, 240, 288, 400, 480):
                        shots.append((frame, label, len(times)))
                    if len(times) % 96 == 0:
                        print(
                            json.dumps(
                                dict(received=len(times), seconds=round(now - start, 1))
                            ),
                            flush=True,
                        )
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await ws.send(json.dumps(dict(type="pause")))
    report = dict(
        frames=len(times),
        display_resolution=[640, 352],
        spawn=spawn,
        seconds=times[-1] - start,
        delivered_fps=(len(times) - 9) / (times[-1] - times[8]),
        gpu_ms_median=statistics.median(gpu[8:]),
        control_response_ms_median=statistics.median(delays[8:]),
        control_response_ms_p95=sorted(delays[8:])[int(0.95 * len(delays[8:]))],
        resets_after_initial_seed=0,
        recorded_future_frames_injected=0,
        protocol="Actual cloud streaming; controls scheduled by wall time. Stored metadata records controls applied by the GPU.",
        video_playback_fps=16,
        video_sha256=sha256(output / "cloud-session.mp4"),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (output / "frames.json").write_text(json.dumps(records, indent=2))
    sheet = Image.new("RGB", (640, 4 * 208), "#10151e")
    for i, (frame, label, index) in enumerate(shots):
        x = i % 2 * 320
        y = i // 2 * 208
        sheet.paste(Image.fromarray(frame).resize((320, 176)), (x, y + 24))
        ImageDraw.Draw(sheet).text(
            (x + 6, y + 6), f"Frame {index} | {label}", fill="white"
        )
    sheet.save(output / "contact-sheet.png")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--spawn", type=int, default=0)
    asyncio.run(check(**vars(p.parse_args())))
