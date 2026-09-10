"""Record actual cloud delivery on a wall clock, without speeding up playback."""

import argparse
import asyncio
import io
import json
from pathlib import Path
import statistics
import time
import urllib.request

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import websockets

from .demo_record import sha256
from .play_sequence import play_controls


async def record(output, spawn=0, seconds=20, fps=16):
    if not 10 <= seconds <= 30:
        raise ValueError("Choose 10 to 30 seconds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sent, packets, records = {}, [], []
    async with websockets.connect(
        f"ws://127.0.0.1:7860/ws?spawn={spawn}", open_timeout=45, max_queue=4
    ) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), 600))
        if not ready.get("reset"):
            raise RuntimeError(ready.get("error", "Expected initial view"))
        seed = await ws.recv()
        packets.append(seed)
        records.append(
            dict(frame=0, received_seconds=0.0, label="STARTING VIEW", control={})
        )
        start = time.perf_counter()

        async def sender():
            sequence = 0
            while True:
                now = time.perf_counter()
                label, action = play_controls(int((now - start) * 16))
                sequence += 1
                sent[sequence] = now, label, action
                await ws.send(json.dumps(dict(type="step", steps=4, fps=fps, **action)))
                await asyncio.sleep(1 / fps)

        task = asyncio.create_task(sender())
        metadata = {}
        try:
            while time.perf_counter() - start < seconds:
                packet = await asyncio.wait_for(ws.recv(), 20)
                now = time.perf_counter()
                if isinstance(packet, str):
                    metadata = json.loads(packet)
                    if metadata.get("error"):
                        raise RuntimeError(metadata["error"])
                    continue
                sent_at, label, action = sent[metadata["control_seq"]]
                records.append(
                    dict(
                        **metadata,
                        received_seconds=now - start,
                        response_ms=(now - sent_at) * 1000,
                        label=label,
                        control=action,
                    )
                )
                packets.append(packet)
                if len(records) % 80 == 0:
                    print(
                        json.dumps(
                            dict(frames=len(records) - 1, seconds=round(now - start, 2))
                        ),
                        flush=True,
                    )
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await ws.send(json.dumps(dict(type="pause")))
    # Encoding after capture keeps disk/video work out of the receive loop.
    with urllib.request.urlopen("http://127.0.0.1:7860/api/info") as response:
        info = json.load(response)
    timestamps = np.asarray([r["received_seconds"] for r in records])
    playback_fps = 30
    duration = records[-1]["received_seconds"]
    last_index, last_image = -1, None
    with imageio.get_writer(
        output / "h100-live.mp4",
        fps=playback_fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    ) as writer:
        for i in range(int(np.ceil(duration * playback_fps))):
            t = i / playback_fps
            index = max(0, int(np.searchsorted(timestamps, t, side="right") - 1))
            if index != last_index:
                last_image = (
                    Image.open(io.BytesIO(packets[index]))
                    .convert("RGB")
                    .resize((960, 528), Image.Resampling.BICUBIC)
                )
                last_index = index
            canvas = Image.new("RGB", (960, 624), "#10151e")
            canvas.paste(last_image, (0, 48))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (16, 16),
                "COUNTERDREAM / ACTUAL CLOUD STREAM / REAL-TIME PLAYBACK",
                fill="white",
            )
            draw.text((16, 595), records[index]["label"], fill="#9cdbcb")
            draw.text(
                (620, 595),
                f'{t:04.1f}s | GENERATED FRAME {records[index]["frame"]}',
                fill="white",
            )
            writer.append_data(np.asarray(canvas))
    measured = records[9:]
    delays = [r["response_ms"] for r in measured]
    report = dict(
        frames=len(records) - 1,
        requested_fps=fps,
        spawn=spawn,
        actual_seconds=duration,
        delivered_fps=(len(measured) - 1)
        / (measured[-1]["received_seconds"] - measured[0]["received_seconds"]),
        gpu_ms_median=statistics.median(r["gpu_ms"] for r in measured),
        control_response_ms_median=statistics.median(delays),
        control_response_ms_p95=sorted(delays)[int(len(delays) * 0.95)],
        video_playback_fps=playback_fps,
        video_timing="Wall-clock delivery; existing frames held until next arrival. No interpolation or time compression.",
        resets_after_initial_seed=0,
        recorded_future_frames_injected=0,
        info=info,
        video_sha256=sha256(output / "h100-live.mp4"),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    (output / "frames.json").write_text(json.dumps(records, indent=2))
    sheet = Image.new("RGB", (960, 800), "#10151e")
    for j, t in enumerate((0, 3, 5, 5.5, 6, 7, 10, 12, 13, 16, 18, 20)):
        index = max(0, int(np.searchsorted(timestamps, t, side="right") - 1))
        frame = Image.open(io.BytesIO(packets[index])).convert("RGB").resize((320, 176))
        x, y = j % 3 * 320, j // 3 * 200
        sheet.paste(frame, (x, y + 24))
        ImageDraw.Draw(sheet).text(
            (x + 5, y + 5), f'{t}s: {records[index]["label"]}', fill="white"
        )
    sheet.save(output / "contact-sheet.jpg", quality=95)
    # Keep received bytes and timestamps so the video timing can be audited.
    np.savez_compressed(
        output / "received-packets.npz",
        **{
            f"frame_{i}": np.frombuffer(p, dtype=np.uint8)
            for i, p in enumerate(packets)
        },
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--spawn", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--fps", type=int, choices=(16, 24), default=16)
    asyncio.run(record(**vars(parser.parse_args())))
