"""Bounded H100 comparison of full-strength motion and learned firing response."""

import json
from pathlib import Path

import modal

from counterdream.cloud_config import RUN, gpu_base_image, volume

app = modal.App("counterdream-h100-playtest")
image = gpu_base_image.add_local_python_source("counterdream")


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/artifacts": volume},
)
def compare():
    import io
    import os
    import statistics
    import time
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from dataclasses import replace
    from counterdream.remote_runtime import SessionEngine
    from counterdream.play_sequence import play_controls
    from counterdream.demo_record import sha256

    volume.reload()
    source = Path(RUN, "demo-upgrade-v1")
    root = Path(RUN, "h100-playtest-v1")
    root.mkdir(exist_ok=False)
    engine = SessionEngine(
        source / "model.pt", source / "seeds.npz", source / "profile.json"
    )
    base = engine.options
    reports = []
    started = time.monotonic()
    for name, scale, center, fire in [
        ("centered-1", 1.0, True, 1.0),
        ("centered-1p5", 1.5, True, 1.0),
        ("centered-fire", 1.0, True, 2.5),
        ("uncentered-0p7", 0.7, False, 1.0),
    ]:
        out = root / name
        out.mkdir()
        engine.settings.update(motion_scale=scale, motion_center=center)
        engine.options = replace(base, fire_guidance=fire)
        profile = dict(
            engine.settings,
            sampling=vars(engine.options),
            note="Full-strength movement comparison; inspect motion_scale and motion_center.",
        )
        (out / "profile.json").write_text(json.dumps(profile, indent=2))
        session = "1" * 32
        engine.frame(session, {"type": "reset", "spawn": 0}, 0)
        frames, shots, timings = [], [], []
        tick = time.monotonic()
        with imageio.get_writer(
            out / "generated.mp4",
            fps=16,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        ) as writer:
            for i in range(320):
                if time.monotonic() - started > 780:
                    raise TimeoutError("Experiment compute bound reached")
                label, action = play_controls(i)
                result = engine.frame(
                    session, dict(type="step", steps=4, **action), i + 1
                )
                pixels = np.asarray(
                    Image.open(io.BytesIO(result["png"])).convert("RGB")
                )
                native = (
                    engine.sessions[session]["context"][0, -1]
                    .float()
                    .add(1)
                    .mul(127.5)
                    .clamp(0, 255)
                    .byte()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                frames.append(native)
                timings.append(result["gpu_ms"])
                canvas = Image.new("RGB", (640, 400), "#10151e")
                canvas.paste(Image.fromarray(pixels), (0, 24))
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (8, 6), f"H100 TEST / {name} / generated frame {i+1}", fill="white"
                )
                draw.text((8, 382), label, fill="#9cdbcb")
                writer.append_data(np.asarray(canvas))
                if i + 1 in (1, 48, 80, 88, 96, 112, 160, 192, 208, 256, 288, 320):
                    shots.append((pixels, label, i + 1))
        np.savez_compressed(out / "native-frames.npz", frames=np.asarray(frames))
        sheet = Image.new("RGB", (960, 4 * 200), "#10151e")
        for j, (pixels, label, index) in enumerate(shots):
            x, y = j % 3 * 320, j // 3 * 200
            sheet.paste(Image.fromarray(pixels).resize((320, 176)), (x, y + 24))
            ImageDraw.Draw(sheet).text(
                (x + 5, y + 5), f"{index}: {label}", fill="white"
            )
        sheet.save(out / "contact-sheet.jpg", quality=95)
        report = dict(
            name=name,
            frames=320,
            spawn=0,
            gpu=torch.cuda.get_device_name(),
            region=os.getenv("MODAL_REGION"),
            gpu_ms_median=statistics.median(timings[8:]),
            generation_seconds=time.monotonic() - tick,
            video_playback_fps=16,
            real_time_stream=False,
            resets_after_initial_seed=0,
            model_sha256=sha256(source / "model.pt"),
            profile=profile,
            video_sha256=sha256(out / "generated.mp4"),
        )
        (out / "report.json").write_text(json.dumps(report, indent=2))
        reports.append(report)
        engine.sessions.clear()
        volume.commit()
        print(json.dumps(report), flush=True)
    (root / "complete.json").write_text(json.dumps(reports, indent=2))
    volume.commit()
    return reports


@app.local_entrypoint()
def run():
    print(json.dumps(compare.remote()), flush=True)


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/artifacts": volume},
)
def firing_check():
    import io
    import time
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
    from dataclasses import replace
    from counterdream.remote_runtime import SessionEngine

    volume.reload()
    source = Path(RUN, "demo-upgrade-v1")
    root = Path(RUN, "h100-firing-v1")
    root.mkdir(exist_ok=False)
    engine = SessionEngine(
        source / "model.pt", source / "seeds.npz", source / "profile.json"
    )
    engine.settings.update(motion_scale=1.0, motion_center=True)
    started = time.monotonic()
    reports = []
    for spawn in (0, 4):
        for sigma in (0.3, 0.7, 1.5, 20.0):
            out = root / f"spawn{spawn}-sigma{sigma}"
            out.mkdir()
            engine.options = replace(
                engine.options,
                sigma_max=sigma,
                fire_guidance=1.0,
                warm_start=sigma < 20,
            )
            session = "2" * 32
            engine.frame(session, dict(type="reset", spawn=spawn), 0)
            shots = []
            with imageio.get_writer(
                out / "fire.mp4", fps=16, codec="libx264", quality=8, macro_block_size=2
            ) as writer:
                for i in range(64):
                    if time.monotonic() - started > 240:
                        raise TimeoutError("Firing test bound")
                    result = engine.frame(
                        session, dict(type="step", steps=4, fire=True), i + 1
                    )
                    pixels = np.asarray(
                        Image.open(io.BytesIO(result["png"])).convert("RGB")
                    )
                    writer.append_data(pixels)
                    if i + 1 in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
                        shots.append((pixels, i + 1))
            sheet = Image.new("RGB", (960, 800), "#10151e")
            for j, (pixels, index) in enumerate(shots):
                x, y = j % 3 * 320, j // 3 * 200
                sheet.paste(Image.fromarray(pixels).resize((320, 176)), (x, y + 24))
                ImageDraw.Draw(sheet).text(
                    (x + 5, y + 5), f"FIRE frame {index} / sigma {sigma}", fill="white"
                )
            sheet.save(out / "contact-sheet.jpg", quality=95)
            reports.append(dict(name=out.name, spawn=spawn, sigma_max=sigma, frames=64))
            engine.sessions.clear()
            print(json.dumps(reports[-1]), flush=True)
    (root / "complete.json").write_text(json.dumps(reports, indent=2))
    volume.commit()
    return reports


@app.local_entrypoint()
def fire():
    print(json.dumps(firing_check.remote()), flush=True)


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=300,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/artifacts": volume},
)
def weapon_check():
    import io
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw
    from counterdream.remote_runtime import SessionEngine
    from counterdream.play_sequence import play_controls

    volume.reload()
    source = Path(RUN, "demo-responsive-v1")
    root = Path(RUN, "h100-weapon-v1")
    root.mkdir(exist_ok=False)
    engine = SessionEngine(
        source / "model.pt", source / "seeds.npz", source / "profile.json"
    )
    reports = []
    for strength in (0.35, 0.7):
        out = root / f"strength{strength}"
        out.mkdir()
        engine.settings["weapon_refinement"] = strength
        session = "3" * 32
        engine.frame(session, dict(type="reset", spawn=0), 0)
        shots = []
        with imageio.get_writer(
            out / "weapon.mp4", fps=16, codec="libx264", quality=8, macro_block_size=2
        ) as writer:
            for i in range(320):
                label, action = play_controls(i)
                result = engine.frame(
                    session, dict(type="step", steps=4, **action), i + 1
                )
                pixels = np.asarray(
                    Image.open(io.BytesIO(result["png"])).convert("RGB")
                )
                writer.append_data(pixels)
                if i + 1 in (48, 80, 84, 88, 96, 112, 160, 176, 192, 208, 288, 320):
                    shots.append((pixels, i + 1, label))
        sheet = Image.new("RGB", (960, 800), "#10151e")
        for j, (pixels, index, label) in enumerate(shots):
            x, y = j % 3 * 320, j // 3 * 200
            sheet.paste(Image.fromarray(pixels).resize((320, 176)), (x, y + 24))
            ImageDraw.Draw(sheet).text(
                (x + 5, y + 5), f"{index} / {label}", fill="white"
            )
        sheet.save(out / "contact-sheet.jpg", quality=95)
        reports.append(dict(name=out.name, strength=strength, frames=320, spawn=0))
        engine.sessions.clear()
        print(json.dumps(reports[-1]), flush=True)
    (root / "complete.json").write_text(json.dumps(reports, indent=2))
    volume.commit()
    return reports


@app.local_entrypoint()
def weapon():
    print(json.dumps(weapon_check.remote()), flush=True)
