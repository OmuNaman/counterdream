"""Create an immutable, checksummed bundle for a bounded demo viewer."""

import argparse
import json
from pathlib import Path
import shutil

import torch

from .demo_record import sha256


def prepare(checkpoint, seeds, profile, output, quality_note):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = json.loads(Path(profile).read_text())
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    shutil.copyfile(checkpoint, output / "model.pt")
    shutil.copyfile(seeds, output / "seeds.npz")
    for field, name in [
        ("upscaler", "upscaler.pth"),
        ("motion_checkpoint", "motion.pt"),
    ]:
        if settings.get(field):
            source = Path(settings[field])
            if not source.is_file():
                source = Path(profile).parent / source
            shutil.copyfile(source, output / name)
            settings[field] = name
    (output / "profile.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    cfg = ckpt["config"]
    enlarged = bool(settings.get("upscaler"))
    report = dict(
        checkpoint_step=ckpt["step"],
        config=cfg,
        sampling_steps=settings["sampling"]["steps"],
        display_resolution=[
            cfg["width"] * (4 if enlarged else 1),
            cfg["height"] * (4 if enlarged else 1),
        ],
        display_note=(
            "4× Real-ESRGAN display enhancement; it cannot repair simulation drift."
            if enlarged
            else "Bicubic display enlargement."
        ),
        quality_note=quality_note,
        world_model_pretrained=False,
        display_upscaler_pretrained=enlarged,
        files={p.name: sha256(p) for p in output.iterdir() if p.is_file()},
    )
    (output / "bundle.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "seeds", "profile", "output", "quality-note"):
        parser.add_argument("--" + name, required=True)
    prepare(**vars(parser.parse_args()))
