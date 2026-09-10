"""Run the checksummed motion-guided bundle on a local CUDA GPU."""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import torch
import uvicorn

from .cloud_viewer import make_cloud_app
from .demo_record import sha256
from .remote_runtime import SessionEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    folder = parser.parse_args().bundle
    manifest = json.loads((folder / "bundle.json").read_text())
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha256(folder / name) != digest:
            raise ValueError("Demo bundle checksum mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("This demo needs a CUDA GPU")
    engine = SessionEngine(
        folder / "model.pt", folder / "seeds.npz", folder / "profile.json"
    )
    with np.load(folder / "seeds.npz", allow_pickle=False) as seeds:
        names = seeds["names"].tolist()
    cfg = manifest["config"]
    metadata = dict(
        spawns=names,
        resolution=[cfg["width"], cfg["height"]],
        context_frames=cfg["context"],
        checkpoint_step=manifest["checkpoint_step"],
        device=torch.cuda.get_device_name(),
        recommended_steps=manifest["sampling_steps"],
        display_note=manifest["display_note"],
        quality_note=manifest["quality_note"],
        session_note="Local GPU · pause to stop generation · no API key needed",
    )

    async def frame(session, control, sequence):
        return await asyncio.to_thread(engine.frame, session, control, sequence)

    app = make_cloud_app(
        metadata,
        frame,
        on_session_end=lambda session: engine.sessions.pop(session, None),
    )
    uvicorn.run(app, host="127.0.0.1", port=7860)


if __name__ == "__main__":
    main()
