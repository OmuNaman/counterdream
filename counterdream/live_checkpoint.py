"""Pin a training snapshot and validation starting views without evaluating test data."""
import hashlib
import json
from pathlib import Path
import re


def snapshot_folder(run, snapshot):
    if not re.fullmatch(r"[0-9a-f]{32}", snapshot):
        raise ValueError("Invalid live snapshot identifier")
    return Path(run, "live", snapshot)


def sha256(path):
    with Path(path).open("rb") as stream:
        return stream_sha256(stream)


def stream_sha256(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(4*1024*1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def snapshot_latest(run, data, snapshot):
    import numpy as np
    import torch
    from .scaled_data import DiskReplay

    out = snapshot_folder(run, snapshot)
    out.mkdir(parents=True, exist_ok=False)
    # One open file remains the same checkpoint even when the trainer atomically
    # replaces latest.pt. Hash that same handle, never reopen the mutable path.
    with Path(run, "latest.pt").open("rb") as stream:
        checkpoint_hash = stream_sha256(stream)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    export = {key: checkpoint[key] for key in ("config", "ema", "step", "run")}
    del checkpoint
    if export["step"] <= 0:
        raise ValueError("No trained checkpoint is available yet")
    cfg = export["config"]
    replay = DiskReplay(data, "val", context=cfg["context"])
    frames, actions, names = [], [], []
    for clip, position in enumerate(np.linspace(0, len(replay.records)-1, 6, dtype=int)):
        obs, acts = replay.episode(int(position))
        start = 120 + clip * 71
        frames.append(obs[start:start+cfg["context"]].transpose(0, 2, 3, 1).copy())
        actions.append(acts[start:start+cfg["context"]-1].copy())
        names.append(f"Val {position+1} / frame {start}")
    seeds = np.stack(frames)
    history = np.stack(actions)
    if seeds.shape != (6, cfg["context"], cfg["height"], cfg["width"], 3):
        raise ValueError("Validation starting frames do not match the model")
    if history.shape != (6, cfg["context"]-1, 51) or not np.isfinite(history).all():
        raise ValueError("Invalid validation action history")
    torch.save(export, out / "model.pt")
    np.savez_compressed(out / "seeds.npz", frames=seeds, actions=history, names=np.array(names))
    report = dict(checkpoint_step=export["step"], checkpoint_sha256=checkpoint_hash,
                  export_sha256=sha256(out / "model.pt"), seeds_sha256=sha256(out / "seeds.npz"),
                  config=cfg, spawns=names, snapshot=snapshot, split="val",
                  pretrained_weights=False, purpose="interactive training preview")
    (out / "preview.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
