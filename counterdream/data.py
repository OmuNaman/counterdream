"""Stream a bounded subset; split by episode before forming temporal windows."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

import h5py
import numpy as np
from PIL import Image
import requests

DATASET = "TeaPearce/CounterStrike_Deathmatch"
SHARD = "hdf5_dm_july2021_1_to_200.tar"
REVISION = "265c6e5ac7aa335f58a2f2e864aad176fecfedde"


class HTTPRangeReader(io.RawIOBase):
    """Seek a remote uncompressed tar without re-downloading skipped episodes."""

    def __init__(self, url):
        self.url = url
        self.position = 0
        self.session = requests.Session()
        self.cached_start = 0
        self.cached = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1):
            raise ValueError("Only absolute/relative seek is supported")
        self.position = offset if whence == 0 else self.position + offset
        if self.position < 0:
            raise ValueError("Negative offset")
        return self.position

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Unbounded remote reads are disabled")
        if size == 0:
            return b""
        offset = self.position - self.cached_start
        if 0 <= offset and offset + size <= len(self.cached):
            data = self.cached[offset : offset + size]
        else:
            end = self.position + max(size, 4096) - 1
            with self.session.get(
                self.url,
                headers={"Range": f"bytes={self.position}-{end}"},
                timeout=(30, 180),
                stream=True,
            ) as response:
                response.raise_for_status()
                expected = f"bytes {self.position}-"
                if response.status_code != 206 or not response.headers.get(
                    "Content-Range", ""
                ).startswith(expected):
                    raise RuntimeError(
                        "Dataset server did not honor the bounded byte-range request"
                    )
                chunks = []
                received = 0
                for chunk in response.iter_content(1024 * 1024):
                    received += len(chunk)
                    if received > end - self.position + 1:
                        raise RuntimeError(
                            "Dataset server exceeded the requested byte range"
                        )
                    chunks.append(chunk)
                payload = b"".join(chunks)
            if size < 4096:
                self.cached_start = self.position
                self.cached = payload
            data = payload[:size]
        self.position += len(data)
        return data

    def close(self):
        self.session.close()
        super().close()


def prepare(root, episodes=100, height=64, width=112, progress=None, revision=REVISION):
    if not 1 <= episodes <= 200:
        raise ValueError("Request 1–200 episodes from this archive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Pin the original experiment's immutable revision for reproducible downloads.
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{SHARD}"
    partial = root / "manifest.partial.json"
    records = []
    if partial.exists():
        prior = json.loads(partial.read_text())
        if (prior["revision"], prior["height"], prior["width"]) != (
            revision,
            height,
            width,
        ):
            raise ValueError(
                "Existing partial dataset uses a different revision or resolution"
            )
        records = prior["episodes"]
        for record in records:
            if not (root / record["file"]).is_file():
                raise ValueError("A committed episode is missing")
        if len(records) >= episodes:
            if len(records) != episodes:
                raise ValueError(
                    "Existing partial dataset has more episodes than requested"
                )
            (root / "manifest.json").write_text(json.dumps(prior, indent=2))
            return prior
    completed = {r["source"] for r in records}
    started = time.time()
    print(
        json.dumps({"resuming_episodes": len(records), "requested_episodes": episodes}),
        flush=True,
    )
    with HTTPRangeReader(url) as source:
        with tarfile.open(fileobj=source, mode="r:") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".hdf5"):
                    continue
                if member.name in completed:
                    continue
                # Read only an expected regular member into memory; never extract paths.
                if member.size > 250_000_000:
                    raise ValueError(f"Unexpected episode size: {member.size}")
                payload = archive.extractfile(member).read()
                digest = hashlib.sha256(payload).hexdigest()
                with h5py.File(io.BytesIO(payload), "r") as episode:
                    indices = sorted(
                        int(k.split("_")[1]) for k in episode if k.endswith("_x")
                    )
                    if indices != list(range(len(indices))):
                        raise ValueError("Non-contiguous source frames")
                    frames, actions = [], []
                    for i in indices:
                        # Dataset stores OpenCV BGR; convert exactly once.
                        rgb = episode[f"frame_{i}_x"][:][..., ::-1]
                        frames.append(
                            np.asarray(
                                Image.fromarray(rgb).resize(
                                    (width, height), Image.Resampling.BOX
                                )
                            )
                        )
                        actions.append(episode[f"frame_{i}_y"][:])
                frames = np.stack(frames).astype(np.uint8)
                actions = np.stack(actions).astype(np.float32)
                if actions.shape != (len(frames), 51) or not np.isfinite(actions).all():
                    raise ValueError(f"Unexpected action format {actions.shape}")
                index = len(records)
                # Every tenth complete episode is held out; windows never cross boundaries.
                split = "val" if index % 10 == 9 else "train"
                target = root / split / f"episode-{index:04d}.npz"
                target.parent.mkdir(exist_ok=True)
                np.savez_compressed(target, frames=frames, actions=actions)
                record = dict(
                    file=str(target.relative_to(root)),
                    source=member.name,
                    sha256=digest,
                    frames=len(frames),
                    split=split,
                )
                records.append(record)
                state = dict(
                    dataset=DATASET,
                    revision=revision,
                    shard=SHARD,
                    height=height,
                    width=width,
                    episodes=records,
                    seconds=time.time() - started,
                )
                (root / "manifest.partial.json").write_text(json.dumps(state, indent=2))
                print(
                    json.dumps(
                        {
                            "prepared_episodes": len(records),
                            "frames": sum(x["frames"] for x in records),
                            "seconds": round(time.time() - started),
                        }
                    ),
                    flush=True,
                )
                if progress:
                    progress()
                if len(records) >= episodes:
                    break
    if len(records) != episodes:
        raise RuntimeError(f"Requested {episodes} episodes; obtained {len(records)}")
    (root / "manifest.json").write_text(json.dumps(state, indent=2))
    return state


class Replay:
    def __init__(self, root, split, context=4):
        import torch

        self.context = context
        self.frames, self.actions, self.names = [], [], []
        for path in sorted((Path(root) / split).glob("*.npz")):
            with np.load(path, allow_pickle=False) as record:
                self.frames.append(
                    torch.from_numpy(record["frames"].copy()).permute(0, 3, 1, 2)
                )
                self.actions.append(torch.from_numpy(record["actions"].copy()))
            self.names.append(path.name)
        if not self.frames:
            raise ValueError(f"No {split} episodes in {root}")
        self.lengths = np.array([len(x) for x in self.frames])

    def batch(self, batch_size, device, rng, horizon=1):
        import torch

        frames, actions = [], []
        for _ in range(batch_size):
            ep = int(rng.integers(len(self.frames)))
            start = int(
                rng.integers(0, int(self.lengths[ep]) - self.context - horizon + 1)
            )
            frames.append(self.frames[ep][start : start + self.context + horizon])
            actions.append(self.actions[ep][start : start + self.context + horizon - 1])
        return (
            torch.stack(frames)
            .to(device, non_blocking=True)
            .float()
            .div_(127.5)
            .sub_(1),
            torch.stack(actions).to(device, non_blocking=True),
        )
