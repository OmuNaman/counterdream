"""Bounded transfer of our prepared corpus between authorized Modal workspaces.

Only CPU transfer workers receive the source credentials as an unnamed Modal
Secret. They only read source files; keys are never function arguments or logs.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time

import modal

from counterdream.cloud_config import DATA, VOLUME_NAME, volume

app = modal.App("counterdream-corpus-transfer")
INDEX_SHA256 = "7045b9e86beec3851cbd6b214147cc0fd694c8844d90325d8195ef2323996915"
PARTS = 12
PROBE = "/artifacts/data-v3-transfer-probe"
REPORTS = "/artifacts/migrations/dust2-v3"
source_keys = ("COUNTERDREAM_SOURCE_TOKEN_ID", "COUNTERDREAM_SOURCE_TOKEN_SECRET")
source_secret = modal.Secret.from_dict({key: os.environ.get(key) for key in source_keys})
cpu_image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("numpy==1.26.4", "h5py==3.12.1", "Pillow==11.1.0", "requests==2.32.3")
             .add_local_python_source("counterdream"))


def source_volume():
    client = modal.Client.from_credentials(*(os.environ[key] for key in source_keys))
    source = modal.Volume.from_name(VOLUME_NAME, environment_name="main", client=client)
    source.hydrate()
    volume.hydrate()
    if source.object_id == volume.object_id:
        raise ValueError("Source and destination must be different volumes")
    return source


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or not path.parts:
        raise ValueError("Invalid corpus-relative path")
    return path.as_posix()


def load_source_index(source):
    payload = b"".join(source.read_file("data-v3/index.json"))
    if hashlib.sha256(payload).hexdigest() != INDEX_SHA256:
        raise ValueError("Source index differs from the verified corpus")
    index = json.loads(payload)
    if index["partial"] or len(index["episodes"]) != 5688:
        raise ValueError("Expected the complete prepared corpus")
    return payload, index


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def copy_array(source, relative, root, kind):
    import numpy as np
    relative = safe_relative(relative)
    target = Path(root) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".transfer-tmp")
    digest = hashlib.sha256()
    size = 0
    with temporary.open("wb") as stream:
        for block in source.read_file("data-v3/" + relative):
            size += len(block)
            if size > 43_000_000:
                raise ValueError("Unexpected prepared-array size")
            digest.update(block)
            stream.write(block)
    array = np.load(temporary, mmap_mode="r", allow_pickle=False)
    expected = ((1000, 3, 88, 160), np.dtype("uint8")) if kind == "frames" else ((1000, 51), np.dtype("float32"))
    if (array.shape, array.dtype) != expected or array.offset + array.nbytes != size:
        raise ValueError("Transferred array shape, dtype, or length mismatch")
    if kind == "actions" and not np.isfinite(array).all():
        raise ValueError("Nonfinite transferred actions")
    del array
    temporary.replace(target)
    return dict(file=relative, bytes=size, sha256=digest.hexdigest())


def copy_episode(source, record, root):
    return dict(source=record["source"], source_hdf5_sha256=record["sha256"],
                files=[copy_array(source, record["file"], root, "frames"),
                       copy_array(source, record["actions"], root, "actions")])


@app.function(image=cpu_image, cpu=2, memory=4096, timeout=600, retries=0,
              max_containers=1, scaledown_window=2, secrets=[source_secret],
              volumes={"/artifacts": volume})
def copy_probe():
    source = source_volume()
    _, index = load_source_index(source)
    selected = []
    for split, count in (("train", 2), ("val", 1), ("test", 1)):
        selected.extend([r for r in index["episodes"] if r["split"] == split and not r.get("expert")][:count])
    report = [copy_episode(source, record, PROBE) for record in selected]
    subset = dict(index, episodes=selected, partial=True,
                  counts={s: sum(r["frames"] for r in selected if r["split"] == s)
                          for s in ("train", "val", "test")})
    atomic_json(Path(PROBE, "index.json"), subset)
    result = dict(complete=True, index_sha256=INDEX_SHA256, episodes=report, counts=subset["counts"])
    atomic_json(Path(REPORTS, "probe.json"), result)
    volume.commit()
    return dict(complete=True, counts=subset["counts"], bytes=sum(f["bytes"] for r in report for f in r["files"]))


@app.function(image=cpu_image, cpu=2, memory=4096, timeout=3600, retries=0,
              max_containers=PARTS, scaledown_window=2, single_use_containers=True,
              secrets=[source_secret], volumes={"/artifacts": volume})
def copy_part(part: int):
    if not 0 <= part < PARTS:
        raise ValueError("Invalid transfer partition")
    source = source_volume()
    _, index = load_source_index(source)
    selected = index["episodes"][part::PARTS]
    path = Path(REPORTS, f"part-{part}.json")
    state = dict(part=part, parts=PARTS, index_sha256=INDEX_SHA256, complete=False, episodes=[])
    if path.exists():
        state = json.loads(path.read_text())
        if state["index_sha256"] != INDEX_SHA256 or state["parts"] != PARTS:
            raise ValueError("Transfer provenance changed")
        if state["complete"]:
            return dict(part=part, episodes=len(state["episodes"]), complete=True)
    done = {row["source"] for row in state["episodes"]}
    started = time.monotonic()
    print(json.dumps(dict(starting_part=part, existing=len(done), target=len(selected))), flush=True)
    for record in selected:
        if record["source"] in done:
            continue
        if time.monotonic() - started > 3300:
            break
        state["episodes"].append(copy_episode(source, record, DATA))
        atomic_json(path, state)
        if len(state["episodes"]) % 20 == 0:
            volume.commit()
            print(json.dumps(dict(part=part, episodes=len(state["episodes"]), target=len(selected),
                                  seconds=round(time.monotonic()-started))), flush=True)
    state["complete"] = len(state["episodes"]) == len(selected)
    atomic_json(path, state)
    volume.commit()
    return dict(part=part, episodes=len(state["episodes"]), complete=state["complete"])


@app.function(image=cpu_image, cpu=1, memory=2048, timeout=600, retries=0,
              secrets=[source_secret], volumes={"/artifacts": volume})
def finish_transfer():
    from counterdream.scaled_data import build_index
    source = source_volume()
    payload, index = load_source_index(source)
    expected = {r["source"]: r for r in index["episodes"]}
    seen = set()
    total_bytes = 0
    for part in range(PARTS):
        report = json.loads(Path(REPORTS, f"part-{part}.json").read_text())
        if not report["complete"] or report["index_sha256"] != INDEX_SHA256:
            raise ValueError("Transfer partition is incomplete")
        for record in report["episodes"]:
            name = record["source"]
            if name in seen or record["source_hdf5_sha256"] != expected[name]["sha256"]:
                raise ValueError("Duplicate or conflicting transferred recording")
            if {f["file"] for f in record["files"]} != {expected[name]["file"], expected[name]["actions"]}:
                raise ValueError("Transferred file names differ from source index")
            for file in record["files"]:
                if Path(DATA, safe_relative(file["file"])).stat().st_size != file["bytes"]:
                    raise ValueError("Transferred file is missing or incomplete")
                total_bytes += file["bytes"]
            seen.add(name)
    if seen != set(expected):
        raise ValueError("Transferred corpus coverage differs from source")
    for shard in index["shards"]:
        folder = (f'expert-dust2-part-{shard["part"]}' if shard["shard"].endswith(".zip") else
                  shard["shard"].removesuffix(".tar") + (f'-part-{shard["part"]}' if shard["parts"] > 1 else ""))
        manifest = b"".join(source.read_file(f"data-v3/{folder}/manifest.json"))
        if hashlib.sha256(manifest).hexdigest() != shard["sha256"]:
            raise ValueError("Source manifest checksum changed")
        target = Path(DATA, folder, "manifest.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(manifest)
    build_index(DATA)
    if hashlib.sha256(Path(DATA, "index.json").read_bytes()).hexdigest() != INDEX_SHA256:
        raise ValueError("Rebuilt destination index differs from source")
    result = dict(complete=True, index_sha256=INDEX_SHA256, episodes=len(seen),
                  frames=sum(index["counts"].values()), counts=index["counts"], bytes=total_bytes,
                  source_modified=False, training_weights_transferred=False)
    atomic_json(Path(REPORTS, "complete.json"), result)
    volume.commit()
    return result


@app.local_entrypoint()
def probe():
    print(json.dumps(copy_probe.remote()), flush=True)


@app.local_entrypoint()
def transfer():
    results = list(copy_part.map(range(PARTS), return_exceptions=True))
    print(json.dumps([r if not isinstance(r, Exception) else dict(part=i, error=type(r).__name__)
                      for i, r in enumerate(results)]), flush=True)
    if any(isinstance(r, Exception) or not r["complete"] for r in results):
        raise RuntimeError("Incomplete transfer; completed files are preserved")
    print(json.dumps(finish_transfer.remote()), flush=True)


@app.local_entrypoint()
def finish():
    print(json.dumps(finish_transfer.remote()), flush=True)
