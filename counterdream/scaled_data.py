"""Immutable, resumable full-corpus preparation and bounded-memory replay.

Raw game files are read in bounded ranges, never extracted as filesystem paths.
Each source file remains one temporal episode. Published DIAMOND holdout files
are excluded from training; an additional deterministic split is validation.
"""

import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time
import zipfile
from collections import OrderedDict

import h5py
import numpy as np
from PIL import Image
import requests

from .data import DATASET, REVISION, HTTPRangeReader


HEIGHT, WIDTH = 88, 160


def replay_weights(records):
    rare = np.array([1+min(4,(r["action_counts"][4]+r["action_counts"][10])/10)
                     for r in records],dtype=float)
    expert = np.array([bool(r.get("expert")) for r in records],dtype=float)
    if expert.sum():
        return .65/len(records)+.20*rare/rare.sum()+.15*expert/expert.sum()
    return .75/len(records)+.25*rare/rare.sum()


class RemoteZipReader(HTTPRangeReader):
    def __init__(self,url,size):
        super().__init__(url)
        self.size = size

    def seek(self,offset,whence=0):
        return super().seek(self.size+offset,0) if whence==2 else super().seek(offset,whence)

    def read(self,size=-1):
        if size<0:
            size = self.size-self.position
            if size>1_048_576:
                raise ValueError("Unbounded zip read outside end-of-file metadata")
        return super().read(min(size,self.size-self.position))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def catalog():
    response = requests.get(
        f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}",
        params={"recursive": "false", "limit": 1000}, timeout=60,
    )
    response.raise_for_status()
    entries = [x for x in response.json() if re.fullmatch(
        r"hdf5_dm_july2021_\d+_to_\d+\.tar", x["path"])]
    return sorted(entries, key=lambda x: int(x["path"].split("_")[3]))


def episode_split(name, test_names):
    name = Path(name).name
    if name in test_names:
        return "test"
    value = int(hashlib.sha256(("counterdream-v3:" + name).encode()).hexdigest()[:8], 16)
    return "val" if value % 20 == 0 else "train"


def prepare_shard(root, shard, test_names, limit=None, max_seconds=3300, commit=None):
    if not re.fullmatch(r"hdf5_dm_july2021_\d+_to_\d+\.tar", shard):
        raise ValueError("Unknown archive name")
    root = Path(root) / shard.removesuffix(".tar")
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    spec = dict(dataset=DATASET, revision=REVISION, shard=shard,
                height=HEIGHT, width=WIDTH, format="npy-rgb-uint8",
                test_split_sha256=hashlib.sha256("\n".join(sorted(test_names)).encode()).hexdigest())
    records = []
    if manifest.exists():
        prior = json.loads(manifest.read_text())
        if any(prior.get(k) != v for k, v in spec.items()):
            raise ValueError("Existing shard has different provenance or dimensions")
        records = prior["episodes"]
        if prior["complete"] or limit is not None and len(records) >= limit:
            return prior
    done = {x["source"] for x in records}
    started = time.monotonic()
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{shard}"
    complete = True
    with HTTPRangeReader(url) as source, tarfile.open(fileobj=source, mode="r:") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".hdf5") or member.name in done:
                continue
            if time.monotonic() - started > max_seconds or limit is not None and len(records) >= limit:
                complete = False
                break
            if not 0 < member.size <= 250_000_000:
                raise ValueError("Unexpected source member size")
            payload = archive.extractfile(member).read()
            name = Path(member.name).name
            if not re.fullmatch(r"hdf5_dm_july2021_\d+\.hdf5", name):
                raise ValueError("Unexpected episode name")
            with h5py.File(io.BytesIO(payload), "r") as ep:
                ids = sorted(int(k.split("_")[1]) for k in ep if k.endswith("_x"))
                if ids != list(range(1000)):
                    raise ValueError("Expected 1000 contiguous observations")
                frames = np.empty((1000, 3, HEIGHT, WIDTH), np.uint8)
                actions = np.empty((1000, 51), np.float32)
                for i in ids:
                    rgb = ep[f"frame_{i}_x"][:][..., ::-1]
                    frames[i] = np.asarray(Image.fromarray(rgb).resize(
                        (WIDTH, HEIGHT), Image.Resampling.BOX)).transpose(2, 0, 1)
                    actions[i] = ep[f"frame_{i}_y"][:]
            if not np.isfinite(actions).all():
                raise ValueError("Nonfinite action labels")
            split = episode_split(name, test_names)
            folder = root / split
            folder.mkdir(exist_ok=True)
            stem = Path(name).stem
            frame_path = folder / (stem + ".frames.npy")
            action_path = folder / (stem + ".actions.npy")
            for path, array in ((frame_path, frames), (action_path, actions)):
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, array, allow_pickle=False)
                temporary.replace(path)
            records.append(dict(source=member.name, sha256=hashlib.sha256(payload).hexdigest(),
                                file=frame_path.relative_to(root).as_posix(),
                                actions=action_path.relative_to(root).as_posix(), frames=len(frames),
                                split=split, action_counts=actions[:, :13].sum(0).astype(int).tolist()))
            write_json(manifest, dict(**spec, complete=False, episodes=records))
            if len(records) % 20 == 0:
                print(json.dumps(dict(shard=shard, episodes=len(records),
                                      seconds=round(time.monotonic()-started))), flush=True)
                if commit:
                    commit()
    result = dict(**spec, complete=complete, episodes=records)
    write_json(manifest, result)
    if commit:
        commit()
    return result


def build_index(root, allow_partial=False):
    root = Path(root)
    records, manifests = [], []
    for path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        if not manifest["complete"] and not allow_partial:
            raise ValueError(f"Shard is incomplete: {path.parent.name}")
        manifests.append(dict(shard=manifest["shard"], complete=manifest["complete"],
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for item in manifest["episodes"]:
            record = dict(item)
            record["file"] = (path.parent / item["file"]).relative_to(root).as_posix()
            record["actions"] = (path.parent / item["actions"]).relative_to(root).as_posix()
            records.append(record)
    unique = {}
    duplicates = []
    for record in records:
        key = record["source"]
        if key in unique:
            if unique[key]["sha256"] != record["sha256"] or unique[key]["split"] != record["split"]:
                raise ValueError("Conflicting duplicate source episodes")
            duplicates.append(key)
        else:
            unique[key] = record
    records = list(unique.values())
    report = dict(dataset=DATASET, revision=REVISION, height=HEIGHT, width=WIDTH,
                  partial=any(not x["complete"] for x in manifests), shards=manifests, identical_duplicates_skipped=duplicates,
                  episodes=records, counts={s: sum(x["frames"] for x in records if x["split"]==s)
                                           for s in ("train", "val", "test")})
    write_json(root / "index.json", report)
    return report


class DiskReplay:
    """Random temporal windows using a bounded LRU of memory-mapped episodes."""

    def __init__(self, root, split="train", context=8, cache_size=32):
        self.root, self.context, self.cache_size = Path(root), context, cache_size
        self.index = json.loads((self.root / "index.json").read_text())
        self.records = [x for x in self.index["episodes"] if x["split"] == split]
        if not self.records:
            raise ValueError(f"No {split} records")
        self.cache = OrderedDict()
        # 75% uniform episodes, 25% moderately favor rare recorded controls.
        self.weights = replay_weights(self.records)

    def episode(self, index):
        if index not in self.cache:
            record = self.records[index]
            self.cache[index] = (np.load(self.root / record["file"], mmap_mode="r", allow_pickle=False),
                                 np.load(self.root / record["actions"], mmap_mode="r", allow_pickle=False))
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def batch_numpy(self, batch_size, rng, horizon=1, balanced=True):
        episode_ids = rng.choice(len(self.records), batch_size, p=self.weights if balanced else None)
        observations, actions = [], []
        for i in episode_ids:
            frames, acts = self.episode(int(i))
            length = self.context + horizon
            start = int(rng.integers(0, len(frames) - length + 1))
            observations.append(frames[start:start+length])
            actions.append(acts[start:start+length-1])
        return np.stack(observations), np.stack(actions)

    def batch(self, batch_size, rng, horizon=1):
        import torch
        obs, acts = self.batch_numpy(batch_size, rng, horizon, balanced=False)
        return (torch.from_numpy(obs).to("cuda").float() / 127.5 - 1,
                torch.from_numpy(acts).to("cuda"))


class GPUShardReplay(DiskReplay):
    """Each rank holds a disjoint uint8 shard, using the five GPUs' aggregate RAM.

    This avoids hundreds of millions of remote random reads during training.
    It does not replicate the whole corpus onto every accelerator.
    """

    def __init__(self, root, rank, world_size, device, context=8):
        import torch
        super().__init__(root, context=context)
        self.records = self.records[rank::world_size]
        self.lengths = np.array([r["frames"] for r in self.records], dtype=np.int64)
        self.offsets = np.concatenate(([0],np.cumsum(self.lengths)[:-1]))
        self.weights = replay_weights(self.records)
        h,w = self.index["height"],self.index["width"]
        total = int(self.lengths.sum())
        required = total*(3*h*w+51*4)
        free,_ = torch.cuda.mem_get_info(device)
        if required + 32_000_000_000 > free:
            raise MemoryError("GPU shard would leave less than 32 GB for training activations")
        started = time.monotonic()
        self.frames_gpu = torch.empty((total,3,h,w),dtype=torch.uint8,device=device)
        self.actions_gpu = torch.empty((total,51),dtype=torch.float32,device=device)
        self.device = device
        for i,record in enumerate(self.records):
            offset = int(self.offsets[i])
            count = record["frames"]
            self.frames_gpu[offset:offset+count].copy_(torch.from_numpy(np.load(self.root/record["file"],allow_pickle=False)))
            self.actions_gpu[offset:offset+count].copy_(torch.from_numpy(np.load(self.root/record["actions"],allow_pickle=False)))
            if (i+1)%100==0 or i+1==len(self.records):
                print(json.dumps(dict(loading_rank=rank,episodes=i+1,total_episodes=len(self.records),
                                      gpu_data_gb=required/1e9,seconds=time.monotonic()-started)),flush=True)
        self.cache.clear()

    def batch_device(self,batch_size,rng,horizon=1):
        import torch
        ep = rng.choice(len(self.records),batch_size,p=self.weights)
        length = self.context+horizon
        starts = (rng.random(batch_size)*(self.lengths[ep]-length+1)).astype(np.int64)+self.offsets[ep]
        ids = torch.as_tensor(starts,device=self.device)[:,None]+torch.arange(length,device=self.device)[None]
        return self.frames_gpu[ids].float()/127.5-1,self.actions_gpu[ids[:,:-1]]


def prepare_expert(root,commit=None,max_seconds=3300):
    shard = "dataset_dm_expert_dust2.zip"
    size = 24274109780
    root = Path(root)/"expert-dust2"
    root.mkdir(parents=True,exist_ok=True)
    manifest = root/"manifest.json"
    spec = dict(dataset=DATASET,revision=REVISION,shard=shard,height=HEIGHT,width=WIDTH,
                format="npy-rgb-uint8",expert=True,
                source_archive_sha256="49bc679d4a7a6c0a80fb35f6c3b09a9dd6161bac87ac5aa2732aec41dcabd19f")
    records = []
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if any(old.get(k)!=v for k,v in spec.items()):
            raise ValueError("Expert preparation provenance changed")
        if old["complete"]:
            return old
        records = old["episodes"]
    done = {x["source"] for x in records}
    started = time.monotonic()
    complete = True
    with RemoteZipReader(f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{shard}",size) as source:
        with zipfile.ZipFile(source) as archive:
            members = [x for x in archive.infolist() if x.filename.endswith(".hdf5")]
            if len(members)!=190:
                raise ValueError("Expected 190 expert source files")
            for member in members:
                if member.filename in done:
                    continue
                if time.monotonic()-started > max_seconds:
                    complete = False
                    break
                if not 0 < member.file_size <= 250_000_000:
                    raise ValueError("Unexpected expert source file size")
                payload = archive.read(member)
                with h5py.File(io.BytesIO(payload),"r") as ep:
                    ids = sorted(int(k.split("_")[1]) for k in ep if k.endswith("_x"))
                    if ids != list(range(1000)):
                        raise ValueError("Expected 1000 contiguous expert frames")
                    frames = np.empty((1000,3,HEIGHT,WIDTH),np.uint8)
                    actions = np.empty((1000,51),np.float32)
                    for i in ids:
                        rgb = ep[f"frame_{i}_x"][:][...,::-1]
                        frames[i] = np.asarray(Image.fromarray(rgb).resize((WIDTH,HEIGHT),Image.Resampling.BOX)).transpose(2,0,1)
                        actions[i] = ep[f"frame_{i}_y"][:]
                if not np.isfinite(actions).all():
                    raise ValueError("Invalid expert action labels")
                name = Path(member.filename).name
                split = episode_split(name,set())
                folder = root/split
                folder.mkdir(exist_ok=True)
                fp = folder/(Path(name).stem+".frames.npy")
                ap = folder/(Path(name).stem+".actions.npy")
                for path,array in ((fp,frames),(ap,actions)):
                    temporary = path.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.save(stream,array,allow_pickle=False)
                    temporary.replace(path)
                records.append(dict(source=member.filename,file=fp.relative_to(root).as_posix(),
                                    actions=ap.relative_to(root).as_posix(),expert=True,
                                    frames=1000,split=split,sha256=hashlib.sha256(payload).hexdigest(),
                                    action_counts=actions[:,:13].sum(0).astype(int).tolist()))
                write_json(manifest,dict(**spec,complete=False,episodes=records))
                if len(records)%20==0:
                    print(json.dumps(dict(expert_episodes=len(records),seconds=time.monotonic()-started)),flush=True)
                    if commit:
                        commit()
    result = dict(**spec,complete=complete,episodes=records)
    write_json(manifest,result)
    if commit:
        commit()
    return result
