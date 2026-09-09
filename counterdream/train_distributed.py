"""Single-node DDP training with disk replay, causal unrolling, and resumable RNGs."""
import argparse
from collections import deque
import copy
from dataclasses import asdict
from datetime import timedelta
import json
import hashlib
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .model import ModelConfig, WorldModel
from .scaled_data import DiskReplay, GPUShardReplay, HEIGHT, WIDTH, write_json
from .train import validate


class SequenceObjective(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs, actions, sampled_context=False):
        count = self.model.cfg.context
        context = obs[:, :count]
        losses = []
        for t in range(obs.shape[1] - count):
            controls = actions[:, t:t+count]
            loss, predicted = self.model.loss(context, controls, obs[:, t+count])
            losses.append(loss)
            # Train against known futures after seeing our own imperfect output.
            # Occasionally use a full sampled frame, not just a denoised target.
            if sampled_context and t == 0:
                with torch.no_grad():
                    predicted = self.model.sample(context, controls, steps=4)
            context = torch.cat((context[:, 1:], predicted.detach()[:, None]), 1)
        return torch.stack(losses).mean()


def distributed_setup():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    return rank, world, torch.device("cuda", local_rank)


def sync(world):
    if world > 1:
        dist.barrier()


@torch.no_grad()
def rollout_score(model, replay, device, frames=32, clips=16):
    rng = np.random.default_rng(76543)
    losses = {str(h):[] for h in (1,8,16,32) if h<=frames}
    for offset in range(0,clips,4):
        obs, acts = replay.batch_numpy(min(4,clips-offset), rng, horizon=frames, balanced=False)
        obs = torch.from_numpy(obs).to(device).float() / 127.5 - 1
        acts = torch.from_numpy(acts).to(device)
        context = obs[:, :model.cfg.context]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for t in range(frames):
                prediction = model.sample(context, acts[:, t:t+model.cfg.context], steps=4,
                                          seed=4000+offset*100+t)
                if str(t+1) in losses:
                    losses[str(t+1)].extend(((prediction-obs[:,t+model.cfg.context])/2).square().mean((1,2,3)).tolist())
                context = torch.cat((context[:, 1:], prediction[:, None]), 1)
    return {h:float(np.mean(values)) for h,values in losses.items()}


def run(args):
    started = time.monotonic()
    rank, world, device = distributed_setup()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    cfg = ModelConfig(height=HEIGHT, width=WIDTH, context=8, base=args.base, cond_dim=768, version=3)
    raw = WorldModel(cfg).to(device)
    model = SequenceObjective(raw)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
    ema = copy.deepcopy(raw).eval().requires_grad_(False) if rank == 0 else None
    optimizer = torch.optim.AdamW(raw.parameters(), lr=1e-4, betas=(.9, .95), weight_decay=.01, fused=True)
    torch.manual_seed(42 + rank * 100003)
    rng = np.random.default_rng(42 + rank * 100003)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    begin, best, previous_seconds = 0, float("inf"), 0.
    data_hash = hashlib.sha256(Path(args.data,"index.json").read_bytes()).hexdigest()
    dataset_changed = False
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        if ckpt["config"] != asdict(cfg) or ckpt["run"]["world_size"] != world:
            raise ValueError("Resume requires the same model and world size")
        raw.load_state_dict(ckpt["model"])
        if ema is not None:
            ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        begin, best, previous_seconds = ckpt["step"], ckpt["best_score"], ckpt["total_seconds"]
        dataset_changed = ckpt["run"].get("dataset_index_sha256") != data_hash
        if dataset_changed:
            best = float("inf")
        state = ckpt["rngs"][rank]
        rng.bit_generator.state = state["numpy"]
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"], device)
        del ckpt
    elif (root / "latest.pt").exists():
        raise ValueError("Refusing to overwrite an existing training run")
    replay = (GPUShardReplay(args.data, rank, world, device, context=cfg.context)
              if args.gpu_replay else DiskReplay(args.data, context=cfg.context))
    validation = DiskReplay(args.data, "val", context=cfg.context) if rank == 0 else None
    info = dict(config=asdict(cfg), world_size=world, seed=42, initialization="random",
                pretrained_weights=False, parameters=sum(p.numel() for p in raw.parameters()),
                batch_per_gpu=args.batch, global_batch=world*args.batch,
                dataset_counts=replay.index["counts"], planned_steps=args.steps,
                dataset_index_sha256=data_hash, validation_selection_reset=dataset_changed,
                replay_storage="disjoint uint8 GPU shards" if args.gpu_replay else "disk mmap",
                max_seconds=args.max_seconds, gpu=torch.cuda.get_device_name(device),
                torch_version=str(torch.__version__), resumed_from_step=begin,
                validation_sampler_steps=4, validation_rollout_clips=16,
                selection="mean rollout MSE at horizons 8,16,32 over 16 validation clips",
                source=json.loads(Path(args.source).read_text()) if args.source else {})
    if rank == 0:
        write_json(root / f"run-from-{begin}.json", info)
        print(json.dumps(info), flush=True)

    def save(step, metrics=None):
        nonlocal best
        digest = hashlib.sha256()
        for name,value in sorted(raw.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        state = dict(numpy=rng.bit_generator.state, torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state(device), model_sha256=digest.hexdigest())
        states = [None]*world
        if world > 1:
            dist.all_gather_object(states, state)
        else:
            states[0] = state
        if len({state["model_sha256"] for state in states}) != 1:
            raise RuntimeError("Distributed ranks disagree on model weights")
        if rank == 0:
            score = float("inf") if metrics is None else metrics["selection_score"]
            # Keep step zero as a reported baseline, not a release candidate.
            improved = step > 0 and score < best
            if step > 0:
                best = min(score, best)
            total_seconds = previous_seconds + time.monotonic() - started
            checkpoint = dict(config=asdict(cfg), model=raw.state_dict(), ema=ema.state_dict(),
                              optimizer=optimizer.state_dict(), step=step, run=info,
                              rngs=states, best_score=best, total_seconds=total_seconds)
            tmp = root / "latest.tmp"
            torch.save(checkpoint, tmp)
            tmp.replace(root / "latest.pt")
            if improved or step > 0 and step % 10000 == 0:
                lightweight = {k: checkpoint[k] for k in ("config", "ema", "step", "run")}
                if improved:
                    tmp = root / "best.tmp"
                    torch.save(lightweight, tmp)
                    tmp.replace(root / "best.pt")
                if step > 0 and step % 10000 == 0:
                    tmp = root / f"ema-step-{step}.tmp"
                    torch.save(lightweight, tmp)
                    tmp.replace(root / f"ema-step-{step}.pt")
            if args.commit_volume:
                import modal
                modal.Volume.from_name(args.commit_volume).commit()
        sync(world)

    def evaluate(step):
        metrics = None
        if rank == 0:
            metrics = validate(ema, validation, device, batches=4, batch_size=8, steps=4)
            metrics["rollout_mse"] = rollout_score(ema, validation, device)
            metrics["selection_score"] = float(np.mean([metrics["rollout_mse"][str(k)] for k in (8,16,32)]))
            metrics.update(step=step, invocation_seconds=time.monotonic()-started,
                           total_seconds=previous_seconds+time.monotonic()-started)
            write_json(root / "metrics.json", metrics)
            with (root / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics)+"\n")
            print("VALIDATION " + json.dumps(metrics), flush=True)
        sync(world)
        return metrics

    sync(world)
    if not args.resume:
        # Exercise the largest unroll before spending on a long allocation.
        probe_rng = np.random.default_rng(100 + rank)
        probe_obs, probe_actions = replay.batch_numpy(args.batch, probe_rng, horizon=4)
        probe_obs = torch.from_numpy(probe_obs).to(device).float()/127.5-1
        probe_actions = torch.from_numpy(probe_actions).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probe_loss = model(probe_obs, probe_actions, sampled_context=True)
        probe_loss.backward()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            print(json.dumps(dict(stress_check="passed", horizon=4,
                                  peak_gb=torch.cuda.max_memory_allocated(device)/1e9)), flush=True)
        del probe_obs, probe_actions, probe_loss
    if not args.resume or dataset_changed:
        metrics = evaluate(begin)
        save(begin, metrics)
    train_started = time.monotonic()
    last_save = train_started
    timings = deque(maxlen=100)
    smooth = None
    def batch(step):
        horizon = 1 if step <= 500 else min(4, 2 + (step-500)//1500)
        if args.gpu_replay:
            return replay.batch_device(args.batch,rng,horizon)
        obs,acts = replay.batch_numpy(args.batch,rng,horizon,True)
        return torch.from_numpy(obs).to(device).float()/127.5-1,torch.from_numpy(acts).to(device)
    stopped_by_time = False
    step = begin
    try:
        for step in range(begin+1, args.steps+1):
            tick = time.monotonic()
            obs,acts = batch(step)
            lr = 1e-4 * min(step/300, 1) * (.25 + .75*.5*(1+math.cos(math.pi*min(step/args.steps, 1))))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(obs, acts, sampled_context=step > 2000 and step % 8 == 0)
            finite = torch.isfinite(loss).int()
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise RuntimeError(f"Nonfinite loss at step {step}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.)
            optimizer.step()
            if ema is not None:
                decay = min(.9995, (1+step)/(10+step))
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()), list(raw.parameters()), 1-decay)
            detached = loss.detach()
            if world > 1:
                dist.all_reduce(detached, op=dist.ReduceOp.AVG)
            torch.cuda.synchronize(device)
            timings.append(time.monotonic()-tick)
            value = float(detached)
            smooth = value if smooth is None else .98*smooth+.02*value
            elapsed = time.monotonic()-started
            # All ranks agree on stopping/checkpoint conditions, preventing deadlock.
            flags = torch.tensor([elapsed >= args.max_seconds-150 or step == args.steps,
                                  time.monotonic()-last_save >= 600 or step % 2000 == 0],
                                 device=device, dtype=torch.int32)
            if world > 1:
                dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            stop, checkpoint_due = flags.tolist()
            if rank == 0 and (step % 50 == 0 or step == begin+1):
                progress = dict(step=step, loss=value, smoothed_loss=smooth, grad_norm=float(grad),
                                invocation_seconds=elapsed, total_seconds=previous_seconds+elapsed,
                                step_seconds_mean=float(np.mean(timings)),
                                global_windows_per_second=world*args.batch/float(np.mean(timings)),
                                gpu_peak_gb=torch.cuda.max_memory_allocated(device)/1e9,
                                world_size=world, horizon=obs.shape[1]-cfg.context)
                write_json(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if stop or checkpoint_due:
                metrics = evaluate(step)
                save(step, metrics)
                last_save = time.monotonic()
            if stop:
                stopped_by_time = step < args.steps
                break
    finally:
        pass
    if rank == 0:
        report = dict(step=step, invocation_seconds=time.monotonic()-started,
                      total_seconds=previous_seconds+time.monotonic()-started,
                      stopped_by_time=stopped_by_time, best_score=best, world_size=world,
                      steady_step_seconds=float(np.mean(timings)))
        write_json(root / "complete.json", report)
        if args.commit_volume:
            import modal
            modal.Volume.from_name(args.commit_volume).commit()
        print("COMPLETE "+json.dumps(report), flush=True)
    sync(world)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--base", type=int, default=160)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--resume")
    parser.add_argument("--source")
    parser.add_argument("--commit-volume")
    parser.add_argument("--gpu-replay",action="store_true")
    parsed = parser.parse_args()
    if not 200 <= parsed.max_seconds <= 21000 or not 1 <= parsed.steps <= 100000:
        parser.error("Bound runs to 200–21000 seconds and 1–100000 steps")
    run(parsed)
