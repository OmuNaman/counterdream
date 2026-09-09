"""Full-corpus evaluation with actual autoregressive videos and explicit provenance."""
import hashlib
import json
import math
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

from .actions import encode
from .evaluate import panel, uint8
from .model import load_model
from .scaled_data import DiskReplay, write_json
from .train import validate


@torch.inference_mode()
def evaluate(checkpoint, data, output, split="val", steps=4, clips=6, horizon=128):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    model, checkpoint_data = load_model(checkpoint, "cuda")
    context_length = model.cfg.context
    replay = DiskReplay(data, split, context=context_length)
    report = validate(model, replay, "cuda", batches=16, batch_size=16, steps=steps)
    report.update(split=split, checkpoint_step=checkpoint_data["step"],
                  checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  config=checkpoint_data["config"], pretrained_weights=False,
                  initialization="random", gpu=torch.cuda.get_device_name(),
                  dataset_counts=replay.index["counts"],
                  evaluation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    seeds, seed_actions, names, durations = [], [], [], []
    errors = {h: [] for h in (1,4,8,16,32,64,128) if h <= horizon}
    repeated = {h: [] for h in errors}
    # Spread samples over source files; fixed positions and seeds across checkpoints.
    positions = np.linspace(0, len(replay.records)-1, clips, dtype=int)
    for clip, pos in enumerate(positions):
        frames, acts = replay.episode(int(pos))
        start = 120 + clip * 71
        obs = torch.from_numpy(frames[start:start+context_length+horizon].copy()).to("cuda").float()/127.5-1
        actions = torch.from_numpy(acts[start:start+context_length+horizon-1].copy()).to("cuda")
        context = obs[:context_length][None]
        seeds.append(frames[start:start+context_length].transpose(0,2,3,1).copy())
        seed_actions.append(acts[start:start+context_length-1].copy())
        names.append(f"{split.title()} {pos+1} / frame {start}")
        contact = []
        with imageio.get_writer(out / f"rollout-{clip+1}.mp4", fps=16,
                                codec="libx264", quality=8, macro_block_size=2) as writer:
            for t in range(horizon):
                tick = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model.sample(context, actions[t:t+context_length][None],
                                              steps=steps, seed=10000+clip*1000+t)
                torch.cuda.synchronize()
                durations.append(time.perf_counter()-tick)
                gt = obs[t+context_length]
                row = panel([uint8(gt), uint8(prediction[0]), uint8(obs[context_length-1])],
                            ["RECORDED GAMEPLAY", "GENERATED: OWN FRAMES FED BACK", "REPEATED SEED"], scale=3)
                writer.append_data(row)
                if t+1 in errors:
                    errors[t+1].append(float(((prediction[0]-gt)/2).square().mean()))
                    repeated[t+1].append(float(((obs[context_length-1]-gt)/2).square().mean()))
                    contact.append(Image.fromarray(row))
                context = torch.cat((context[:, 1:], prediction[:, None]), 1)
        sheet = Image.new("RGB", (contact[0].width, sum(row.height for row in contact)))
        y = 0
        for row in contact:
            sheet.paste(row, (0,y))
            y += row.height
        sheet.save(out / f"rollout-{clip+1}.png")
    np.savez_compressed(out / "seeds.npz", frames=np.stack(seeds), actions=np.stack(seed_actions), names=np.array(names))
    report["rollout"] = {str(h): dict(mse=float(np.mean(errors[h])),
                                     repeat_mse=float(np.mean(repeated[h])), clips=clips)
                         for h in errors}
    report["inference_ms_median"] = float(np.median(durations)*1000)
    report["inference_ms_p95"] = float(np.percentile(durations,95)*1000)
    # Scripted controls provide inspectable examples, not a claim of correct physics.
    branches = [("LEFT", encode(dx=-30)), ("RIGHT", encode(dx=30)),
                ("FORWARD", encode(keys=["w"])), ("JUMP", encode(keys=["space"])),
                ("IDLE", encode())]
    contexts = torch.from_numpy(np.repeat(np.array(seeds[:1]),len(branches),axis=0)).to("cuda").permute(0,1,4,2,3).float()/127.5-1
    histories = torch.from_numpy(np.repeat(np.array(seed_actions[:1]),len(branches),axis=0)).to("cuda")
    with imageio.get_writer(out / "controls.mp4", fps=16, codec="libx264", quality=8, macro_block_size=2) as writer:
        for t in range(64):
            current = np.stack([a if name!="JUMP" or t%16==0 else encode() for name,a in branches])
            history = torch.cat((histories,torch.from_numpy(current).to("cuda")[:,None]),1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted = model.sample(contexts, history, steps=steps, seed=9900+t)
            row = panel([uint8(x) for x in predicted],[x[0] for x in branches],scale=2)
            writer.append_data(row)
            if t==15:
                Image.fromarray(row).save(out / "controls.png")
            contexts = torch.cat((contexts[:,1:],predicted[:,None]),1)
            histories = history[:,1:]
    torch.save({k: checkpoint_data[k] for k in ("config","ema","step","run")},out / "model.pt")
    report["export_sha256"] = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
    write_json(out / "evaluation.json",report)
    print(json.dumps(report),flush=True)
    return report
