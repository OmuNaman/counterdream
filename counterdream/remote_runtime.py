"""Private GPU session state with bounded lifetime and idempotent frame requests."""
import time
import numpy as np
import torch

from .actions import encode
from .model import load_model
from .serve import Control, png


class SessionEngine:
    def __init__(self, checkpoint, seeds):
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        self.model, self.checkpoint = load_model(checkpoint, "cuda")
        with np.load(seeds, allow_pickle=False) as data:
            self.seeds = torch.from_numpy(data["frames"].copy()).to("cuda").permute(0,1,4,2,3).float()/127.5-1
            self.actions = torch.from_numpy(data["actions"].copy()).to("cuda")
        self.sessions = {}
        self.generated = 0
        self.created = time.monotonic()

    @torch.inference_mode()
    def frame(self, session_id, command, sequence):
        if not isinstance(session_id,str) or len(session_id)!=32 or not all(x in "0123456789abcdef" for x in session_id):
            raise ValueError("Invalid session identifier")
        control = Control.model_validate(command)
        now = time.monotonic()
        self.sessions = {k:v for k,v in self.sessions.items() if now-v["seen"] < 900}
        prior = self.sessions.get(session_id)
        if prior is not None and sequence == prior["sequence"]:
            return prior["last_result"]
        if prior is not None and sequence < prior["sequence"]:
            return {"error":"Old control request ignored."}
        if control.type == "reset":
            if control.spawn >= len(self.seeds):
                raise ValueError("Unknown spawn")
            if prior is None and len(self.sessions)>=4:
                return {"error":"GPU session limit reached. Try again after an inactive session expires."}
            state = dict(context=self.seeds[control.spawn:control.spawn+1].clone(),
                         actions=self.actions[control.spawn:control.spawn+1].clone(),
                         frame=0, seen=now, sequence=sequence, spawned=now)
            frame = (state["context"][0,-1].add(1).mul(127.5).round().byte().permute(1,2,0).cpu().numpy())
            result = dict(png=png(frame), frame=0, gpu_ms=0.)
            state["last_result"] = result
            self.sessions[session_id] = state
            return result
        if prior is None:
            return {"error":"The idle GPU session expired. Reset the world to continue."}
        if now-prior["spawned"] > 900 or now-self.created > 2700 or self.generated>=20000:
            return {"error":"Cloud session allocation finished. Restart the viewer for another allocation."}
        if sequence != prior["sequence"]+1:
            return {"error":"Missing control request. Reset the world."}
        controls = torch.as_tensor(encode(control.keys,control.dx,control.dy,control.fire,control.scope),device="cuda")[None,None]
        history = torch.cat((prior["actions"],controls),1)
        tick = time.perf_counter()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            predicted = self.model.sample(prior["context"],history,steps=control.steps,
                                          seed=1000+prior["frame"])
        frame = predicted[0].float().add(1).mul(127.5).round().clamp(0,255).byte().permute(1,2,0).cpu().numpy()
        gpu_ms = (time.perf_counter()-tick)*1000
        prior["context"] = torch.cat((prior["context"][:,1:],predicted[:,None]),1)
        prior["actions"] = history[:,1:]
        prior["frame"] += 1
        prior["sequence"],prior["seen"] = sequence,now
        self.generated += 1
        result = dict(png=png(frame), frame=prior["frame"],gpu_ms=round(gpu_ms,1))
        prior["last_result"] = result
        return result
