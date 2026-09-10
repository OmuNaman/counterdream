"""Private GPU session state with bounded lifetime and idempotent frame requests."""
import time
import io
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from PIL import Image

from .actions import encode
from .model import load_model
from .serve import Control, png


class SessionEngine:
    def __init__(self, checkpoint, seeds, profile=None):
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        self.model, self.checkpoint = load_model(checkpoint, "cuda")
        self.settings = json.loads(Path(profile).read_text()) if profile else {}
        for field in ('upscaler','motion_checkpoint'):
            if self.settings.get(field) and not Path(self.settings[field]).is_absolute():
                self.settings[field] = str(Path(profile).parent / self.settings[field])
        if self.settings.get('motion_only') and not self.settings.get('motion_checkpoint'):
            raise ValueError('Motion-only profile requires motion weights')
        if self.settings.get('temporal_detail',0):
            raise ValueError('Temporal-detail experiments are not enabled in the live viewer')
        self.options = self.motion = self.upscaler = None
        if self.settings:
            from .demo_sampling import SamplingOptions
            self.options = SamplingOptions(**self.settings['sampling'])
        if self.settings.get('motion_checkpoint'):
            from .motion_model import MotionPredictor
            self.motion = MotionPredictor().to('cuda').eval()
            self.motion.load_state_dict(torch.load(self.settings['motion_checkpoint'],map_location='cpu',weights_only=True)['model'])
        if self.settings.get('upscaler'):
            from .upscale import DisplayUpscaler
            self.upscaler = DisplayUpscaler.from_weights(self.settings['upscaler'])
        with np.load(seeds, allow_pickle=False) as data:
            self.seeds = torch.from_numpy(data["frames"].copy()).to("cuda").permute(0,1,4,2,3).float()/127.5-1
            self.actions = torch.from_numpy(data["actions"].copy()).to("cuda")
        self.sessions = {}
        self.generated = 0
        self.created = time.monotonic()

    def render(self, image):
        """Enhance only the transmitted display; model feedback stays native."""
        if self.upscaler is not None:
            with torch.autocast('cuda',dtype=torch.float16):
                pixels = self.upscaler(image.float().add(1).mul(.5))
            frame = pixels[0].float().mul(255).round().clamp(0,255).byte().permute(1,2,0).cpu().numpy()
            stream = io.BytesIO()
            Image.fromarray(frame).save(stream,format='JPEG',quality=90,subsampling=0)
            return stream.getvalue()
        frame = image[0].float().add(1).mul(127.5).round().clamp(0,255).byte().permute(1,2,0).cpu().numpy()
        return png(frame)

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
            result = dict(png=self.render(state['context'][:,-1]), frame=0, gpu_ms=0.)
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
            initial = None
            if self.motion is not None:
                from .motion_model import guided_motion
                initial=guided_motion(self.motion,prior['context'][:,-2:],history[:,-2:],
                                      scale=self.settings.get('motion_scale',1.),center=self.settings.get('motion_center',False))
            if self.settings.get('motion_only'):
                predicted = initial
            elif self.options:
                from .demo_sampling import sample
                predicted = sample(self.model,prior['context'],history,replace(self.options,steps=control.steps),
                                   seed=1000+prior['frame'],initial_image=initial)
            else:
                predicted = self.model.sample(prior["context"],history,steps=control.steps,
                                              seed=1000+prior["frame"])
        if control.fire and self.options and self.settings.get('weapon_refinement',0):
            from .weapon_refine import refine_firing
            with torch.autocast('cuda',dtype=torch.bfloat16):
                predicted=refine_firing(self.model,prior['context'],history,predicted,self.options,
                                        seed=1000+prior['frame'],strength=self.settings['weapon_refinement'])
        encoded = self.render(predicted)
        gpu_ms = (time.perf_counter()-tick)*1000
        prior["context"] = torch.cat((prior["context"][:,1:],predicted[:,None]),1)
        prior["actions"] = history[:,1:]
        prior["frame"] += 1
        prior["sequence"],prior["seen"] = sequence,now
        self.generated += 1
        # Historical transport key is 'png'; browsers decode JPEG/PNG by bytes.
        result = dict(png=encoded, frame=prior["frame"],gpu_ms=round(gpu_ms,1))
        prior["last_result"] = result
        return result
