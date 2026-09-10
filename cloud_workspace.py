"""Short actual-training check in a newly authorized workspace."""
import json

import modal
from cloud_scale import _launch, source_manifest, gpu_image
from counterdream.cloud_config import volume

app = modal.App("counterdream-workspace-training-check")
workspace_image = gpu_image.add_local_python_source("cloud_scale")


@app.function(image=workspace_image, gpu="H100:5", cpu=20, memory=65536,
              timeout=600, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifacts": volume})
def train_check(source):
    return _launch(300, 12, source, False, "/artifacts/runs/workspace-check-v3",
                   target_steps=128, data="/artifacts/data-v3-transfer-probe")


@app.local_entrypoint()
def check():
    print(json.dumps(train_check.remote(source_manifest())), flush=True)
