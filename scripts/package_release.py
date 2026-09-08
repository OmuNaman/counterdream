"""Package an evaluated checkpoint; refuse mismatched weights and measurements."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(evaluation, output, tag, training_state=None):
    import torch

    source, target = Path(evaluation).resolve(), Path(output).resolve()
    project = Path(__file__).resolve().parents[1]
    metrics = json.loads((source / "evaluation.json").read_text())
    checkpoint = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["step"] != metrics["checkpoint_step"]:
        raise ValueError("Checkpoint step differs from evaluation")
    if digest(source / "model.pt") != metrics["checkpoint_sha256"]:
        raise ValueError("Checkpoint weights differ from evaluated weights")
    if checkpoint["run"]["pretrained_weights"] or metrics["pretrained_weights"]:
        raise ValueError("This release expects CounterDream's training from scratch")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("LICENSE", "NOTICE.md"):
        shutil.copyfile(project / name, target / name)
    names = ["model.pt", "seeds.npz", "evaluation.json"]
    for name in [
        *names,
        "counterfactual.png",
        *[f"rollout-{i}.mp4" for i in range(1, 5)],
    ]:
        shutil.copyfile(source / name, target / name)
    if training_state:
        state = torch.load(training_state, map_location="cpu", weights_only=True)
        if not all(
            key in state
            for key in ("optimizer", "model", "ema", "cuda_rng", "numpy_rng")
        ):
            raise ValueError("Incomplete training state")
        shutil.copyfile(training_state, target / f"training-step-{state['step']}.pt")
    manifest = {
        "tag": tag,
        "checkpoint_step": checkpoint["step"],
        "files": {
            name: {
                "bytes": (target / name).stat().st_size,
                "sha256": digest(target / name),
            }
            for name in names
        },
    }
    (project / "counterdream" / "release.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (target / "SHA256SUMS.txt").write_text(
        "".join(
            f"{digest(file)}  {file.name}\n"
            for file in sorted(target.iterdir())
            if file.is_file() and file.name != "SHA256SUMS.txt"
        )
    )
    print(
        json.dumps({"directory": str(target), "tag": tag, "step": checkpoint["step"]})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", default="v0.1.0")
    parser.add_argument("--training-state")
    args = parser.parse_args()
    package(args.evaluation, args.output, args.tag, args.training_state)
