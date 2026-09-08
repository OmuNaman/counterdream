"""Summarize recorded action coverage without loading gameplay image arrays."""

import argparse
import json
from pathlib import Path

import numpy as np

from counterdream.actions import KEYS, MOUSE_X, MOUSE_Y


def audit(root, output):
    root = Path(root)
    report = {
        "note": "Action activation frequencies; not evidence of learned control accuracy.",
        "splits": {},
    }
    labels = [
        *KEYS,
        "fire",
        "scope",
        *[f"mouse_x_{x}" for x in MOUSE_X],
        *[f"mouse_y_{y}" for y in MOUSE_Y],
    ]
    for split in ("train", "val"):
        count = 0
        totals = np.zeros(51, dtype=np.int64)
        files = sorted((root / split).glob("*.npz"))
        for file in files:
            with np.load(file, allow_pickle=False) as record:
                actions = record["actions"]
            count += len(actions)
            totals += (actions > 0.5).sum(0)
        report["splits"][split] = {
            "episodes": len(files),
            "frames": count,
            "active_frames": dict(zip(labels, totals.tolist())),
        }
    Path(output).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.data, args.output)
