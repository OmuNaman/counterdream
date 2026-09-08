"""Render measured learning curves from a CounterDream run directory."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(run, output):
    run = Path(run)
    rows = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text().splitlines()
        if line
    ]
    if not rows:
        raise ValueError("No measured validation history")
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    axes[0].plot(
        steps,
        [r["next_frame_psnr"] for r in rows],
        color="#367d46",
        lw=2,
        label="World model",
    )
    axes[0].plot(
        steps,
        [r["repeat_frame_psnr"] for r in rows],
        color="#737c89",
        ls="--",
        label="Repeat previous frame",
    )
    axes[0].set(
        title="Held-out next-frame prediction",
        ylabel="PSNR (dB) · higher is better",
        xlabel="Optimizer steps",
    )
    axes[0].legend(frameon=False)
    axes[1].plot(steps, [r["train_loss"] for r in rows], color="#336b9e", lw=2)
    axes[1].set(
        title="Smoothed training objective",
        ylabel="EDM denoising loss",
        xlabel="Optimizer steps",
    )
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(style="plain", axis="x")
    fig.suptitle(
        "CounterDream · random initialization · Dust II", fontsize=14, fontweight="bold"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("output")
    a = p.parse_args()
    plot(a.run, a.output)
