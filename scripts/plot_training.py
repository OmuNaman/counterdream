"""Render measured learning curves from a CounterDream run directory."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(run, output, reports=None):
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
        title="Periodic validation monitor (64 windows)",
        ylabel="PSNR (dB) · higher is better",
        xlabel="Optimizer steps",
    )
    axes[0].legend(frameon=False)
    if reports:
        measured = [
            json.loads(path.read_text())
            for path in Path(reports).glob("step-*/evaluation.json")
        ]
        measured = sorted(
            (
                r
                for r in measured
                if r.get("sampler_sigma_max") == 20 and r.get("sampler_steps") == 8
            ),
            key=lambda r: r["checkpoint_step"],
        )
        if not measured:
            raise ValueError("No reports with the release sampling configuration")
        axes[0].clear()
        eval_steps = [r["checkpoint_step"] for r in measured]
        axes[0].plot(
            eval_steps,
            [r["next_frame_psnr"] for r in measured],
            "o-",
            color="#367d46",
            label="World model",
        )
        axes[0].plot(
            eval_steps,
            [r["repeat_frame_psnr"] for r in measured],
            "--",
            color="#737c89",
            label="Repeat previous frame",
        )
        axes[0].set(
            title="Release sampler · 256 validation windows",
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
    if not reports and not all("sampler_sigma_max" in row for row in rows):
        axes[0].set_title(
            "Legacy monitor: sigma_max=5 (not release sampler)", fontsize=10
        )
        fig.supxlabel(
            "Release evaluation uses sigma_max=20; compare its metrics separately.",
            fontsize=9,
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("output")
    p.add_argument(
        "--reports", help="Directory containing step-*/evaluation.json reports"
    )
    a = p.parse_args()
    plot(a.run, a.output, a.reports)
