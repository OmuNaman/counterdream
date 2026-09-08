# Experiment record

This record separates training changes from evaluation changes so that released
results can be traced to the weights that produced them.

## Data and initial run

The experiment downloaded 100 complete source episodes from the first Dust II
archive in `TeaPearce/CounterStrike_Deathmatch`, pinned to revision
`265c6e5ac7aa335f58a2f2e864aad176fecfedde`. Every tenth episode is held out. All
models use 90,000 training frames and 10,000 validation frames at 112 × 64 RGB.
The loader records original source-file hashes and does not redistribute archives.

A 100-step H100 pilot verified gradients, checkpoint export, and end-to-end
generation. It was a pipeline check, not a quality result. An initial full attempt
using version 1 was stopped at 6,000 steps because rollout samples and validation
were poor. These weights are not the released model.

## Version 2

Version 2 started again from random weights with seed 42. It uses smooth Gaussian
Fourier noise features and explicitly trains the clean-context condition on 20%
of examples. The former version's high-frequency embedding and untrained clean
condition were unreliable for sampling.

The training source is commit
`d9c65da1ed21d903127ebacbce4194dbb4283905`. Its immutable source-file hashes are
included in `run.json`. Training uses one H100, batch 48, AdamW, an EMA, context
noise augmentation, and alternating two-step unrolling after step 2,000.

## Sampling calibration

At 10,000 steps, an independent evaluation exposed a mismatch in the initial
sampling noise scale. The original sampler started at sigma 5; version 2 release
inference starts at sigma 20, within the training noise range. The training
objective and optimizer were unchanged. The corrected 8-step sampler obtained
19.506 dB PSNR on 256 validation windows, versus 18.608 dB for repeating the previous
frame. Shuffling action histories increased MSE from 0.011206 to 0.015441.

The running training process retained its original sigma-5 periodic monitor. Its
`metrics.jsonl`, `best.pt`, and automatic root `model.pt` therefore reflect that
legacy sampler. Their scores must not be plotted as if they used the release
sampler. The release is selected from separately evaluated snapshots using sigma
20, and the evaluator exports the exact EMA weights it measured. An evaluation
of the final training state is requested with `assess --latest`.

Sampling settings and checkpoint choices were tuned on this validation split.
The released metrics are not an untouched test estimate. The source episodes are
from the same map and collection; neighboring episodes may share visual content.

## Rollout stability check

At step 30,000, next-frame PSNR improved to 19.984 dB on the 256-window evaluation.
Longer autoregressive clips still drifted toward walls and lost weapon detail. A
separate bounded check compared eight versus sixteen denoising steps and two
nonzero context-noise conditioning levels on the same checkpoint. It used 64
validation windows and four 64-frame rollouts per setting.

Sixteen steps slightly reduced error at horizon 64, but worsened one-step error
and horizons 16 and 32. Context-noise conditioning of 0.03 or 0.1 did not consistently
improve rollouts. The release therefore retains eight steps and the trained clean
context condition. These settings do not solve long-horizon drift.

[Measurements](reports/stability-comparison.json) ·
[Bounded comparison script](../scripts/sampler_stability_modal.py).
