# CounterDream Dust II — model card

## Intended use and behavior

CounterDream is a small research model for action-conditioned next-frame generation
and autoregressive neural simulation of recorded CS:GO gameplay on Dust II. The
viewer starts from four recorded frames. Every subsequent frame is generated and
fed back into the model. No game engine supplies those subsequent frames.

## Provenance

- Independently implemented conditional U-Net, **9,660,291 parameters**.
- **Random initialization**, seed 42, no imported pretrained weights.
- Version 2: widths 64/128/192/256, residual FiLM conditioning, four-head bottleneck
  attention, Gaussian Fourier noise features, EDM preconditioning, context noise
  augmentation with 20% clean contexts, and EMA inference weights.
- One **H100 80 GB**, PyTorch **2.6.0 + CUDA 12.4**, batch **48**.
- AdamW, learning rate 0.0002, 200-step warmup, cosine decay to 20% of the base
  rate, weight decay 0.01, gradient clipping at 1, and alternating two-step
  unrolling after step 2,000. EMA decay increases to 0.999.
- Public `TeaPearce/CounterStrike_Deathmatch` Dust II gameplay/action dataset.
- Exact dataset revision and source-file hashes are recorded in the data manifest.
- The experiment targets at most 60,000 optimizer steps with a 7,200-second loop
  allowance. The exact released step is recorded in its checkpoint and evaluation.

[Run configuration and source hashes](reports/run.json) ·
[Experiment history, including the discarded first attempt](EXPERIMENTS.md).
The random seed and pinned dependencies aid reproducibility; different CUDA
hardware and kernels are not promised to reproduce weights bit for bit.

## Data and action coverage

We use 100 complete 1,000-frame source files from the dataset's first Dust II
archive: **90,000 training frames and 10,000 validation frames**. Every tenth source
file is held out before temporal windows are formed. A window never crosses a
source-file boundary. Source BGR images are converted once to RGB. Action `a[t]`
with frame `x[t]` predicts `x[t+1]`.

The pinned dataset revision is `265c6e5ac7aa335f58a2f2e864aad176fecfedde`.
[The manifest](reports/data-manifest.json) records original source members and
their SHA-256 hashes. This is the same map and collection, not unseen maps or
independently collected sessions. Nearby source files may share locations.

Selected action coverage, measured over complete source frames:

| Input active | Training frames | Validation frames |
|---|---:|---:|
| Forward | 46,516 | 4,767 |
| Strafe left | 23,486 | 2,582 |
| Strafe right | 22,425 | 2,668 |
| Fire | 5,309 | 647 |
| Jump | 755 | 90 |
| Reload | 157 | 14 |
| Crouch | 4,500 | 0 |

Counts measure recorded inputs, not successful independent actions or learned
mechanic accuracy. Jumping and reloading are sparsely represented. No held-out
crouch frames occur in this subset, so this evaluation cannot support a claim of
validated crouch behavior. [Full action coverage](reports/data-coverage.json).

## Boundaries

This model predicts pixels. It has no authoritative game state, collision system,
health logic, inventory, bot policy, or multiplayer synchronization. Generated
content can drift, change identity, or fail to follow input. Four context frames
provide limited memory. The output is 112 × 64; enlarging it adds no true detail.

The model is trained on one map and a bounded dataset subset. It is not tested for
unseen games, unseen maps, arbitrary images, or out-of-domain prompts. Its
keyboard/mouse controls follow the training dataset's discretization.

## Evaluation protocol

The release sampler uses eight Euler steps on a Karras schedule with sigma_max 20.
Evaluation includes 256 fixed-seed one-step windows, repeat-previous-frame and
matched-noise shuffled-action baselines, four 64-frame autoregressive rollouts,
and matched-noise left/idle/right action branches. MSE is measured in RGB scaled
to [0,1]; aggregate PSNR is -10 log10 of that mean MSE.

Reported inference latency measures model sampling on the named GPU, excluding
browser display and network transport. Videos are presented at eight frames per
second for inspection; this is separate from the measured sampling throughput.

Sampling and checkpoint choices were tuned on the validation split. These metrics
are not an untouched test-set estimate. PSNR can favor blur. Lower shuffled-action
accuracy and distinct counterfactual branches indicate action dependence, but do
not prove correct geometry, game physics, or combat. Long rollouts can diverge even
when one-step predictions are plausible. All four evaluated clips should be
inspected, including failures.

The original running process used a legacy sigma-5 periodic validation monitor.
Release evaluation uses sigma 20 and exports the exact measured weights; see the
[experiment record](EXPERIMENTS.md) for the distinction and source revisions.

## Licenses

CounterDream code and released model weights are provided under MIT. The dataset
card declares MIT; original game imagery and trademarks retain their respective
rights. Seed frames and evaluation visuals come from that dataset and the model.
This project is not affiliated with Valve, World Labs, Google, or DIAMOND's authors.
See [NOTICE.md](../NOTICE.md) for dataset and method attribution.
