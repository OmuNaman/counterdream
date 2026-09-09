# Five-GPU CS:GO experiment

Status: implementation and short benchmark in progress. No improved playable
quality is claimed until generated rollouts have been evaluated.

The user chose training from random initialization, with no DIAMOND or other
pretrained weights. v0.1.0 remains available unchanged as the original release.

## What changes

| | Released v0.1.0 | Scale experiment |
|---|---:|---:|
| Model parameters | 9,660,291 | 61,827,651 |
| RGB output | 112 × 64 | 160 × 88 |
| Visual history | 4 frames | 8 frames |
| Maximum training unroll | 2 frames | 4 frames |
| GPUs per training allocation | 1 H100 | 5 H100 |
| Source corpus target | 100,000 frames | Approximately 5.5 million frames |

Eight-frame history is approximately half a second at the source recording rate.
It is not persistent spatial memory. Larger data and compute do not guarantee
stable physics, object identity, or indefinitely playable trajectories.

The dataset is pinned to revision
`265c6e5ac7aa335f58a2f2e864aad176fecfedde` in
[TeaPearce/CounterStrike_Deathmatch](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch).
Its 28 main Dust II archives total 706,623,969,792 bytes. CPU workers process
bounded byte ranges directly into RGB uint8 arrays in a Modal Volume; the raw
archives are not copied to the user's laptop. Source episode SHA-256 values,
dimensions, split assignment, and action counts are recorded.
An additional 190,000 frames from the expert Dust II archive provide cleaner
control labels. The sampler mixes 65% uniform episodes, 20% rare-action-weighted
episodes, and 15% expert episodes when expert data is available. Expert files
also have a separate deterministic validation split.

The published DIAMOND test-file list is excluded from training. An additional
deterministic 5% of remaining source files is validation. Files are not guaranteed
to represent independent play sessions, so this is not an unseen-session claim.
Sampler and checkpoint choices use validation only; final test results are
reported separately. Changing corpus size resets validation checkpoint selection.

The trainer uses one process per GPU with PyTorch DistributedDataParallel. Each
rank independently samples causal windows. The full run divides the training
episodes across five GPUs and loads each disjoint shard as uint8 observations;
approximately 40 GB of data per GPU leaves room for training activations.
Loading happens once before optimization, avoiding repeated random cloud reads.
The short pilot uses memory-mapped episode files. Rank weight checksums are
compared at each checkpoint, and per-rank RNG state is preserved for resume.
Training unrolls from one to four predictions and periodically replaces a
denoised training context with a fully sampled generated frame. The final
checkpoint is selected by validation rollout error at 8, 16, and 32 steps,
rather than one-step image error alone. Human inspection of videos is also
required: low pixel error can reward blur.

## Cost and runtime bounds

Rates checked 2026-09-09 at [Modal pricing](https://modal.com/pricing).
Five H100s cost $19.746/hour GPU-only. With 20 CPU cores and 64 GiB memory,
the requested allocation is approximately $21.20/hour at base rates, before
network/storage/account-specific charges or credits.

The benchmark function has a 900-second hard timeout (about $5.30 maximum base
compute). The full training function has a 21,600-second hard timeout (about
$127.20 maximum base compute), with its loop limited to 19,800 seconds and
early checkpointing. It has no automatic retries and records a persistent
marker preventing an accidental second full allocation. Eight CPU workers can
prepare data concurrently; each call is limited to 3,600 seconds. The initial
28-shard invocation has roughly a $3.54 CPU/RAM upper bound at requested resources.

These are application limits, not a Modal account spending cap. Existing project
spend was estimated below $15, not reconciled against an invoice. Reserve room
within the user's $200–$300 budget for data retries, evaluation, and inference.
GPU quantity is not a fivefold speedup guarantee; use measured distributed
throughput and dataset I/O to estimate runtime.

## Commands

```sh
modal run cloud_scale.py::prepare --shards 28
modal run cloud_scale.py::expert
modal run cloud_scale.py::index
modal run cloud_scale.py::benchmark --batch 12
modal run cloud_scale.py::train --batch 12 --seconds 19800
modal run cloud_scale.py::assess --split val --steps 4
modal run cloud_scale.py::assess --split test --steps 4
modal run cloud_scale.py::fetch --weights
modal run cloud_scale.py::play
```

Preparation is resumable per source episode. Complete the corpus before full
training. A deliberately small prepared subset can be used for the short
benchmark; its cached-data throughput is not full-corpus throughput.
The benchmark is saved separately in `runs/dust2-v3`. The full-corpus model
starts freshly from random weights in `runs/dust2-v3-full`; pilot weights
are not reused as a substitute for training on the complete corpus.

Inference runs on one private cloud H100 near India or Singapore, with traffic
routed through Mumbai. Modal's narrow-region selection adds a 1.75× multiplier
to this inference allocation (approximately $7.47/hour including CPU/RAM).
Training uses base-price placement. The browser connects to a local
loopback proxy; only control messages and PNG frames cross the cloud link.
The rolling visual history stays on the GPU, and Modal credentials remain in
the local Python process. There is no public GPU endpoint. Idle containers
scale down after 15 seconds, which can require resetting an expired session.
Network round-trip time adds to model latency. The viewer has a 12,000-frame
budget and 15-minute connection limit; the GPU process has a 20,000-frame
and 45-minute allocation limit. Closing the local app stops its Modal app.

## Attribution

`counterdream/assets/diamond-test-split.txt` is copied from
[DIAMOND's CS:GO branch](https://github.com/eloialonso/diamond/tree/csgo),
copyright 2024 Eloi Alonso, under the MIT license reproduced in
`counterdream/assets/DIAMOND-LICENSE.txt`. It defines the held-out files only;
no upstream model weights are used.
