# Five-GPU CS:GO experiment

Status: five-GPU benchmark, full data preparation, and cloud streaming checks
completed. Full training started on 2026-09-09 at 08:18 UTC from clean commit
`fd5813cb6c1055922cde2c554c5db46ebef6a24b`, with an absolute allocation deadline
of 13:48 UTC. Improved playable quality is not established until generated
rollouts have been evaluated.

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
| Prepared corpus | 100,000 frames | 5,688,000 frames |

The complete index contains **4,956,000 training frames**, **232,000 validation
frames**, and **500,000 test frames**. All 42 preparation partitions are complete.
Three byte-identical boundary recordings are counted once; two corrupt expert
recordings are excluded as documented below. The index checksum, source manifests,
action counts, and split counts are recorded in [the dataset report](reports/v3-data.json).

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
Two slow archives use four readers each. Their completed files are reused by
partition manifests, then counted once in the dataset index. No raw recording
is redownloaded merely to change its partition.
The expert Dust II archive contains 190 recording entries with cleaner control
labels. Two HDF5 entries (expert recordings 90 and 96) are truncated in the source archive. Such files
are excluded only after verifying their ZIP member length and CRC; their source
name, size, checksum, and exclusion reason remain in the dataset index. The
usable expert set contains 188,000 frames.
The sampler mixes 65% uniform episodes, 20% rare-action-weighted
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
checkpoint is selected by validation rollout error across 16 clips at 8, 16, and 32 steps,
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
early checkpointing. It has no application-level automatic retries and records
a persistent marker preventing an accidental second full allocation. Modal can
restart preempted functions even with retries disabled: the same training input
can resume its checkpoint only within its original absolute allocation deadline.
The deadline does not restart with the container. Sixteen CPU workers can
prepare data concurrently; each call is limited to 3,600 seconds. The initial
preparation dispatch skips completed parts and uses one input per container,
preventing fast no-op inputs from skewing the scheduling of longer work. The
34-part invocation has roughly a $4.30 CPU/RAM bound before platform preemptions
or explicit resumptions; eight expert-data partitions add at most about $1.01
per invocation before retries.

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
modal run --detach cloud_scale.py::train --batch 12 --seconds 19800
modal run cloud_scale.py::assess --split val --steps 4
modal run cloud_scale.py::assess --split test --steps 4
modal run cloud_scale.py::fetch --weights
modal run cloud_stream.py::play
```

Preparation is resumable per source episode. Complete the corpus before full
training. A deliberately small prepared subset can be used for the short
benchmark; its cached-data throughput is not full-corpus throughput.
The detached training app survives a local client disconnect; the same allocation
deadline and function timeout still apply.
When migrating an existing unsplit preparation, stop that preparation app and
run `modal run cloud_scale.py::partition_slow` once before resuming `prepare`.
This retires only the two replaced manifests and keeps their frame files.
The benchmark is saved separately in `runs/dust2-v3`. The full-corpus model
starts freshly from random weights in `runs/dust2-v3-full`; pilot weights
are not reused as a substitute for training on the complete corpus.

Inference runs on one cloud H100 near India or Singapore. Modal's narrow-region selection adds a 1.75× multiplier
to this inference allocation (approximately $7.47/hour including CPU/RAM).
Training uses base-price placement. The browser connects to a local
loopback proxy; only control messages and PNG frames cross the cloud link.
The rolling visual history stays on the GPU, and Modal credentials remain in
the local Python process. A direct TLS WebSocket tunnel uses a fresh strong
Bearer token held only by the two server processes. Its address is publicly
reachable, but unauthenticated connections are rejected. It serves one client
on one explicitly started GPU; it cannot autoscale into additional allocations.
Controls and frames flow independently, with a target of 16 generated frames
per second. Missing control heartbeats stop generation after half a second.
Network latency still delays control response, even when frames arrive smoothly.
The viewer has a 12,000-frame budget and 15-minute connection limit. The cloud
function shuts down after 90 idle seconds or 30 active minutes, with a 1,900-second
hard timeout. Restarting the local viewer explicitly allocates another session.
Closing the local app cancels its GPU call. The older per-frame RPC viewer in
`cloud_scale.py` is retained for comparison, not the recommended play command.

## Pilot measurements

The five-H100 pilot completed 3,200 optimizer steps in 467 seconds on a tiny
2,000-frame training subset. A three-frame training unroll took about 0.175
seconds per step; a separate four-frame memory check peaked at 25.2 GB per GPU
before loading the large corpus. A subsequent disjoint GPU-data test verified
identical model weights across ranks and approximately 30.3 GB peak usage with
5.1 GB of data per GPU. These checks establish that the pipeline works, not
that the full model has learned playable dynamics.

The pilot's four-step sampler took about 23 ms per generated frame on an H100.
Per-frame cloud RPC took approximately 230 ms end to end, motivating the direct
continuous stream. A 96-frame direct-stream probe delivered 15.95 fps. A second
96-frame check through the local viewer proxy delivered 15.88 fps, with 131 ms
median control response and 380 ms p95. GPU inference was 24.7 ms median. These
short network samples are specific to this computer and cloud placement, not
a latency guarantee. Reports are in `docs/reports/v3-pilot/network-*.json`.
Pause, resume, reset, starting-view selection, and frame display were also
checked in the browser. Reproduce the local stream benchmark with
`python -m counterdream.benchmark_stream` while the viewer is running and no
browser client is connected.

Main training
is planned for roughly 4–6 hours including overhead, bounded as above; final
runtime and quality remain to be measured on the full corpus.

## Attribution

`counterdream/assets/diamond-test-split.txt` is copied from
[DIAMOND's CS:GO branch](https://github.com/eloialonso/diamond/tree/csgo),
copyright 2024 Eloi Alonso, under the MIT license reproduced in
`counterdream/assets/DIAMOND-LICENSE.txt`. It defines the held-out files only;
no upstream model weights are used.
