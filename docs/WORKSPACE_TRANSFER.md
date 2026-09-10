# Continuing in another authorized Modal workspace

The original workspace stopped the first full allocation after approximately
17 minutes of data loading because its spend limit was exceeded. It produced
no optimizer steps or full-run checkpoint. The user supplied a different active
workspace and confirmed at least $130 of remaining spend allowance. A CPU-only
compute check succeeded there, and the original workspace still permits reading
the saved corpus. The original stopped app and allocation marker are retained.

The migration completed on 2026-09-09: 5,688 recordings and 241,422,928,128 bytes
of arrays transferred. All 12 partitions completed, file sizes were checked, and
the rebuilt destination index exactly matches the original SHA-256
`7045b9e86beec3851cbd6b214147cc0fd694c8844d90325d8195ef2323996915`.
The source was not modified and no training weights were transferred.

`cloud_transfer.py` copies only the prepared dataset into the destination's
`counterdream-artifacts-v1` Volume. It verifies the pinned original index and
source manifests, validates transferred array shapes and lengths, records array
checksums, and rebuilds the destination index for an exact checksum comparison.
Twelve disjoint CPU workers can resume completed recordings. Each invocation is
capped at one hour, with a 55-minute work allowance; no GPUs are used for transfer.
No training weights are transferred. The destination full model starts randomly.

Source credentials are supplied through `COUNTERDREAM_SOURCE_TOKEN_ID` and
`COUNTERDREAM_SOURCE_TOKEN_SECRET`, passed only to the CPU transfer workers using
an unnamed Modal Secret. Destination credentials use the normal Modal profile or
environment. Neither credential pair belongs in source files, function arguments,
browser code, or logs. The source client only reads the original volume.

After selecting the destination workspace and providing the source credentials:

```sh
modal volume create counterdream-artifacts-v1
modal run cloud_transfer.py::probe
modal run cloud_workspace.py::check
modal run cloud_transfer.py::transfer
```

Create the volume only if it does not already exist. The probe copies two training
recordings, one validation recording, and one test recording into a separate
directory. The GPU check performs 128 optimizer steps on five H100s, including
the existing four-frame memory check and rank-weight consistency checks. Its
function is capped at ten minutes and its training allowance at five minutes;
the resulting weights are not reused for full training.

After the GPU check and full transfer succeed, the regular five-H100 full run
uses its existing 5.5-hour allowance and six-hour function timeout. The previous
allocation's estimated $6.08 and all earlier project costs remain part of the
user's original $200–$300 total budget. The new allocation's base compute bound
is approximately $127.20, leaving room for transfer, the short test, evaluation,
and limited inference within a conservative $200 working target. This is a cost
estimate, not a reconciled account invoice. No existing deadline is reset and no
active allocation is duplicated; the new workspace records its own bounded run.
