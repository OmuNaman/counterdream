# Compute limits and cost accounting

Each training and evaluation job has explicit resource and time limits. Review
the per-invocation costs below before allocating compute. The source-level limits
do not configure an account-wide spending cap.

Rates checked on 2026-09-08 at [Modal pricing](https://modal.com/pricing):

| Resource | Published rate |
|---|---:|
| H100 | $0.001097 / second ($3.9492 / hour) |
| L4 | $0.000222 / second ($0.7992 / hour) |
| CPU physical core | $0.0000131 / second |
| Memory | $0.00000222 / GiB / second |

These are compute estimates, not account invoices. Startup, evaluation, image
builds, data preparation, CPU, RAM, storage, and network can add charges. Free
credits and unrelated account workloads are not included in these estimates.

## Built-in controls

- Data preparation: no GPU, one container, 3,600-second function timeout, at most
  200 episodes per invocation. The initial dataset targets 100 episodes.
- Training: one H100 per invocation, no automatic retry, explicit maximum
  optimizer steps, and a wall-time check with checkpointing. The initial full run
  is configured for 7,200 seconds. The enclosing function times out at 15,000
  seconds, including loading and cleanup; that is the backstop if code stalls.
- Evaluation: one H100, no retries, 1,200-second timeout.
- Interactive inference: one L4, zero minimum containers, 15-second scale-down
  window, a local-only viewer, and 2,000 generated frames per viewer process.
- There are no cron jobs, background schedules, public GPU endpoints, or training
  jobs triggered by GitHub Actions.

One fully timed-out training invocation can consume about $16.46 in H100 time
alone. The planned two-hour training loop corresponds to about $7.90 GPU-only,
before other charges. These limits apply per invocation, not across repeated
manual runs. Multiple independent Modal Apps can each allocate a container.

If stopping early, use the exact App ID printed for that run:

```bash
modal app stop APP_ID
```

Do not stop unrelated applications. Do not launch a replacement writer until the
prior run has stopped. The Modal dashboard is the source of truth for actual
billed spend; these source-level controls do not set a workspace billing cap.
