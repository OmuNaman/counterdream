# Contributing

CounterDream is a small, reproducible experiment in learned game dynamics. Useful
contributions improve measured action control, rollout stability, inference speed,
or the ability to repeat an experiment.

## Local checks

Install Python 3.11 or 3.12, PyTorch, and `pip install -e ".[dev]"`, then run
`python -m pytest tests -q`. The tests run on CPU and never allocate cloud GPUs.
For formatting, use Ruff's default formatter. Keep training and inference
dependencies out of the browser.

## Model changes

Keep original checkpoints and dataset manifests. Record the source revision,
initialization, seed, training steps, sampler settings, and hardware. Compare
against repeat-frame and matched-noise shuffled-action baselines using the same
windows. Include an autoregressive video with generated frames fed back into the
model. A lower denoising loss alone is not evidence of a better simulator.

Changes that alter checkpoint interpretation must use a new model version and
preserve loading of released weights. Report validation tuning honestly; do not
describe the same split as an untouched test set.

## Issues and pull requests

For bugs, include the OS, Python/PyTorch versions, checkpoint release, exact
reproduction steps, and a short error message. For research changes, describe the
observed behavior and attach the relevant measurements. Never include credentials,
private account details, or raw dataset archives in issues or commits.

Project code is MIT. Retain relevant dataset and method attribution.
