# CounterDream Dust II — model card

## Intended use

Small-budget research into action-conditioned next-frame generation and
autoregressive neural simulation of a fixed first-person game environment.
The viewer lets a user inspect what the model learned and where it fails.

## Provenance

- Independently implemented conditional U-Net, 9,660,291 parameters.
- Random initialization, no imported pretrained weights.
- Public `TeaPearce/CounterStrike_Deathmatch` Dust II gameplay/action dataset.
- Exact dataset revision and source-file hashes are recorded in the data manifest.
- Training and final evaluation details will be added after the completed run.

## Boundaries

This model predicts pixels. It has no authoritative game state, collision system,
health logic, inventory, bot policy, or multiplayer synchronization. Generated
content can drift, change identity, or fail to follow input. Four context frames
provide limited memory. The output is 112 × 64; enlarging it adds no true detail.

The model is trained on one map and a bounded dataset subset. It is not tested for
unseen games, unseen maps, arbitrary images, or out-of-domain prompts. Its
keyboard/mouse controls follow the training dataset's discretization.

## Evaluation protocol

Whole held-out episodes are separated before window sampling. Model selection
uses 64 fixed-seed validation windows. The final evaluation uses 256 windows from
that same held-out split, four 64-frame autoregressive clips, and matched-noise
left/idle/right branches. It is a validation report, not an untouched test-set
estimate. See README for baselines and caveats.

## Licenses

Our code is MIT. The dataset card declares MIT; original game imagery and
trademarks retain their respective rights. This project is not affiliated with
Valve, World Labs, Google, or DIAMOND's authors. See NOTICE.md for references.
