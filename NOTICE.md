# Attribution and scope

CounterDream's model, training loop, data loader, evaluation tools, and viewer
are independently implemented. The network is initialized from random weights.
No DIAMOND, GameNGen, Marble, or other pretrained model weights are used.

The architecture follows established conditional U-Net / EDM methods. The
action format and frame/action alignment follow the public CS:GO dataset and
were checked against DIAMOND's CS:GO implementation.

- Tim Pearce and Jun Zhu. *Counter-Strike Deathmatch with Large-Scale Behavioural
  Cloning*, IEEE CoG 2022. [Dataset](https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch),
  [source](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning).
  The dataset card declares MIT. The data manifest records its exact revision,
  source members, and SHA-256 hashes. Raw gameplay archives are not included here.
- Eloi Alonso et al. *Diffusion for World Modeling: Visual Details Matter in Atari*,
  NeurIPS 2024. [DIAMOND](https://github.com/eloialonso/diamond/tree/csgo), MIT.
  Reference checkout: `851cefb497733d27f1b85c804104638765860fca`.
- Tero Karras et al. *Elucidating the Design Space of Diffusion-Based Generative
  Models*, NeurIPS 2022. [Paper](https://arxiv.org/abs/2206.00364).

The repository's MIT license covers our code. Third-party datasets, game
imagery, fonts, trademarks, and dependencies retain their respective rights.
CS:GO and Dust II belong to their respective owners. This is an unaffiliated
research project, not a replacement for the original game.
