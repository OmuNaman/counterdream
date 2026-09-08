# Which world models inspired this project?

Checked against primary project pages on 2026-09-08.

| Project | What it demonstrates | Relation to CounterDream |
|---|---|---|
| [World Labs Marble](https://docs.worldlabs.ai/) | Creation and editing of persistent 3D environments from text, images, video, or coarse structure | A useful spatial-generation reference; not the architecture we train here |
| [Google GameNGen](https://gamengen.github.io/) | A diffusion model simulating interactive Doom gameplay | The neural-game-engine idea the initial request referred to |
| [Oasis / Decart and Etched](https://oasis-model.github.io/) | Interactive Minecraft-like video world generation | One possible Minecraft project behind the request; [inference source](https://github.com/etched-ai/open-oasis) |
| [Microsoft MineWorld](https://github.com/microsoft/mineworld) | An action-conditioned autoregressive Minecraft world model | Another open-source Minecraft reference; we did not assume it was the exact project mentioned |
| [Skywork Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | A family of interactive world-generation models | A further possible Minecraft reference |
| [DIAMOND CS:GO](https://github.com/eloialonso/diamond/tree/csgo) | A playable diffusion world model trained on recorded CS:GO frames and actions | The most directly relevant reference for our data alignment and experimental design |

CounterDream trains its own small network from random weights. Its scope is much
smaller than these systems. It predicts action-conditioned frames on one map;
it does not generate a persistent editable 3D scene graph like a 3D authoring tool.

The downloadable code and model weights of any reference retain their own
licenses. Calling a project open-source here is not a blanket license for every
underlying game asset, training example, or commercial use.
