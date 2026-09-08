# Running the released model locally

The local viewer needs a trained checkpoint and four short recorded starting
views. It does not need CS:GO installed, a Modal account, or an API key.

## Windows with an NVIDIA GPU

Use Python 3.11 or 3.12 and an NVIDIA driver supporting CUDA 12.4. From the cloned
repository, run these commands in PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m counterdream.download
.venv\Scripts\python.exe -m counterdream.serve --checkpoint artifacts/model.pt --seeds artifacts/seeds.npz
```

Use `py -3.11` on the first line if that is your installed Python version. This
avoids PowerShell activation-policy changes. The install downloads PyTorch and
dependencies in addition to the roughly 39 MB checkpoint.

## Linux with an NVIDIA GPU

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -e .
.venv/bin/python -m counterdream.download
.venv/bin/python -m counterdream.serve --checkpoint artifacts/model.pt --seeds artifacts/seeds.npz
```

These commands match the experiment's PyTorch 2.6 / CUDA 12.4 environment. For a
different GPU generation or driver, choose a suitable build from the
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/).
The model was tested on an H100 and an RTX 4070 Laptop GPU. This implementation
uses CUDA where available; its CPU fallback works much more slowly. It does not
currently use Apple's MPS backend.

## Play

Open **http://127.0.0.1:7860**, then press **Connect & Play**. The visible frame
counter counts model-generated frames since reset. The initial view is recorded;
subsequent frames use the model's own predictions as context.

WASD moves, arrow keys turn, F fires, Space jumps, and R reloads. Click and drag
also turns the view; left click fires while held. Escape pauses. After a reset,
press Resume to continue. Controls are inputs to the learned model: an input does
not guarantee an accurate simulation of that game mechanic.

The viewer requests at most eight generated frames per second by default. The
model's reported GPU latency excludes browser display and may differ from the
rate you experience. Eight diffusion steps are the evaluated default; four is
faster and sixteen is slower. Those options can change visual quality.

## Common issues

- **The page says CPU:** check `python -c "import torch; print(torch.cuda.is_available())"`
  using the same environment. Install a CUDA-enabled PyTorch build if you have a
  supported NVIDIA GPU. Installing a separate CUDA toolkit is not generally needed
  for the prebuilt wheel; a compatible driver is needed.
- **The world looks blurry or drifts:** this is a 112 × 64 research model with four
  frames of memory. Reset to a saved starting view. Browser zoom does not increase
  the model's resolution.
- **Session frame limit:** the server allows 2,000 generated frames. Stop the local
  server with Ctrl+C and run the same viewer command again to begin another session.
  Local inference does not spend Modal credits.
- **Port 7860 is in use:** stop the earlier CounterDream viewer before starting a
  second one. The viewer intentionally binds to the local computer.
- **Download checksum mismatch or partial download:** the downloader keeps incomplete
  files separate and does not install them as checkpoints. Retry after checking
  connectivity; it will reuse existing files whose size and hash match the release.
