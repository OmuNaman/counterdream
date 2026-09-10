"""Shared Modal image and artifact locations without entrypoint imports."""
import modal

VOLUME_NAME = "counterdream-artifacts-v1"
DATA = "/artifacts/data-v3"
PILOT = "/artifacts/runs/dust2-v3"
RUN = "/artifacts/runs/dust2-v3-full"
volume = modal.Volume.from_name(VOLUME_NAME)
gpu_base_image = (modal.Image.debian_slim(python_version="3.11")
                 .pip_install("torch==2.6.0", "numpy==1.26.4", "Pillow==11.1.0",
                              "h5py==3.12.1", "requests==2.32.3", "imageio==2.37.0",
                              "imageio-ffmpeg==0.6.0", "fastapi==0.115.8", "uvicorn==0.34.0"))
