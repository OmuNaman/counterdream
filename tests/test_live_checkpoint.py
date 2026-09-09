import json

import numpy as np
import pytest
import torch

from counterdream.live_checkpoint import sha256, snapshot_folder, snapshot_latest


def test_live_export_pins_latest_ema_and_uses_validation_only(tmp_path, monkeypatch):
    from counterdream import scaled_data

    class ValidationReplay:
        def __init__(self, data, split, context):
            assert split == "val" and context == 8
            self.records = list(range(6))

        def episode(self, index):
            return (np.full((1000, 3, 2, 4), index, dtype=np.uint8),
                    np.full((1000, 51), index, dtype=np.float32))

    monkeypatch.setattr(scaled_data, "DiskReplay", ValidationReplay)
    cfg = dict(context=8, height=2, width=4)
    latest = dict(config=cfg, step=34000, run={}, model={"weight": torch.tensor(1)},
                  ema={"weight": torch.tensor(2)}, optimizer={"must_not_export": True})
    torch.save(latest, tmp_path / "latest.pt")
    torch.save(dict(latest, step=2000), tmp_path / "best.pt")
    original = sha256(tmp_path / "latest.pt")
    report = snapshot_latest(tmp_path, "unused", "a"*32)
    folder = snapshot_folder(tmp_path, "a"*32)
    export = torch.load(folder / "model.pt", weights_only=True)
    assert export["step"] == report["checkpoint_step"] == 34000
    assert export["ema"]["weight"].item() == 2
    assert "model" not in export and "optimizer" not in export
    assert sha256(tmp_path / "latest.pt") == original == report["checkpoint_sha256"]
    assert sha256(folder / "model.pt") == report["export_sha256"]
    with np.load(folder / "seeds.npz", allow_pickle=False) as seeds:
        assert seeds["frames"].shape == (6, 8, 2, 4, 3)
        assert seeds["actions"].shape == (6, 7, 51)
        assert all(name.startswith("Val ") for name in seeds["names"])
    # A newer training save must not silently change this playable snapshot.
    torch.save(dict(latest, step=36000), tmp_path / "latest.pt")
    assert sha256(folder / "model.pt") == report["export_sha256"]
    assert json.loads((folder / "preview.json").read_text())["checkpoint_step"] == 34000
    with pytest.raises(FileExistsError):
        snapshot_latest(tmp_path, "unused", "a"*32)


def test_live_snapshot_rejects_paths_outside_the_run(tmp_path):
    for value in ("../latest.pt", "", "f"*31, "/absolute", "g"*32):
        with pytest.raises(ValueError):
            snapshot_folder(tmp_path, value)
