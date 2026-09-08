import numpy as np
import pytest
from counterdream.data import Replay


def test_windows_never_cross_episodes_and_actions_are_causal(tmp_path):
    root = tmp_path / "train"
    root.mkdir()
    for ep in range(2):
        frames = np.full((12, 8, 8, 3), ep * 200, dtype=np.uint8)
        actions = np.repeat(np.arange(12)[:, None], 51, axis=1).astype(np.float32)
        np.savez(root / f"{ep}.npz", frames=frames, actions=actions)
    replay = Replay(tmp_path, "train")
    obs, acts = replay.batch(30, "cpu", np.random.default_rng(3), horizon=2)
    assert obs.shape == (30, 6, 3, 8, 8)
    assert acts.shape == (30, 5, 51)
    for seq, actions in zip(obs, acts):
        assert (seq == seq[0]).all()
        assert (actions[1:, 0] - actions[:-1, 0] == 1).all()


def test_missing_split_fails(tmp_path):
    with pytest.raises(ValueError, match="No val episodes"):
        Replay(tmp_path, "val")
