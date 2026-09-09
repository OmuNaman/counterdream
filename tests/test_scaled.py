import json
import numpy as np
import torch

from counterdream.model import ModelConfig, WorldModel
from counterdream.scaled_data import DiskReplay, GPUShardReplay, episode_split, build_index, replay_weights
from counterdream.train_distributed import SequenceObjective


def test_published_test_files_never_enter_training():
    heldout = {"hdf5_dm_july2021_1.hdf5", "hdf5_dm_july2021_2.hdf5"}
    for name in heldout:
        assert episode_split(name, heldout) == "test"
    assert episode_split("anything.hdf5", heldout) == episode_split("anything.hdf5", heldout)


def test_identical_duplicate_archive_members_are_counted_once(tmp_path):
    for name in ("one","two"):
        folder=tmp_path/name
        folder.mkdir()
        record=dict(source="same.hdf5",sha256="same-content",file="frames.npy",actions="actions.npy",frames=1000,split="train")
        (folder/"manifest.json").write_text(json.dumps(dict(shard=name,complete=True,episodes=[record])))
    report=build_index(tmp_path)
    assert report["counts"]["train"]==1000
    assert report["identical_duplicates_skipped"]==["same.hdf5"]


def test_expert_sampling_keeps_coverage_and_adds_clean_examples():
    records=[dict(action_counts=[0]*13,expert=i==0) for i in range(100)]
    weights=replay_weights(records)
    assert abs(weights.sum()-1)<1e-12
    assert weights[0]>.15
    assert (weights[1:]>0).all()


def test_disk_replay_causality_and_bounded_open_episodes(tmp_path,monkeypatch):
    records = []
    for i in range(4):
        frames = np.zeros((24, 3, 8, 8), dtype=np.uint8)
        frames[:, 0] = i * 40
        frames[:, 1] = np.arange(24)[:, None, None]
        acts = np.repeat(np.arange(24)[:, None], 51, 1).astype(np.float32)
        np.save(tmp_path / f"{i}.frames.npy", frames)
        np.save(tmp_path / f"{i}.actions.npy", acts)
        records.append(dict(file=f"{i}.frames.npy", actions=f"{i}.actions.npy", split="train",
                            action_counts=[0]*13, frames=24))
    (tmp_path / "index.json").write_text(json.dumps(dict(episodes=records,height=8,width=8)))
    replay = DiskReplay(tmp_path, context=8, cache_size=2)
    obs, acts = replay.batch_numpy(50, np.random.default_rng(17), horizon=4)
    assert len(replay.cache) <= 2
    assert obs.shape == (50, 12, 3, 8, 8)
    for frames, actions in zip(obs, acts):
        assert (frames[:, 0] == frames[0, 0]).all()
        np.testing.assert_array_equal(frames[:-1, 1, 0, 0], actions[:, 0])
        np.testing.assert_array_equal(np.diff(frames[:, 1, 0, 0].astype(int)), np.ones(11))
    monkeypatch.setattr(torch.cuda,"mem_get_info",lambda device:(100_000_000_000,100_000_000_000))
    shards=[GPUShardReplay(tmp_path,rank,2,"cpu") for rank in range(2)]
    assert {r["file"] for r in shards[0].records}.isdisjoint(r["file"] for r in shards[1].records)
    assert sum(len(s.records) for s in shards)==len(records)
    for shard in shards:
        frames,actions = shard.batch_device(20,np.random.default_rng(123),horizon=4)
        original = (frames+1)*127.5
        torch.testing.assert_close(original[:,:-1,1,0,0],actions[:,:,0],atol=1e-4,rtol=0)


def test_unrolled_objective_trains_and_uses_generated_context():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    model = WorldModel(ModelConfig(height=16, width=24, context=8, base=8, cond_dim=32, version=3))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    obs = torch.rand(2, 11, 3, 16, 24)*2-1
    acts = torch.rand(2, 10, 51)
    for _ in range(3):
        optimizer.zero_grad()
        SequenceObjective(model)(obs, acts).backward()
        optimizer.step()
    optimizer.zero_grad()
    observed = []
    hook = model.register_forward_pre_hook(lambda module, args: observed.append(args[2].detach().clone()))
    loss = SequenceObjective(model)(obs, acts)
    loss.backward()
    hook.remove()
    assert torch.isfinite(loss)
    assert model.action_emb[0].weight.grad.abs().sum() > 0
    assert len(observed) == 3
    assert not torch.allclose(observed[1][:, -1], obs[:, 8])
