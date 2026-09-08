import torch
from counterdream.model import ModelConfig, WorldModel, load_model

torch.set_num_threads(2)


def test_loss_backward_and_conditioning():
    torch.manual_seed(4)
    model = WorldModel(ModelConfig(base=16))
    obs = torch.rand(2, 4, 3, 64, 112) * 2 - 1
    act = torch.zeros(2, 4, 51)
    target = torch.rand(2, 3, 64, 112) * 2 - 1
    loss, pred = model.loss(obs, act, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.out[-1].weight.grad.abs().sum() > 0
    assert pred.shape == target.shape
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    opt.step()
    # Zero-initialized residual/output layers start transmitting conditioning
    # gradients after the first updates, not on the very first backward pass.
    for _ in range(3):
        opt.zero_grad()
        loss, _ = model.loss(obs, act, target)
        loss.backward()
        opt.step()
    first = model.sample(obs, act, steps=2, seed=12)
    changed = act.clone()
    changed[:, :, 0] = 1
    second = model.sample(obs, changed, steps=2, seed=12)
    assert not torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert first.min() >= -1 and first.max() <= 1


def test_seed_and_inference_checkpoint(tmp_path):
    from dataclasses import asdict

    torch.manual_seed(5)
    model = WorldModel(ModelConfig(base=16)).eval()
    path = tmp_path / "model.pt"
    torch.save(
        {"config": asdict(model.cfg), "ema": model.state_dict(), "step": 0}, path
    )
    loaded, checkpoint = load_model(path)
    context = torch.zeros(1, 4, 3, 64, 112)
    actions = torch.zeros(1, 4, 51)
    one = loaded.sample(context, actions, steps=2, seed=42)
    two = loaded.sample(context, actions, steps=2, seed=42)
    assert torch.equal(one, two)
    assert checkpoint["step"] == 0
