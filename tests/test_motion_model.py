import torch

from counterdream.motion_model import MotionPredictor, warp


def test_zero_flow_preserves_picture_and_translation_moves_pixels():
    frame = (
        torch.arange(32, dtype=torch.float32).reshape(1, 1, 1, 32).expand(1, 3, 16, 32)
        / 32
    )
    flow = torch.zeros(1, 2, 16, 32)
    torch.testing.assert_close(warp(frame, flow), frame, rtol=1e-5, atol=1e-6)
    flow[:, 0] = 2
    torch.testing.assert_close(
        warp(frame, flow)[:, :, :, :-2], frame[:, :, :, 2:], rtol=1e-5, atol=1e-6
    )


def test_motion_initial_identity_and_gradients():
    torch.set_num_threads(2)
    model = MotionPredictor()
    context = torch.rand(2, 2, 3, 88, 160) * 2 - 1
    actions = torch.rand(2, 2, 51)
    predicted, flow, residual = model(context, actions)
    assert flow.count_nonzero() == 0 and residual.count_nonzero() == 0
    torch.testing.assert_close(predicted, context[:, -1], atol=3e-5, rtol=3e-5)
    (predicted - context[:, 0]).square().mean().backward()
    assert model.flow.weight.grad.abs().sum() > 0
