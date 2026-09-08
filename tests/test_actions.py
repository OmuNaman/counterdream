import pytest
from counterdream.actions import encode, describe


def test_real_action_layout():
    idle = encode()
    assert idle.shape == (51,)
    assert idle[24] == 1 and idle[43] == 1 and idle.sum() == 2
    action = encode(["w", "a", "r"], dx=-60, dy=50, fire=True)
    assert action[[0, 1, 10, 11]].sum() == 4
    assert action[13 + 5] == 1 and action[36 + 12] == 1
    assert action[13:36].sum() == 1 and action[36:51].sum() == 1
    assert "FIRE" in describe(action)


def test_bad_mouse_rejected():
    with pytest.raises(ValueError):
        encode(dx=float("nan"))
