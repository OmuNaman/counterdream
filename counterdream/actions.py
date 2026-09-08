"""Public dataset's 51-dimensional keyboard/mouse convention."""

import numpy as np

KEYS = ["w", "a", "s", "d", "space", "ctrl", "shift", "1", "2", "3", "r"]
MOUSE_X = [
    -1000,
    -500,
    -300,
    -200,
    -100,
    -60,
    -30,
    -20,
    -10,
    -4,
    -2,
    0,
    2,
    4,
    10,
    20,
    30,
    60,
    100,
    200,
    300,
    500,
    1000,
]
MOUSE_Y = [-200, -100, -50, -20, -10, -4, -2, 0, 2, 4, 10, 20, 50, 100, 200]


def encode(keys=(), dx=0.0, dy=0.0, fire=False, scope=False):
    if not np.isfinite(dx) or not np.isfinite(dy):
        raise ValueError("Mouse deltas must be finite")
    action = np.zeros(51, dtype=np.float32)
    for i, key in enumerate(KEYS):
        action[i] = float(key in keys)
    action[11] = bool(fire)
    action[12] = bool(scope)
    action[13 + int(np.argmin(np.abs(np.array(MOUSE_X) - dx)))] = 1
    action[36 + int(np.argmin(np.abs(np.array(MOUSE_Y) - dy)))] = 1
    return action


def describe(action):
    pressed = [k.upper() for k, a in zip(KEYS, action[:11]) if a > 0.5]
    if action[11] > 0.5:
        pressed.append("FIRE")
    dx = MOUSE_X[int(np.argmax(action[13:36]))]
    dy = MOUSE_Y[int(np.argmax(action[36:51]))]
    if dx or dy:
        pressed.append(f"LOOK {dx},{dy}")
    return " + ".join(pressed) or "IDLE"
