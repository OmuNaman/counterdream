"""Twenty seconds of normal controls, with firing early in the sequence."""


def play_controls(frame):
    if frame < 48:
        return "FORWARD / W", {"keys": ["w"]}
    if frame < 80:
        return "BACKWARD / S", {"keys": ["s"]}
    if frame < 112:
        return "FIRE HELD", {"fire": True}
    if frame < 136:
        return "LOOK LEFT", {"dx": -10}
    if frame < 160:
        return "LOOK RIGHT", {"dx": 10}
    if frame < 192:
        return "FORWARD + FIRE", {"keys": ["w"], "fire": True}
    if frame < 208:
        return "RELOAD", {"keys": ["r"]}
    if frame < 232:
        return "STRAFE LEFT", {"keys": ["a"]}
    if frame < 256:
        return "STRAFE RIGHT", {"keys": ["d"]}
    if frame < 272:
        return "JUMP", {"keys": ["space"] if frame == 256 else []}
    if frame < 304:
        return "FIRE BURSTS", {"fire": frame % 8 < 4}
    return "BACKWARD / S", {"keys": ["s"]}
