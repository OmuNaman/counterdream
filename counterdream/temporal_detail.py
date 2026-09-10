"""Retain aligned detail from a generated frame, with confidence gating."""

import cv2
import numpy as np
import torch


@torch.inference_mode()
def restore_detail(previous, predicted, strength=0.5):
    if not 0 <= strength <= 1 or previous.shape != predicted.shape:
        raise ValueError("Invalid detail restoration input")
    if previous.shape[0] != 1:
        raise ValueError("Temporal detail currently supports one viewer")
    before = previous[0].float().add(1).mul(0.5).permute(1, 2, 0).cpu().numpy()
    after = predicted[0].float().add(1).mul(0.5).permute(1, 2, 0).cpu().numpy()
    gray_before = cv2.cvtColor(
        (before * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY
    )
    gray_after = cv2.cvtColor(
        (after * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY
    )
    # Backward flow maps each proposed output pixel into the previous frame.
    flow = cv2.calcOpticalFlowFarneback(
        gray_after, gray_before, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    h, w = gray_after.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = xx + flow[:, :, 0], yy + flow[:, :, 1]
    aligned = cv2.remap(
        before, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    valid = ((mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)).astype(np.float32)
    low_before = cv2.GaussianBlur(aligned, (0, 0), 1.0)
    low_after = cv2.GaussianBlur(after, (0, 0), 1.0)
    disagreement = np.mean(np.abs(low_before - low_after), axis=2)
    confidence = np.exp(-disagreement / 0.06) * valid
    old_detail = aligned - low_before
    new_detail = after - low_after
    # Restore only missing high frequencies. Low-frequency geometry stays generated.
    delta = np.clip(old_detail - new_detail, -0.08, 0.08)
    result = np.clip(after + strength * confidence[:, :, None] * delta, 0, 1)
    return (
        torch.from_numpy(result)
        .to(predicted.device)
        .permute(2, 0, 1)[None]
        .mul(2)
        .sub(1)
    )
