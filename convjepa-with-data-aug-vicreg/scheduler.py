"""Cosine learning-rate schedule with linear warmup."""

from __future__ import annotations

import math
from typing import List


def cosine_warmup_schedule(base_lr: float, min_lr: float,
                           total_steps: int, warmup_steps: int,
                           start_lr: float = 0.0) -> List[float]:
    """
    Piecewise LR schedule:
      - Linear warmup from `start_lr` to `base_lr` over `warmup_steps`.
      - Cosine decay from `base_lr` to `min_lr` over the remaining steps.
    Returns a list of length `total_steps`.
    """
    warmup_steps = max(0, min(warmup_steps, total_steps))
    schedule: List[float] = []

    for step in range(warmup_steps):
        alpha = (step + 1) / max(1, warmup_steps)
        schedule.append(start_lr + (base_lr - start_lr) * alpha)

    remaining = total_steps - warmup_steps
    for step in range(remaining):
        # t ∈ [0, 1]
        t = step / max(1, remaining - 1) if remaining > 1 else 0.0
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        schedule.append(min_lr + (base_lr - min_lr) * cos)

    assert len(schedule) == total_steps
    return schedule


class CosineWarmupLR:
    """Tiny LR scheduler that steps a precomputed schedule once per training step."""

    def __init__(self, optimizer, base_lr: float, min_lr: float,
                 total_steps: int, warmup_steps: int):
        self.optimizer = optimizer
        self.schedule = cosine_warmup_schedule(base_lr, min_lr, total_steps, warmup_steps)
        self.step_idx = 0

    def step(self):
        if self.step_idx < len(self.schedule):
            lr = self.schedule[self.step_idx]
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
            self.step_idx += 1

    def get_last_lr(self) -> float:
        i = min(self.step_idx, len(self.schedule) - 1)
        return float(self.schedule[max(0, i)])

    def state_dict(self) -> dict:
        return {"step_idx": self.step_idx, "schedule": list(self.schedule)}

    def load_state_dict(self, state: dict):
        self.step_idx = int(state["step_idx"])
        self.schedule = list(state["schedule"])
