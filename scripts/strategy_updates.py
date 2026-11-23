"""Optional exploration and update schedule for the strategic policy."""
from dataclasses import dataclass
import hashlib
import statistics

import numpy as np


@dataclass(frozen=True)
class UpdateConfig:
    temperature: float
    learning_rate: float
    normalize_advantage: bool
    max_score_gap_change: float | None


def update_config(schedule, round_number, warmup_rounds):
    if schedule == "standard":
        return UpdateConfig(1.0, 0.6, False, None)
    progress = min(1.0, (round_number - 1) / max(1, warmup_rounds - 1))
    return UpdateConfig(
        temperature=1.4 - 0.5 * progress,
        learning_rate=0.9 - 0.6 * progress,
        normalize_advantage=True,
        max_score_gap_change=0.35 - 0.2 * progress,
    )


def training_rng(seed, rollout_index):
    # The first rollout preserves the original sampler for existing checkpoints.
    key = seed if rollout_index == 0 else f"{seed}/rollout/{rollout_index}"
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def update_weights(weights, samples, settings):
    """One batch policy-gradient update and an upper bound on its score-gap movement."""
    if not samples or (settings.normalize_advantage and len(samples) < 2):
        return weights.copy(), {"optimizer_steps": 0, "advantage_std": None, "step_l2": 0.0,
                                "gap_change_bound": 0.0, "step_limited": False}
    rewards = [reward for reward, _, _ in samples]
    center = statistics.mean(rewards) if len(rewards) > 1 else .05
    spread = statistics.pstdev(rewards)
    advantage = np.asarray(rewards) - center
    if settings.normalize_advantage:
        advantage = np.clip(advantage / max(spread, .005), -2.5, 2.5)
    direction = np.mean([float(value) * gradient for value, (_, gradient, _) in zip(advantage, samples, strict=True)], axis=0)
    proposed = np.asarray(np.clip(weights + settings.learning_rate * direction, -2.0, 2.0), dtype=weights.dtype)
    step = proposed - weights
    pair_bound = max(bound for _, _, bound in samples)
    movement = float(np.linalg.norm(step)) * pair_bound
    limit = settings.max_score_gap_change
    limited = limit is not None and movement > limit
    updated = weights + step * (limit / movement) if limited else proposed
    size = float(np.linalg.norm(updated - weights))
    return updated, {"optimizer_steps": 1, "advantage_std": spread, "step_l2": size,
                     "gap_change_bound": size * pair_bound, "step_limited": limited}
