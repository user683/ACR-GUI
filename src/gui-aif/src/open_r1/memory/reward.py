import math

import numpy as np

from .grounding_memory import GroundingMemoryBank


def compute_memory_reward(
    instruction: str,
    pred_point: tuple[float, float],
    memory_bank: GroundingMemoryBank,
    top_k: int = 5,
    sigma: float = 0.15,
) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    results = memory_bank.search(instruction, top_k=top_k)
    if not results:
        return 0.0

    similarities = np.asarray([r.similarity for r in results], dtype=np.float32)
    similarities = similarities - similarities.max()
    weights = np.exp(similarities)
    weights = weights / weights.sum()

    pred = np.asarray(pred_point, dtype=np.float32)
    reward = 0.0
    for weight, result in zip(weights, results):
        mem_point = np.asarray(result.item.point, dtype=np.float32)
        dist_sq = float(np.sum((pred - mem_point) ** 2))
        reward += float(weight) * math.exp(-dist_sq / (2.0 * sigma * sigma))
    return float(reward)
