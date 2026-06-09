import math
import re
from typing import Optional

import numpy as np

from .grounding_memory import GroundingMemoryBank

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def infer_layout_zone(point: tuple[float, float]) -> str:
    x, y = point
    if y < 0.18:
        return "top_bar"
    if y > 0.82:
        return "bottom_bar"
    if x < 0.18:
        return "left_sidebar"
    if x > 0.82:
        return "right_sidebar"
    return "content"


def bbox_to_anchor(bbox: tuple[float, float, float, float]) -> tuple[tuple[float, float], tuple[float, float], str]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    size = (max(0.0, x2 - x1), max(0.0, y2 - y1))
    return point, size, infer_layout_zone(point)


def normalize_bbox(
    bbox: tuple[float, float, float, float],
    image_size: Optional[tuple[int, int]] = None,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if image_size is None or (max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5):
        return (x1, y1, x2, y2)

    width, height = image_size
    if width <= 0 or height <= 0:
        return (x1, y1, x2, y2)
    return (x1 / width, y1 / height, x2 / width, y2 / height)


def _softmax(scores: list[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    values = values - values.max()
    weights = np.exp(values)
    denom = weights.sum()
    if denom <= 0:
        return np.zeros_like(weights)
    return weights / denom


def _text_sim(left: Optional[str], right: Optional[str]) -> float:
    left_tokens = {m.group(0).lower() for m in _TOKEN_RE.finditer(left or "")}
    right_tokens = {m.group(0).lower() for m in _TOKEN_RE.finditer(right or "")}
    if not left_tokens and not right_tokens:
        return 0.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _exact_match_score(left: Optional[str], right: Optional[str]) -> float:
    if not left or not right:
        return 0.0
    return 1.0 if left.strip().lower() == right.strip().lower() else 0.0


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


def compute_anchor_reward(
    instruction: str,
    pred_bbox: tuple[float, float, float, float],
    memory_bank: GroundingMemoryBank,
    *,
    image_size: Optional[tuple[int, int]] = None,
    context_text: Optional[str] = None,
    element_type: Optional[str] = None,
    domain: Optional[str] = None,
    top_k: int = 5,
    tau_sim: float = 0.35,
    tau_conf: float = 0.8,
    eta_point: float = 1.0,
    eta_size: float = 0.5,
    eta_zone: float = 0.25,
    beta_text: float = 0.2,
    beta_element: float = 0.2,
    beta_layout: float = 0.15,
    beta_domain: float = 0.2,
    sigma_point: float = 0.15,
    sigma_size: float = 0.15,
) -> float:
    if sigma_point <= 0 or sigma_size <= 0:
        raise ValueError("sigma_point and sigma_size must be positive")

    pred_bbox = normalize_bbox(pred_bbox, image_size=image_size)
    pred_point, pred_size, pred_zone = bbox_to_anchor(pred_bbox)
    results = memory_bank.search(instruction, top_k=top_k)
    if not results:
        return 0.0

    max_semantic_sim = max(float(result.similarity) for result in results)
    max_conf = max(float(result.item.confidence) for result in results)
    if max_semantic_sim <= tau_sim or max_conf <= tau_conf:
        return 0.0

    scores = []
    for result in results:
        item = result.item
        item_zone = item.layout_role or infer_layout_zone(item.point)
        score = float(result.similarity)
        score += beta_text * _text_sim(context_text, item.context_text)
        score += beta_element * _exact_match_score(element_type, item.element_type)
        score += beta_layout * (1.0 if pred_zone == item_zone else 0.0)
        score += beta_domain * max(_text_sim(domain, item.domain), _exact_match_score(domain, item.domain))
        scores.append(score)

    weights = _softmax(scores)
    point_reward = 0.0
    size_reward = 0.0
    zone_reward = 0.0

    pred_point_arr = np.asarray(pred_point, dtype=np.float32)
    pred_size_arr = np.asarray(pred_size, dtype=np.float32)
    for weight, result in zip(weights, results):
        item = result.item
        item_point = np.asarray(item.point, dtype=np.float32)
        item_size = np.asarray(item.size, dtype=np.float32)
        item_zone = item.layout_role or infer_layout_zone(item.point)

        point_dist_sq = float(np.sum((pred_point_arr - item_point) ** 2))
        size_dist_sq = float(np.sum((pred_size_arr - item_size) ** 2))
        local_sigma_point = max(float(sigma_point), float(max(item.size)) * 0.5)
        local_sigma_size = max(float(sigma_size), float(max(item.size)) * 0.5)

        point_reward += float(weight) * math.exp(-point_dist_sq / (2.0 * local_sigma_point * local_sigma_point))
        size_reward += float(weight) * math.exp(-size_dist_sq / (2.0 * local_sigma_size * local_sigma_size))
        zone_reward += float(weight) * (1.0 if pred_zone == item_zone else 0.0)

    return float(eta_point * point_reward + eta_size * size_reward + eta_zone * zone_reward)
