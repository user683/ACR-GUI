from collections import OrderedDict, deque
from typing import Sequence

from .schema import GroundingMemoryItem


def select_top_n(
    items: Sequence[GroundingMemoryItem],
    capacity: int,
    per_domain: bool = True,
) -> list[GroundingMemoryItem]:
    """Select up to ``capacity`` memory items, ranked by confidence.

    With ``per_domain=True`` the selection is balanced across domains via
    round-robin over per-domain confidence-sorted queues: every domain keeps its
    strongest anchors before any domain contributes a second one. This prevents a
    large / high-confidence domain from evicting an entire older domain (which
    would reintroduce catastrophic forgetting). Leftover capacity is naturally
    absorbed by domains that still have items.
    """
    items = list(items)
    if capacity is None or capacity <= 0 or len(items) <= capacity:
        return items

    if not per_domain:
        return sorted(items, key=lambda it: float(it.confidence), reverse=True)[:capacity]

    # Group by domain, preserving first-seen order for determinism.
    buckets: "OrderedDict[str, list[GroundingMemoryItem]]" = OrderedDict()
    for it in items:
        buckets.setdefault(it.domain or "unknown", []).append(it)
    queues = OrderedDict(
        (domain, deque(sorted(bucket, key=lambda it: float(it.confidence), reverse=True)))
        for domain, bucket in buckets.items()
    )

    selected: list[GroundingMemoryItem] = []
    while len(selected) < capacity and any(queues.values()):
        for queue in queues.values():
            if not queue:
                continue
            selected.append(queue.popleft())
            if len(selected) >= capacity:
                break
    return selected
