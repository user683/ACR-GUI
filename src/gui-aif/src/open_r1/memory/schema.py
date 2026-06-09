from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GroundingMemoryItem:
    id: str
    domain: str
    instruction: str
    embedding: Optional[list[float]]
    bbox: tuple[float, float, float, float]
    point: tuple[float, float]
    size: Optional[tuple[float, float]] = None
    context_text: Optional[str] = None
    element_type: Optional[str] = None
    layout_role: Optional[str] = None
    confidence: float = 0.0
    success: bool = False
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.bbox = tuple(float(v) for v in self.bbox)
        self.point = tuple(float(v) for v in self.point)
        if self.size is None:
            self.size = (
                max(0.0, float(self.bbox[2]) - float(self.bbox[0])),
                max(0.0, float(self.bbox[3]) - float(self.bbox[1])),
            )
        else:
            self.size = tuple(float(v) for v in self.size)
