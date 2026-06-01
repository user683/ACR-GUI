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
    ocr_text: Optional[str] = None
    element_type: Optional[str] = None
    layout_role: Optional[str] = None
    confidence: float = 0.0
    success: bool = False
    extra: dict = field(default_factory=dict)
