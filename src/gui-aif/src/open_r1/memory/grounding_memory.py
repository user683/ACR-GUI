import json
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from .embedding import TextEmbeddingEncoder
from .faiss_store import FaissOrNumpyIndex
from .schema import GroundingMemoryItem


@dataclass
class GroundingMemorySearchResult:
    item: GroundingMemoryItem
    similarity: float


class GroundingMemoryBank:
    def __init__(self, encoder: Optional[TextEmbeddingEncoder] = None):
        self.encoder = encoder or TextEmbeddingEncoder()
        self.items: list[GroundingMemoryItem] = []
        self.embeddings = np.zeros((0, self.encoder.dim), dtype=np.float32)
        self.index = FaissOrNumpyIndex(self.encoder.dim)
        self._dirty = False

    def add(self, item: GroundingMemoryItem) -> None:
        embedding = self._embedding_for_item(item)
        item.embedding = embedding.tolist()
        self.items.append(item)
        self.embeddings = np.vstack([self.embeddings, embedding.reshape(1, -1)])
        self._dirty = True
        self._rebuild_index()

    def search(self, instruction: str, top_k: int = 5) -> list[GroundingMemorySearchResult]:
        if not self.items or top_k <= 0:
            return []
        self._rebuild_index()
        query = self.encoder.encode(instruction)
        scores, indices = self.index.search(query, top_k)
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            results.append(GroundingMemorySearchResult(self.items[int(idx)], float(score)))
        return results

    def save(self, jsonl_path: str, npy_path: str) -> None:
        self._rebuild_index()
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in self.items:
                payload = asdict(item)
                payload["embedding"] = None
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        np.save(npy_path, self.embeddings.astype(np.float32))

    def load(self, jsonl_path: str, npy_path: str) -> None:
        items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("bbox") is not None:
                    payload["bbox"] = tuple(payload["bbox"])
                if payload.get("point") is not None:
                    payload["point"] = tuple(payload["point"])
                items.append(GroundingMemoryItem(**payload))

        embeddings = np.load(npy_path).astype(np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(items):
            raise ValueError("embedding matrix does not match JSONL metadata")
        if embeddings.shape[1] != self.encoder.dim:
            raise ValueError(f"embedding dim mismatch: expected {self.encoder.dim}, got {embeddings.shape[1]}")

        for item, embedding in zip(items, embeddings):
            item.embedding = embedding.tolist()
        self.items = items
        self.embeddings = embeddings
        self._dirty = True
        self._rebuild_index()

    def _embedding_for_item(self, item: GroundingMemoryItem) -> np.ndarray:
        if item.embedding is not None:
            embedding = np.asarray(item.embedding, dtype=np.float32)
            if embedding.shape == (self.encoder.dim,):
                return self._normalize(embedding)
        return self.encoder.encode(item.instruction)

    def _rebuild_index(self) -> None:
        if not self._dirty:
            return
        self.embeddings = np.asarray([self._normalize(v) for v in self.embeddings], dtype=np.float32)
        if self.embeddings.size == 0:
            self.embeddings = np.zeros((0, self.encoder.dim), dtype=np.float32)
        self.index.build(self.embeddings)
        self._dirty = False

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
