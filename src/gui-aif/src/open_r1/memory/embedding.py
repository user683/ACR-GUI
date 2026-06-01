import hashlib
import re
from typing import Optional

import numpy as np


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


class TextEmbeddingEncoder:
    """Text encoder with sentence-transformers support and deterministic fallback."""

    def __init__(self, model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2", dim: int = 384):
        self.model_name = model_name
        self.dim = dim
        self.model = self._try_load_sentence_transformer(model_name) if model_name else None
        if self.model is not None:
            inferred_dim = self.model.get_sentence_embedding_dimension()
            if inferred_dim:
                self.dim = int(inferred_dim)

    @staticmethod
    def _try_load_sentence_transformer(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(model_name)
        except Exception:
            return None

    def encode(self, text: str) -> np.ndarray:
        if self.model is not None:
            vector = self.model.encode(text or "", normalize_embeddings=True)
            return np.asarray(vector, dtype=np.float32)
        return hash_text_embedding(text or "", dim=self.dim)


def hash_text_embedding(text: str, dim: int = 384) -> np.ndarray:
    if dim <= 0:
        raise ValueError("dim must be positive")

    vector = np.zeros(dim, dtype=np.float32)
    tokens = [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
    if not tokens:
        return vector

    features = tokens + [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big", signed=False)
        vector[value % dim] += 1.0 if ((value >> 8) & 1) == 0 else -1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector
