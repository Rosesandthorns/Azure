import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List

import chromadb
import numpy as np

from memory_service import MemoryEntry


@dataclass
class RetrievalResult:
    memory: MemoryEntry
    score: float


class LocalEmbeddingFunction:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        tokens = [t for t in text.lower().split() if t]
        vector = np.zeros(self.dim, dtype=np.float32)
        if not tokens:
            return vector.tolist()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.tolist()


class RetrievalEngine:
    def __init__(self, chroma_path: str, embedding_dim: int) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        os.makedirs(chroma_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection("memories")
        self.embedder = LocalEmbeddingFunction(embedding_dim)

    def upsert_memory(self, memory: MemoryEntry) -> None:
        embedding = self.embedder.embed(memory.content)
        metadata = {
            "type": memory.type,
            "confidence": memory.confidence,
            "importance": memory.importance,
            "timestamp": memory.timestamp,
            "user_id": memory.user_id,
        }
        self.collection.upsert(
            ids=[memory.id],
            documents=[memory.content],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def delete_memory(self, memory_id: str) -> None:
        self.collection.delete(ids=[memory_id])

    def query(self, query_text: str, top_k: int = 5) -> List[str]:
        embedding = self.embedder.embed(query_text)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        return list(results.get("ids", [[]])[0])
