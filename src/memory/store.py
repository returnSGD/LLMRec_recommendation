"""
FAISS-backed long-term memory store with time decay.

Each memory entry is stored as:
  - vector: compressed interaction embedding (memory_dim)
  - metadata: user_id, item_id, timestamp, action_type
  - timestamp: for time-decay weighted retrieval

Time decay: weight = exp(-λ · Δt) where Δt is days since the event.
"""
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import faiss


@dataclass
class MemoryEntry:
    """A single memory fragment stored in FAISS."""
    vector: np.ndarray  # (memory_dim,)
    user_id: int
    item_id: int
    action_type: int  # 0=click, 1=purchase, 2=skip, 3=return
    timestamp: float  # unix timestamp
    memory_id: int  # unique ID in the store

    @property
    def days_since_epoch(self) -> float:
        return self.timestamp / 86400.0


class FAISSVectorStore:
    """
    FAISS-backed vector store for long-term memory.
    Supports IVF-based approximate nearest neighbor search and time-decay
    weighted retrieval.
    """

    def __init__(self, dim: int = 256, index_type: str = "IVFFlat",
                 nlist: int = 100, time_decay_lambda: float = 0.01):
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.time_decay_lambda = time_decay_lambda

        # Create FAISS index
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(dim)  # inner product for normalized vecs
        elif index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = min(10, nlist)  # number of clusters to search
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.metadata: Dict[int, MemoryEntry] = {}  # memory_id → MemoryEntry
        self.next_id: int = 0
        self.is_trained: bool = (index_type == "Flat")  # Flat needs no training

    def add(self, vectors: np.ndarray, user_ids: List[int],
            item_ids: List[int], action_types: List[int],
            timestamps: List[float]) -> List[int]:
        """
        Add new memory entries to the store.

        Args:
            vectors: (N, dim) L2-normalized memory vectors
            user_ids: list of user IDs
            item_ids: list of item IDs
            action_types: list of action type codes
            timestamps: list of unix timestamps

        Returns:
            List of assigned memory_ids
        """
        N = vectors.shape[0]

        # Train IVF if needed
        if not self.is_trained and self.index_type != "Flat":
            if self.index.ntotal + N >= self.nlist * 10:
                # Use existing + new vectors for training
                train_data = vectors.astype(np.float32)
                if self.index.ntotal > 0:
                    existing = self._get_all_vectors()
                    train_data = np.vstack([existing, train_data])
                self.index.train(train_data)
                self.is_trained = True

        self.index.add(vectors.astype(np.float32))

        ids = []
        for i in range(N):
            mid = self.next_id
            ids.append(mid)
            self.metadata[mid] = MemoryEntry(
                vector=vectors[i],
                user_id=user_ids[i],
                item_id=item_ids[i],
                action_type=action_types[i],
                timestamp=timestamps[i],
                memory_id=mid,
            )
            self.next_id += 1

        return ids

    def search(self, query: np.ndarray, k: int = 20,
               current_time: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve top-k memory fragments by similarity × time_decay.

        Args:
            query: (D,) or (1, D) query vector
            k: number of results
            current_time: unix timestamp for time-decay weighting

        Returns:
            distances: (k,) similarity scores (time-decayed)
            indices: (k,) FAISS indices
            confidences: (k,) per-result confidence scores
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Retrieve more than k to apply time decay and re-rank
        search_k = min(k * 3, self.index.ntotal)
        if search_k == 0:
            return (np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0))

        distances, indices = self.index.search(query.astype(np.float32), search_k)

        # Squeeze batch dimension
        distances = distances[0]
        indices = indices[0]

        # Filter invalid indices
        valid_mask = indices != -1
        distances = distances[valid_mask]
        indices = indices[valid_mask]

        if len(distances) == 0:
            return (np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0))

        # Apply time decay if current_time is provided
        if current_time is not None:
            time_weights = np.array([
                self._time_weight(idx, current_time)
                for idx in indices
            ])
            distances = distances * time_weights

        # Re-rank by time-decayed similarity
        rerank_order = np.argsort(distances)[::-1][:k]
        distances = distances[rerank_order]
        indices = indices[rerank_order]

        # Compute per-result confidence based on similarity score distribution
        confidences = self._compute_confidence(distances)

        return distances, indices, confidences

    def search_by_user(self, user_id: int, query: np.ndarray, k: int = 20,
                       current_time: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve top-k memories filtered to a specific user.
        Uses FAISS search + post-filter by user_id.
        """
        # Search wider, then filter
        search_k = min(k * 50, self.index.ntotal)
        if search_k == 0:
            return (np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0))

        if query.ndim == 1:
            query = query.reshape(1, -1)

        distances, indices = self.index.search(query.astype(np.float32), search_k)
        distances = distances[0]
        indices = indices[0]

        # Filter by user
        valid_mask = np.array([
            idx != -1 and self.metadata.get(idx) and self.metadata[idx].user_id == user_id
            for idx in indices
        ])
        distances = distances[valid_mask]
        indices = indices[valid_mask]

        if len(distances) == 0:
            return (np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0))

        # Time decay
        if current_time is not None:
            time_weights = np.array([self._time_weight(i, current_time) for i in indices])
            distances = distances * time_weights

        rerank_order = np.argsort(distances)[::-1][:k]
        confidences = self._compute_confidence(distances[rerank_order])

        return distances[rerank_order], indices[rerank_order], confidences

    def _time_weight(self, faiss_idx: int, current_time: float) -> float:
        """Compute time-decay weight for a memory entry."""
        entry = self.metadata.get(faiss_idx)
        if entry is None:
            return 1.0
        delta_days = (current_time - entry.timestamp) / 86400.0
        return np.exp(-self.time_decay_lambda * max(delta_days, 0))

    def _compute_confidence(self, similarities: np.ndarray) -> np.ndarray:
        """
        Estimate retrieval confidence from the entropy of the top-k
        similarity distribution. High entropy = low confidence.
        """
        if len(similarities) <= 1:
            return np.ones(len(similarities))

        # Softmax normalization
        sims = np.clip(similarities, -50, 50)
        probs = np.exp(sims - sims.max())
        probs = probs / (probs.sum() + 1e-8)

        # Entropy H = -Σ p log p, normalized by log(k)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(len(sims))
        normalized_entropy = entropy / (max_entropy + 1e-8)

        # Confidence = 1 - normalized_entropy
        # Also scale by absolute similarity (low sim → low confidence)
        confidence = 1.0 - normalized_entropy
        confidence = confidence * np.clip(sims / (sims.max() + 1e-8), 0.3, 1.0)

        return confidence.astype(np.float32)

    def _get_all_vectors(self) -> np.ndarray:
        """Retrieve all stored vectors (for IVF training)."""
        if self.index.ntotal == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = np.zeros((self.index.ntotal, self.dim), dtype=np.float32)
        self.index.reconstruct_n(0, self.index.ntotal, vectors)
        return vectors

    def get_user_history(self, user_id: int) -> List[MemoryEntry]:
        """Get all memory entries for a specific user (by scanning metadata)."""
        return [e for e in self.metadata.values() if e.user_id == user_id]

    def get_user_periodic_patterns(self, user_id: int,
                                   min_occurrences: int = 2) -> Dict[int, Dict]:
        """
        Detect periodic purchase patterns for a user.
        Returns dict: item_id → {avg_interval_days, last_timestamp, count}
        """
        entries = sorted(
            [e for e in self.metadata.values()
             if e.user_id == user_id and e.action_type == 1],  # purchase
            key=lambda e: e.timestamp,
        )

        # Group by item_id
        from collections import defaultdict
        item_timestamps = defaultdict(list)
        for e in entries:
            item_timestamps[e.item_id].append(e.timestamp)

        patterns = {}
        for item_id, tss in item_timestamps.items():
            if len(tss) < min_occurrences:
                continue
            intervals = []
            for i in range(1, len(tss)):
                interval = (tss[i] - tss[i - 1]) / 86400.0
                if 1 < interval < 365:
                    intervals.append(interval)
            if len(intervals) >= 1:
                avg_interval = np.mean(intervals)
                cv = np.std(intervals) / (avg_interval + 1e-8)
                if cv < 0.5:  # regular pattern (from data exploration: ~29% of users)
                    patterns[item_id] = {
                        "avg_interval_days": avg_interval,
                        "last_timestamp": tss[-1],
                        "count": len(tss),
                        "cv": cv,
                    }
        return patterns

    def __len__(self) -> int:
        return self.index.ntotal
