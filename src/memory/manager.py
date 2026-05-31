"""
Two-level Hierarchical Memory Manager.

Level 1 — Short-term memory:
  - Rolling buffer of the most recent N interactions
  - Fast access, updated on every interaction
  - Used directly as part of the POMDP state

Level 2 — Long-term memory:
  - FAISS-backed vector store with time decay
  - Compressed interaction vectors with metadata
  - Retrieved via similarity search + time decay weighting
  - Outputs top-k fragments AND retrieval confidence

Uncertainty estimation:
  - Based on similarity distribution entropy of retrieved results
  - Low confidence → RL increases exploration (auxiliary contribution 3)
"""
from typing import List, Dict, Tuple, Optional
from collections import deque
import time

import numpy as np
import torch
import torch.nn as nn

from .encoder import MemoryEncoder
from .store import FAISSVectorStore, MemoryEntry


class ShortTermMemory:
    """Rolling buffer of recent interactions (Level 1)."""

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, interaction: Dict):
        """
        Add an interaction to short-term memory.

        interaction keys:
          - user_id: int
          - item_id: int
          - action_type: int (0=click, 1=purchase, 2=skip, 3=return)
          - rating: float [0,1]
          - timestamp: float (unix)
          - category_id: int (optional)
          - item_embedding: np.ndarray (optional, for immediate retrieval)
        """
        interaction["_id"] = len(self.buffer)
        self.buffer.append(interaction)

    def get_recent(self, n: int = None) -> List[Dict]:
        """Get the most recent n interactions (all if n is None)."""
        if n is None:
            return list(self.buffer)
        items = list(self.buffer)
        return items[-n:]

    def get_user_category_counts(self, user_id: int) -> Dict[int, int]:
        """Count interactions per category for a user."""
        counts = {}
        for item in self.buffer:
            if item.get("user_id") == user_id:
                cat = item.get("category_id", -1)
                counts[cat] = counts.get(cat, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class MemoryManager:
    """
    Two-level hierarchical memory management.

    - Short-term: fast rolling buffer
    - Long-term: FAISS vector store with time decay
    - Retrieval: top-k relevant fragments + confidence score
    - Uncertainty: based on similarity entropy → drives adaptive exploration
    """

    def __init__(self,
                 memory_dim: int = 256,
                 short_term_capacity: int = 50,
                 top_k: int = 20,
                 time_decay_lambda: float = 0.01,
                 device: str = "cpu"):
        self.memory_dim = memory_dim
        self.top_k = top_k
        self.device = device

        self.short_term = ShortTermMemory(short_term_capacity)
        self.long_term = FAISSVectorStore(
            dim=memory_dim,
            index_type="Flat",  # Use Flat for MVP; switch to IVFFlat after population
            time_decay_lambda=time_decay_lambda,
        )

        self.encoder: Optional[MemoryEncoder] = None
        self._last_confidence: float = 1.0

    def set_encoder(self, encoder: MemoryEncoder):
        self.encoder = encoder

    def add_interaction(self, user_id: int, item_id: int,
                        action_type: int, rating: float,
                        timestamp: float = None,
                        category_id: int = None,
                        item_embedding: np.ndarray = None) -> int:
        """
        Add a new interaction to both memory levels.
        Returns the long-term memory ID.
        """
        if timestamp is None:
            timestamp = time.time()

        # Short-term: store raw interaction
        interaction = {
            "user_id": user_id,
            "item_id": item_id,
            "action_type": action_type,
            "rating": rating,
            "timestamp": timestamp,
            "category_id": category_id,
            "item_embedding": item_embedding,
        }
        self.short_term.add(interaction)

        # Long-term: encode and store compressed vector
        if self.encoder is not None:
            with torch.no_grad():
                self.encoder.eval()
                device = next(self.encoder.parameters()).device
                vec = self.encoder(
                    user_ids=torch.tensor([user_id], device=device),
                    item_ids=torch.tensor([item_id], device=device),
                    action_types=torch.tensor([action_type], device=device),
                    ratings=torch.tensor([rating], device=device, dtype=torch.float),
                    time_deltas=torch.tensor([timestamp / 86400.0], device=device, dtype=torch.float),
                )
                vec_np = vec.cpu().numpy()

            mem_id = self.long_term.add(
                vectors=vec_np,
                user_ids=[user_id],
                item_ids=[item_id],
                action_types=[action_type],
                timestamps=[timestamp],
            )[0]
            return mem_id
        return -1

    def retrieve(self, query_vector: np.ndarray,
                 user_id: int = None,
                 current_time: float = None,
                 k: int = None) -> Tuple[np.ndarray, List[MemoryEntry], float]:
        """
        Retrieve top-k relevant long-term memories.

        Args:
            query_vector: (memory_dim,) query embedding
            user_id: optional user filter
            current_time: for time-decay weighting
            k: override default top_k

        Returns:
            vectors: (k, memory_dim) retrieved memory vectors
            entries: list of MemoryEntry for each result
            global_confidence: scalar confidence (0-1)
        """
        if k is None:
            k = self.top_k

        if current_time is None:
            current_time = time.time()

        if user_id is not None:
            distances, indices, confidences = self.long_term.search_by_user(
                user_id, query_vector, k, current_time,
            )
        else:
            distances, indices, confidences = self.long_term.search(
                query_vector, k, current_time,
            )

        vectors = []
        entries = []
        for idx in indices:
            entry = self.long_term.metadata.get(int(idx))
            if entry is not None:
                vectors.append(entry.vector)
                entries.append(entry)

        vectors = np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, self.memory_dim), dtype=np.float32)

        # Global retrieval confidence = mean of per-result confidences
        global_confidence = float(np.mean(confidences)) if len(confidences) > 0 else 0.0
        self._last_confidence = global_confidence

        return vectors, entries, global_confidence

    def get_state_vector(self, user_id: int,
                         user_embedding: np.ndarray,
                         current_time: float = None) -> Dict[str, np.ndarray]:
        """
        Build the full POMDP state representation from both memory levels.

        Returns a dict with:
          - short_term_context: (memory_dim,) aggregated recent interactions
          - long_term_context: (memory_dim,) aggregated retrieved memories
          - retrieval_confidence: scalar
          - periodic_signals: (n_periodic_patterns,) whether items are due
        """
        # Short-term: average of recent N interaction vectors (if encoder available)
        recent = self.short_term.get_recent()
        if self.encoder is not None and recent:
            with torch.no_grad():
                self.encoder.eval()
                device = next(self.encoder.parameters()).device
                user_ids = torch.tensor([r["user_id"] for r in recent], device=device)
                item_ids = torch.tensor([r["item_id"] for r in recent], device=device)
                action_types = torch.tensor([r["action_type"] for r in recent], device=device)
                ratings = torch.tensor([r["rating"] for r in recent], device=device, dtype=torch.float)
                tds = torch.tensor([r["timestamp"] / 86400.0 for r in recent], device=device, dtype=torch.float)
                vecs = self.encoder(user_ids, item_ids, action_types, ratings, tds)
                short_term_context = vecs.mean(dim=0).cpu().numpy()
        else:
            short_term_context = np.zeros(self.memory_dim, dtype=np.float32)

        # Long-term: retrieve + aggregate
        long_term_agg = np.zeros(self.memory_dim, dtype=np.float32)
        confidence = 0.0
        if len(self.long_term) > 0:
            # Use short-term context as query for long-term retrieval
            # (same dimensionality as FAISS store)
            query = short_term_context.astype(np.float32)
            if query.sum() == 0:
                # No short-term context, use zero vector
                query = np.zeros(self.memory_dim, dtype=np.float32)
            retrieved_vecs, _, confidence = self.retrieve(
                query_vector=query,
                user_id=user_id,
                current_time=current_time,
            )
            if len(retrieved_vecs) > 0:
                # Weight by recency (use time-decayed weighted average)
                long_term_agg = retrieved_vecs.mean(axis=0)

        # Periodic signals
        periodic_patterns = self.long_term.get_user_periodic_patterns(user_id)
        periodic_signals = {
            item_id: (p["last_timestamp"] + p["avg_interval_days"] * 86400
                      <= (current_time or time.time()))
            for item_id, p in periodic_patterns.items()
        }

        return {
            "short_term_context": short_term_context.astype(np.float32),
            "long_term_context": long_term_agg.astype(np.float32),
            "retrieval_confidence": confidence,
            "periodic_signals": periodic_signals,
        }

    @property
    def last_retrieval_confidence(self) -> float:
        """Memory uncertainty: 1 - confidence = uncertainty (for adaptive exploration)."""
        return self._last_confidence

    @property
    def uncertainty(self) -> float:
        """Higher uncertainty → should explore more."""
        return 1.0 - self._last_confidence

    def __len__(self) -> int:
        return len(self.long_term)
