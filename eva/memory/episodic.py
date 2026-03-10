"""Episodic Memory — importance-weighted circular buffer.

Stores experiences as episodes with emotional importance weighting.
Supports similarity-based recall and rest-period consolidation
(merging similar, low-importance episodes).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class Episode:
    """A single episodic memory entry.

    Attributes:
        state_embedding: Hidden state at time of episode.
        action: Action taken (token ID).
        outcome: Outcome observed (token ID).
        surprise: Prediction error / surprise level.
        emotional_importance: How emotionally significant this was.
        source_tag: Who was involved ("self", "human", "scaffold", "ancestor").
        timestamp: Step number when this occurred.
    """

    state_embedding: torch.Tensor
    action: int
    outcome: int
    surprise: float
    emotional_importance: float
    source_tag: str
    timestamp: int


class EpisodicMemory:
    """Fixed-size circular buffer for episodic memories.

    When full, evicts the entry with lowest emotional_importance
    using a min-heap for O(log n) eviction instead of O(n) scan.
    Supports batched cosine-similarity retrieval and rest-period
    consolidation.

    Args:
        max_size: Maximum number of episodes to store.
    """

    def __init__(
        self, max_size: int = 10000, capacity: int | None = None
    ) -> None:
        self._max_size = capacity if capacity is not None else max_size
        self._buffer: list[Episode] = []
        # Min-heap of (importance, index) for O(log n) eviction
        self._importance_heap: list[tuple[float, int]] = []
        self._next_idx: int = 0

    def store(self, episode: Episode) -> None:
        """Store a new episode, evicting lowest importance if full.

        Uses a min-heap for efficient O(log n) eviction.

        Args:
            episode: The episode to store.
        """
        if len(self._buffer) >= self._max_size:
            # Pop lowest importance from heap
            while self._importance_heap:
                min_imp, min_idx = self._importance_heap[0]
                # Validate heap entry is still current
                if (
                    min_idx < len(self._buffer)
                    and self._buffer[min_idx].emotional_importance
                    == min_imp
                ):
                    break
                heapq.heappop(self._importance_heap)

            if not self._importance_heap:
                # Rebuild heap if all entries were stale
                self._rebuild_heap()

            min_imp, min_idx = self._importance_heap[0]
            # Only evict if new episode is more important
            if episode.emotional_importance > min_imp:
                heapq.heappop(self._importance_heap)
                self._buffer[min_idx] = episode
                heapq.heappush(
                    self._importance_heap,
                    (episode.emotional_importance, min_idx),
                )
        else:
            idx = len(self._buffer)
            self._buffer.append(episode)
            heapq.heappush(
                self._importance_heap,
                (episode.emotional_importance, idx),
            )

    def _rebuild_heap(self) -> None:
        """Rebuild the importance heap from the current buffer."""
        self._importance_heap = [
            (ep.emotional_importance, i)
            for i, ep in enumerate(self._buffer)
        ]
        heapq.heapify(self._importance_heap)

    def recall(
        self, query_embedding: torch.Tensor, k: int = 5
    ) -> list[Episode]:
        """Retrieve most similar episodes by cosine similarity.

        Uses batched matrix operations for efficient similarity
        computation instead of per-episode iteration.

        Args:
            query_embedding: Query vector to match against.
            k: Number of episodes to retrieve.

        Returns:
            List of k most similar episodes.
        """
        if not self._buffer:
            return []

        k = min(k, len(self._buffer))

        # Flatten query
        query = query_embedding.float().flatten()

        # Determine common dimension
        emb_dim = query.shape[0]
        for ep in self._buffer:
            ep_dim = ep.state_embedding.flatten().shape[0]
            emb_dim = min(emb_dim, ep_dim)

        # Batch similarity computation
        query_vec = query[:emb_dim].unsqueeze(0)  # (1, dim)
        embeddings = torch.stack(
            [ep.state_embedding.float().flatten()[:emb_dim]
             for ep in self._buffer]
        )  # (n, dim)

        # Batched cosine similarity
        similarities = F.cosine_similarity(
            query_vec, embeddings, dim=1
        )  # (n,)

        # Top-k indices
        topk_vals, topk_indices = torch.topk(similarities, k)
        return [self._buffer[idx.item()] for idx in topk_indices]

    def consolidate(self, batch_size: int = 256) -> int:
        """Consolidate memories during rest periods.

        Uses batched similarity computation for efficiency.
        Finds pairs of low-importance episodes with cosine
        similarity > 0.9 and merges them.

        Args:
            batch_size: Number of episodes to process at once.

        Returns:
            Number of episodes merged.
        """
        if len(self._buffer) < 2:
            return 0

        # Filter to low-importance episodes only
        low_imp_indices = [
            i for i, ep in enumerate(self._buffer)
            if ep.emotional_importance <= 0.5
        ]

        if len(low_imp_indices) < 2:
            return 0

        merged_count = 0
        to_remove: set[int] = set()

        # Process in batches for memory efficiency
        for batch_start in range(0, len(low_imp_indices), batch_size):
            batch_indices = low_imp_indices[
                batch_start:batch_start + batch_size
            ]
            # Remove already-merged indices
            batch_indices = [
                i for i in batch_indices if i not in to_remove
            ]
            if len(batch_indices) < 2:
                continue

            # Determine common embedding dimension
            emb_dim = min(
                self._buffer[i].state_embedding.flatten().shape[0]
                for i in batch_indices
            )

            # Stack embeddings for batch computation
            embeddings = torch.stack([
                self._buffer[i].state_embedding.float().flatten()[:emb_dim]
                for i in batch_indices
            ])  # (batch, dim)

            # Compute pairwise cosine similarity matrix
            norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
            normalized = embeddings / norms
            sim_matrix = normalized @ normalized.T  # (batch, batch)

            # Find pairs with similarity > 0.9 (upper triangle only)
            for bi in range(len(batch_indices)):
                if batch_indices[bi] in to_remove:
                    continue
                for bj in range(bi + 1, len(batch_indices)):
                    if batch_indices[bj] in to_remove:
                        continue
                    if sim_matrix[bi, bj].item() > 0.9:
                        idx_i = batch_indices[bi]
                        idx_j = batch_indices[bj]
                        ep_i = self._buffer[idx_i]
                        ep_j = self._buffer[idx_j]

                        # Merge
                        merged_emb = (
                            ep_i.state_embedding + ep_j.state_embedding
                        ) / 2.0
                        merged_importance = (
                            ep_i.emotional_importance
                            + ep_j.emotional_importance
                        )
                        newer_ts = max(ep_i.timestamp, ep_j.timestamp)
                        newer_ep = (
                            ep_i if ep_i.timestamp > ep_j.timestamp
                            else ep_j
                        )

                        self._buffer[idx_i] = Episode(
                            state_embedding=merged_emb,
                            action=newer_ep.action,
                            outcome=newer_ep.outcome,
                            surprise=(
                                ep_i.surprise + ep_j.surprise
                            ) / 2.0,
                            emotional_importance=merged_importance,
                            source_tag=ep_i.source_tag,
                            timestamp=newer_ts,
                        )
                        to_remove.add(idx_j)
                        merged_count += 1

        # Remove merged episodes (in reverse order to preserve indices)
        for idx in sorted(to_remove, reverse=True):
            self._buffer.pop(idx)

        # Rebuild heap after consolidation
        if merged_count > 0:
            self._rebuild_heap()

        return merged_count

    def size(self) -> int:
        """Current number of stored episodes."""
        return len(self._buffer)

    def clear(self) -> None:
        """Empty the buffer. Used for portage — memories don't travel."""
        self._buffer.clear()
