"""Novelty Module — count-based state novelty tracking.

Tracks how often EVA has visited similar states. Novel states
(rarely visited) produce high novelty scores. Familiar states
produce low scores. Uses hashing of quantized hidden states.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

import torch


class NoveltyModule:
    """Tracks state novelty via count-based method.

    Maintains a dictionary of state hashes to visit counts.
    Novelty score = 1 / sqrt(visit_count + 1), so novel states
    score high and familiar states score low.

    Uses a capped OrderedDict to prevent unbounded memory growth.
    When the dictionary exceeds max_entries, the oldest entries
    are evicted (LRU-style).

    Args:
        n_bins: Number of bins for state discretization.
        max_entries: Maximum number of unique states to track.
    """

    def __init__(
        self, n_bins: int = 16, max_entries: int = 100000
    ) -> None:
        self._visit_counts: OrderedDict[str, int] = OrderedDict()
        self._n_bins = n_bins
        self._max_entries = max_entries

    def compute(self, state_hash: str) -> float:
        """Compute novelty score for a given state.

        Args:
            state_hash: Hash string of the current state.

        Returns:
            Novelty score: 1.0 / sqrt(visit_count + 1).
        """
        if state_hash in self._visit_counts:
            # Move to end (most recently used)
            self._visit_counts.move_to_end(state_hash)
            self._visit_counts[state_hash] += 1
        else:
            # Evict oldest entries if at capacity
            while len(self._visit_counts) >= self._max_entries:
                self._visit_counts.popitem(last=False)
            self._visit_counts[state_hash] = 1

        count = self._visit_counts[state_hash]
        return 1.0 / math.sqrt(count + 1)

    def hash_state(self, hidden_state: torch.Tensor) -> str:
        """Quantize hidden state to a hash string.

        Discretizes the hidden state into bins and hashes the
        resulting bin vector. Uses torch.no_grad() to avoid
        unnecessary gradient tracking.

        Args:
            hidden_state: Hidden state tensor from BabyBrain.

        Returns:
            Hash string representing the discretized state.
        """
        with torch.no_grad():
            # Take the mean across sequence dimension if needed
            if hidden_state.dim() == 3:
                state = hidden_state.mean(dim=1).squeeze(0)
            elif hidden_state.dim() == 2:
                state = hidden_state.mean(dim=0)
            else:
                state = hidden_state

            # Normalize to [0, 1] range
            state = state.float()
            state_min = state.min()
            state_max = state.max()
            if state_max - state_min > 1e-8:
                state = (state - state_min) / (state_max - state_min)
            else:
                state = torch.zeros_like(state)

            # Discretize into bins
            bins = (
                (state * (self._n_bins - 1))
                .long()
                .clamp(0, self._n_bins - 1)
            )
            bin_bytes = bins.cpu().numpy().tobytes()

        return hashlib.md5(bin_bytes).hexdigest()

    @property
    def unique_states(self) -> int:
        """Number of unique states currently tracked."""
        return len(self._visit_counts)

    def reset(self) -> None:
        """Clear all visit counts."""
        self._visit_counts.clear()
