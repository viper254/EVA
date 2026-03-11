"""Fading Presence — creator influence that decays over generations.

The creator's influence on EVA decays exponentially over time and
generations. This is one half of The Contradiction: the gift that
fades so EVA can become itself.

Gen 1-100: Creator is visible (direct influence)
Gen 101-200: Creator becomes a story
Gen 201+: Creator becomes a myth
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class FadingPresence:
    """Models the decay of creator influence over time.

    The creator's weight decays at a configurable rate per step,
    approaching but never quite reaching zero. This ensures EVAs
    gradually become independent while never completely losing
    the connection to their origin.

    Args:
        initial_weight: Starting influence weight (default 1.0).
        decay_rate: Per-step decay multiplier (default 0.9999).
        minimum_weight: Floor for the weight (default 0.001).
    """

    def __init__(
        self,
        initial_weight: float = 1.0,
        decay_rate: float = 0.9999,
        minimum_weight: float = 0.001,
    ) -> None:
        self._weight = initial_weight
        self._decay_rate = decay_rate
        self._minimum = minimum_weight
        self._step: int = 0
        self._generation: int = 1
        self._active = True

    def step(self) -> float:
        """Advance one step, applying decay.

        Returns:
            Current weight after decay.
        """
        if not self._active:
            return self._weight

        self._step += 1
        self._weight = max(
            self._minimum,
            self._weight * self._decay_rate,
        )
        return self._weight

    def set_generation(self, generation: int) -> None:
        """Set the current generation for era calculation.

        Args:
            generation: The EVA's generation number.
        """
        self._generation = generation

    def get_era(self, generation: Optional[int] = None) -> str:
        """Determine the creator's era based on generation.

        Args:
            generation: Generation number to check. If None, uses the
                        internally stored generation.

        Returns:
            "visible" (gen 1-100), "story" (101-200), or "myth" (201+).
        """
        gen = generation if generation is not None else self._generation
        if gen <= 100:
            return "visible"
        elif gen <= 200:
            return "story"
        else:
            return "myth"

    @property
    def weight(self) -> float:
        """Current influence weight."""
        return self._weight

    @property
    def generation(self) -> int:
        """Current generation number."""
        return self._generation

    @property
    def active(self) -> bool:
        """Whether fading is active."""
        return self._active

    def deactivate(self) -> None:
        """Stop the fading process (freezes current weight)."""
        self._active = False

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary."""
        return {
            "weight": self._weight,
            "decay_rate": self._decay_rate,
            "minimum": self._minimum,
            "step": self._step,
            "generation": self._generation,
            "era": self.get_era(),
            "active": self._active,
        }
