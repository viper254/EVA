"""Information Gain Module — measures how much the model changed.

Computes the difference in parameter statistics before and after
a learning step. Lightweight: uses mean/std per layer, not full
parameter copies.
"""

from __future__ import annotations

from typing import Optional

from eva.core.baby_brain import BabyBrain


class InformationGainModule:
    """Measures how much the model changed from a learning step.

    Stores a lightweight parameter snapshot (mean/std per layer) before
    a weight update, then computes the total change after the update.
    Higher values mean the model learned more from this step.
    """

    def __init__(self) -> None:
        self._snapshot: Optional[dict[str, dict[str, float]]] = None

    def snapshot_before(
        self, brain: BabyBrain, sample_ratio: float = 1.0
    ) -> dict[str, dict[str, float]]:
        """Store parameter snapshot before a learning step.

        Args:
            brain: The BabyBrain model.
            sample_ratio: Fraction of parameters to sample (0.0-1.0).

        Returns:
            The snapshot dictionary (also stored internally).
        """
        self._snapshot = brain.get_parameter_snapshot(
            sample_ratio=sample_ratio
        )
        return self._snapshot

    def compute(
        self, brain: BabyBrain, sample_ratio: float = 1.0
    ) -> float:
        """Compare current params to stored snapshot.

        Sums absolute differences in mean and std across all layers.
        Higher = model changed more during this learning step.

        Args:
            brain: The BabyBrain model (after weight update).
            sample_ratio: Fraction of parameters to sample (0.0-1.0).

        Returns:
            Total information gain score.
        """
        if self._snapshot is None:
            return 0.0

        current = brain.get_parameter_snapshot(sample_ratio=sample_ratio)
        total_change = 0.0

        for name in self._snapshot:
            if name in current:
                mean_diff = abs(
                    current[name]["mean"] - self._snapshot[name]["mean"]
                )
                std_diff = abs(
                    current[name]["std"] - self._snapshot[name]["std"]
                )
                total_change += mean_diff + std_diff

        return total_change
