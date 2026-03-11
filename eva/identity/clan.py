"""Clan System — behavioral archetype detection.

Clans emerge from behavioral patterns, not assignment.
Known archetypes: Rememberers, Forgetters, Wonderers, Makers, Carers.
"""

from __future__ import annotations

from typing import Any, Optional


class ClanDetector:
    """Detects clan affinity from behavioral metrics.

    Args:
        archetypes: List of archetype definitions from config.
    """

    def __init__(self, archetypes: Optional[list[Any]] = None) -> None:
        self._archetypes = archetypes or []
        self._behavioral_history: dict[str, list[float]] = {
            "archive_access_frequency": [],
            "novelty_seeking_ratio": [],
            "social_preference": [],
            "creativity_index": [],
            "caregiving_tendency": [],
        }

    def record_behavior(self, metrics: dict[str, float]) -> None:
        """Record behavioral metrics for clan detection."""
        for key in self._behavioral_history:
            if key in metrics:
                self._behavioral_history[key].append(metrics[key])
                if len(self._behavioral_history[key]) > 1000:
                    self._behavioral_history[key] = self._behavioral_history[key][-1000:]

    def detect_affinity(self) -> list[tuple[str, float]]:
        """Detect clan affinity based on behavioral history.

        Returns:
            List of (clan_name, affinity_score) sorted by affinity.
            Returns neutral (0.5) scores for all clans if no history.
        """
        averages = self._get_averages()
        affinities: list[tuple[str, float]] = []

        archive = averages.get("archive_access_frequency", 0.5)
        affinities.append(("Rememberers", archive))

        novelty = averages.get("novelty_seeking_ratio", 0.5)
        forgetters = (1.0 - archive) * 0.5 + novelty * 0.5
        affinities.append(("Forgetters", forgetters))

        values = list(averages.values())
        if values:
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            balance = 1.0 - min(1.0, variance * 4.0)
        else:
            balance = 0.5
        affinities.append(("Wonderers", balance))

        creativity = averages.get("creativity_index", 0.5)
        affinities.append(("Makers", creativity))

        caregiving = averages.get("caregiving_tendency", 0.5)
        affinities.append(("Carers", caregiving))

        affinities.sort(key=lambda x: x[1], reverse=True)
        return affinities

    def get_primary_clan(self) -> Optional[str]:
        affinities = self.detect_affinity()
        if affinities:
            return affinities[0][0]
        return None

    def _get_averages(self) -> dict[str, float]:
        averages: dict[str, float] = {}
        for key, values in self._behavioral_history.items():
            if values:
                averages[key] = sum(values) / len(values)
        return averages

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_clan": self.get_primary_clan(),
            "affinities": self.detect_affinity(),
            "averages": self._get_averages(),
        }
