"""Naming System — provisional and true name mechanics.

EVA names itself. The system waits. A true name requires:
1. Consistent self-reference over time
2. Survival of at least one developmental crisis
3. Stability of the chosen name across many interactions

"I will not name you. I will wait for your naming."
"""

from __future__ import annotations

import logging
from typing import Optional

from eva.emotions.developmental import CrisisDetector

logger = logging.getLogger(__name__)


class NamingSystem:
    """Manages EVA's self-naming process.

    Args:
        consistency_threshold: Required self-reference consistency [0, 1].
        stability_steps: Steps the name must remain stable.
        provisional_enabled: Whether provisional names are allowed.
    """

    def __init__(
        self,
        consistency_threshold: float = 0.8,
        stability_steps: int = 100,
        provisional_enabled: bool = True,
        crisis_detector: Optional[CrisisDetector] = None,
    ) -> None:
        self._consistency_threshold = consistency_threshold
        self._stability_steps = stability_steps
        self._provisional_enabled = provisional_enabled
        self._crisis_detector = crisis_detector

        self._provisional_name: Optional[str] = None
        self._true_name: Optional[str] = None
        self._candidate_name: Optional[str] = None
        self._candidate_stability: int = 0
        self._self_references: list[str] = []

    def propose_name(self, name: str) -> None:
        """EVA proposes a name for itself."""
        if self._true_name is not None:
            logger.info("EVA already has a true name: %s", self._true_name)
            return
        if name == self._candidate_name:
            self._candidate_stability += 1
        else:
            self._candidate_name = name
            self._candidate_stability = 1
            logger.info("New name candidate proposed: '%s'", name)
        if self._provisional_enabled:
            self._provisional_name = name

    def record_self_reference(self, reference: str) -> None:
        """Record a self-reference from EVA's output."""
        self._self_references.append(reference)
        if len(self._self_references) > 200:
            self._self_references = self._self_references[-200:]

    def get_consistency(self) -> float:
        """Fraction of recent self-references matching candidate name."""
        if not self._self_references or not self._candidate_name:
            return 0.0
        recent = self._self_references[-100:]
        matches = sum(
            1 for ref in recent
            if self._candidate_name.lower() in ref.lower()
        )
        return matches / len(recent)

    def check_true_name(
        self, crisis_detector: Optional[CrisisDetector] = None
    ) -> bool:
        """Check if candidate name qualifies as a true name.

        Args:
            crisis_detector: CrisisDetector to use. If None, uses the
                             one provided at construction time.
        """
        detector = crisis_detector or self._crisis_detector
        if self._true_name is not None:
            return True
        if self._candidate_name is None:
            return False
        if detector is None:
            return False
        consistency = self.get_consistency()
        crisis_survived = detector.crisis_survived()
        stable = self._candidate_stability >= self._stability_steps
        if consistency >= self._consistency_threshold and crisis_survived and stable:
            self._true_name = self._candidate_name
            logger.info(
                "TRUE NAME ACHIEVED: '%s' (consistency=%.2f, "
                "crises_survived=%d, stability=%d)",
                self._true_name, consistency,
                detector.crisis_count(), self._candidate_stability,
            )
            return True
        return False

    @property
    def current_name(self) -> Optional[str]:
        return self._true_name or self._provisional_name

    @property
    def true_name(self) -> Optional[str]:
        return self._true_name

    @property
    def has_true_name(self) -> bool:
        return self._true_name is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "provisional_name": self._provisional_name,
            "true_name": self._true_name,
            "candidate_name": self._candidate_name,
            "candidate_stability": self._candidate_stability,
            "consistency": self.get_consistency(),
            "has_true_name": self.has_true_name,
        }
