"""Developmental Emotions and Crisis Detection.

Named emotion regions as hyperrectangles in 5D affect space.
Detects when EVA experiences wonder, attachment, pride, shame,
or curiosity-pain. Includes circuit breaker implementations
and crisis detection for developmental milestones.
"""

from __future__ import annotations

from typing import Any, Optional

from eva.emotions.affect import AffectiveState


class DevelopmentalEmotions:
    """Named emotion regions as hyperrectangles in 5D space.

    Loaded from config. Each emotion has a region (min/max for each
    dimension), a danger (what happens if it persists too long),
    and a breaker (which circuit breaker addresses the danger).
    """

    DIMENSION_NAMES = [
        "valence", "arousal", "dominance", "novelty_feeling", "social"
    ]

    def __init__(self, config: Any) -> None:
        """Initialize from developmental_emotions config section.

        Args:
            config: ConfigSection or dict containing emotion definitions.
        """
        self._emotions: dict[str, dict[str, Any]] = {}
        self._durations: dict[str, int] = {}

        # Parse emotion definitions from config
        for emotion_name in [
            "wonder", "attachment", "pride", "shame", "curiosity_pain"
        ]:
            emotion_cfg = self._get_attr_or_item(config, emotion_name)
            if emotion_cfg is not None:
                region = self._get_attr_or_item(emotion_cfg, "region")
                if region is not None:
                    self._emotions[emotion_name] = {
                        "region": {
                            dim: self._get_attr_or_item(
                                region, dim, [-1.0, 1.0]
                            )
                            for dim in self.DIMENSION_NAMES
                        },
                        "danger": self._get_attr_or_item(
                            emotion_cfg, "danger", "unknown"
                        ),
                        "breaker": self._get_attr_or_item(
                            emotion_cfg, "breaker", "none"
                        ),
                    }
                    self._durations[emotion_name] = 0

    @staticmethod
    def _get_attr_or_item(
        obj: Any, key: str, default: Any = None
    ) -> Any:
        """Get a value from an object by attribute or dict key."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def detect(
        self, state: AffectiveState
    ) -> list[tuple[str, float]]:
        """Detect which named emotions are active.

        Args:
            state: Current affective state.

        Returns:
            List of (emotion_name, intensity) tuples for active emotions.
            Intensity = how deep into the region center (0-1).
        """
        vector = state.to_dict()
        active: list[tuple[str, float]] = []

        for name, emotion in self._emotions.items():
            region = emotion["region"]
            in_region = True
            total_depth = 0.0

            for dim in self.DIMENSION_NAMES:
                val = vector.get(dim, 0.0)
                bounds = region.get(dim, [-1.0, 1.0])
                low, high = bounds[0], bounds[1]

                if val < low or val > high:
                    in_region = False
                    break

                # Depth: how far from the boundary toward center
                range_size = high - low
                if range_size > 0:
                    center = (low + high) / 2.0
                    dist_from_center = abs(val - center)
                    max_dist = range_size / 2.0
                    depth = 1.0 - (dist_from_center / max_dist)
                    total_depth += depth

            if in_region:
                intensity = total_depth / len(self.DIMENSION_NAMES)
                active.append((name, intensity))
                self._durations[name] = self._durations.get(name, 0) + 1
            else:
                self._durations[name] = 0

        return active

    def check_danger(
        self, emotion: str, duration: int, config: Any
    ) -> Optional[str]:
        """Check if an emotion has been active dangerously long.

        Args:
            emotion: Name of the emotion.
            duration: How many steps it has been active.
            config: Circuit breaker config section.

        Returns:
            Danger type string if dangerous, None otherwise.
        """
        if emotion not in self._emotions:
            return None

        danger = self._emotions[emotion]["danger"]
        breaker = self._emotions[emotion]["breaker"]

        # Check against perseveration limit
        limit = getattr(config, "perseveration_limit", 50)
        if duration > limit:
            return danger

        return None

    def get_duration(self, emotion: str) -> int:
        """Get how many consecutive steps an emotion has been active."""
        return self._durations.get(emotion, 0)

    @staticmethod
    def apply_perseveration_limit(
        steps_on_stimulus: int, limit: int
    ) -> bool:
        """Check if perseveration limit is exceeded.

        Args:
            steps_on_stimulus: Steps focused on the same stimulus.
            limit: Maximum allowed steps.

        Returns:
            True if limit exceeded (caller should force redirect).
        """
        return steps_on_stimulus > limit

    @staticmethod
    def apply_gradual_adaptation(
        social_dim: float, target_unavailable_steps: int
    ) -> float:
        """Gradually reduce social dimension when target is unavailable.

        Args:
            social_dim: Current social affect dimension.
            target_unavailable_steps: Steps since target was available.

        Returns:
            Adjusted social dimension.
        """
        decay_per_step = 0.001
        target = 0.3
        adjustment = min(
            decay_per_step * target_unavailable_steps,
            max(0.0, social_dim - target),
        )
        return social_dim - adjustment

    @staticmethod
    def apply_dominance_decay(dominance: float, decay: float) -> float:
        """Apply decay to dominance to prevent arrogance.

        Args:
            dominance: Current dominance value.
            decay: Decay multiplier (e.g., 0.99).

        Returns:
            Decayed dominance value.
        """
        return dominance * decay

    @staticmethod
    def apply_valence_floor(valence: float, floor: float) -> float:
        """Apply floor to valence to prevent self-destruction.

        Args:
            valence: Current valence value.
            floor: Minimum allowed valence.

        Returns:
            Clamped valence value.
        """
        return max(valence, floor)

    @staticmethod
    def apply_redirect(arousal: float) -> float:
        """Reduce arousal when frustration is detected.

        Args:
            arousal: Current arousal value.

        Returns:
            Reduced arousal (20% reduction).
        """
        return arousal * 0.8


class CrisisDetector:
    """Tracks valence history to detect developmental crises.

    A crisis = valence below -0.5 for 20+ consecutive steps,
    then recovers above 0.0. Surviving crises is a developmental
    milestone required for true naming.
    """

    def __init__(
        self,
        crisis_threshold: float = -0.5,
        crisis_duration: int = 20,
        recovery_threshold: float = 0.0,
    ) -> None:
        self._threshold = crisis_threshold
        self._duration = crisis_duration
        self._recovery = recovery_threshold

        self._below_count: int = 0
        self._in_crisis: bool = False
        self._crisis_count: int = 0
        self._crises_survived: int = 0

    def update(self, valence: float) -> None:
        """Update crisis detection with new valence value.

        Args:
            valence: Current valence value.
        """
        if valence < self._threshold:
            self._below_count += 1
            if self._below_count >= self._duration and not self._in_crisis:
                self._in_crisis = True
                self._crisis_count += 1
        else:
            if self._in_crisis and valence > self._recovery:
                # Crisis survived — recovered above threshold
                self._crises_survived += 1
                self._in_crisis = False
            self._below_count = 0

    def crisis_survived(self) -> bool:
        """Check if at least one crisis has been detected and resolved.

        Returns:
            True if at least one crisis was survived.
        """
        return self._crises_survived > 0

    def crisis_count(self) -> int:
        """Total number of crises detected (including current)."""
        return self._crisis_count

    @property
    def crises_survived(self) -> int:
        """Number of crises that were survived (recovered from)."""
        return self._crises_survived
