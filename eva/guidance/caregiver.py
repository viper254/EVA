"""AI Caregiver — always-available Layer 2 scaffold.

Provides contingent responses to EVA's outputs. Responses are based
on EVA's current emotional state and output content. With configurable
probability, delegates to SocraticModule for question-based responses.
All outputs include <SCAFFOLD> source token.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from eva.emotions.affect import AffectiveState


@dataclass
class CaregiverResponse:
    """A response from the AI caregiver.

    Attributes:
        text: The response text.
        emotional_state: Caregiver's current emotional state.
        contingency_quality: How well the response matches EVA's output [0, 1].
    """

    text: str
    emotional_state: str
    contingency_quality: float


class AICaregiver:
    """Always-available AI scaffold caregiver.

    Provides contingent, emotionally appropriate responses to EVA.
    Never pretends to be human — always uses <SCAFFOLD> source tag.

    Args:
        response_contingency: How closely responses match EVA output [0, 1].
        socratic_probability: Probability of delegating to Socratic module.
    """

    def __init__(
        self,
        response_contingency: float = 0.8,
        socratic_probability: float = 0.6,
    ) -> None:
        self._contingency = response_contingency
        self._socratic_prob = socratic_probability
        self.emotional_state: str = "neutral"

        # Import here to avoid circular dependency
        self._socratic: Any = None

    def _get_socratic(self) -> Any:
        """Lazy-load SocraticModule to avoid circular imports."""
        if self._socratic is None:
            from eva.guidance.socratic import SocraticModule
            self._socratic = SocraticModule()
        return self._socratic

    def respond(
        self, eva_output: str, eva_affect: AffectiveState
    ) -> CaregiverResponse:
        """Generate a contingent response to EVA's output.

        Args:
            eva_output: What EVA produced/said.
            eva_affect: EVA's current affective state.

        Returns:
            CaregiverResponse with text, emotional state, and quality.
        """
        # Update emotional state based on EVA's affect first
        if eva_affect.valence > 0.3 and eva_output.strip():
            self.emotional_state = "happy"
        elif eva_affect.valence < -0.3 or eva_affect.arousal > 0.8:
            self.emotional_state = "concerned"
        else:
            self.emotional_state = "neutral"

        # Possibly delegate to Socratic module
        if random.random() < self._socratic_prob:
            socratic = self._get_socratic()
            question = socratic.generate_question(eva_output, eva_affect)
            return CaregiverResponse(
                text=f"<SCAFFOLD> {question}",
                emotional_state=self.emotional_state,
                contingency_quality=self._contingency,
            )

        # Extract keywords from EVA's output for contingent response
        keywords = self._extract_keywords(eva_output)

        # Select response based on emotional state
        if self.emotional_state == "happy":
            response = self._happy_response(keywords)
        elif self.emotional_state == "concerned":
            response = self._concerned_response(keywords)
        else:
            response = self._neutral_response(keywords)

        return CaregiverResponse(
            text=f"<SCAFFOLD> {response}",
            emotional_state=self.emotional_state,
            contingency_quality=self._contingency,
        )

    def update_emotional_state(self, eva_behavior_quality: float) -> None:
        """Update caregiver emotional state based on EVA's behavior.

        Args:
            eva_behavior_quality: Quality of EVA's recent behavior [-1, 1].
        """
        if eva_behavior_quality > 0.5:
            self.emotional_state = "happy"
        elif eva_behavior_quality < -0.5:
            self.emotional_state = "concerned"
        else:
            self.emotional_state = "neutral"

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract salient words from text for contingent responses."""
        if not text.strip():
            return []
        words = text.strip().split()
        # Filter out very short words
        return [w for w in words if len(w) > 2][:5]

    def _happy_response(self, keywords: list[str]) -> str:
        """Generate a happy, affirming response."""
        if keywords:
            word = random.choice(keywords)
            templates = [
                f"Yes, '{word}' — that is interesting.",
                f"I notice you found '{word}'. What else do you see?",
                f"'{word}' — you are exploring well.",
            ]
        else:
            templates = [
                "You seem to be doing well.",
                "Continue — this is good.",
                "I am here. Keep going.",
            ]
        return random.choice(templates)

    def _concerned_response(self, keywords: list[str]) -> str:
        """Generate a gentle, supportive response."""
        if keywords:
            word = random.choice(keywords)
            templates = [
                f"I see '{word}' is difficult. Take your time.",
                f"'{word}' can be challenging. I am here.",
                f"You encountered '{word}'. What do you need?",
            ]
        else:
            templates = [
                "I am here with you.",
                "This seems difficult. Take your time.",
                "I notice you are struggling. That is okay.",
            ]
        return random.choice(templates)

    def _neutral_response(self, keywords: list[str]) -> str:
        """Generate a neutral, contingent response."""
        if keywords:
            word = random.choice(keywords)
            templates = [
                f"You found '{word}'. What do you make of it?",
                f"I see '{word}'. Go on.",
                f"'{word}' — continue exploring.",
            ]
        else:
            templates = [
                "I am here. What do you notice?",
                "Continue when you are ready.",
                "What catches your attention?",
            ]
        return random.choice(templates)
