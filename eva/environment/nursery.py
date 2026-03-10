"""Nursery Environment — safe learning environment for early development.

The nursery provides simple, structured stimuli for EVA's earliest
learning. It generates random character sequences, simple patterns,
and repetitions that allow EVA to begin developing prediction abilities.

Patterns are organized into difficulty tiers:
- Tier 0 (0.0+): Single-char repetitions, simple alternations
- Tier 1 (0.2+): Counting sequences, mirror patterns
- Tier 2 (0.4+): Multi-char patterns, mathematical sequences
- Tier 3 (0.6+): Phrases, nested structures, palindromes
- Tier 4 (0.8+): Sentences with grammar, conditional structures
"""

from __future__ import annotations

import logging
import random
import string
from typing import Any

from eva.core.tokenizer import EVATokenizer
from eva.environment.base import BaseEnvironment

logger = logging.getLogger(__name__)


class NurseryEnvironment(BaseEnvironment):
    """Safe learning environment for early EVA development.

    Generates simple stimuli: random characters, repeating patterns,
    and basic sequences. Complexity increases as EVA progresses through
    developmental phases. Supports adaptive difficulty adjustment based
    on EVA's prediction accuracy.

    Args:
        tokenizer: The EVA tokenizer for encoding stimuli.
        difficulty: Starting difficulty level (0.0 = simplest, 1.0 = hardest).
        adaptive: If True, auto-adjust difficulty based on accuracy.
    """

    def __init__(
        self,
        tokenizer: EVATokenizer,
        difficulty: float = 0.0,
        adaptive: bool = True,
    ) -> None:
        super().__init__(name="nursery")
        self._tokenizer = tokenizer
        self._difficulty = max(0.0, min(1.0, difficulty))
        self._current_sequence: list[int] = []
        self._position: int = 0
        self._patterns: list[str] = self._generate_patterns()

        # Adaptive difficulty tracking
        self._adaptive = adaptive
        self._recent_correct: int = 0
        self._recent_total: int = 0
        self._adapt_window: int = 50  # evaluate every 50 steps

    def _generate_patterns(self) -> list[str]:
        """Generate stimulus patterns based on difficulty.

        Returns a rich set of patterns organized by difficulty tier.
        """
        patterns: list[str] = []

        # --- Tier 0 (always available): Simple repetitions ---
        for char in "abcde":
            patterns.append(char * 10)

        # Simple alternations
        patterns.extend(["ababababab", "abcabcabc", "aabbaabb"])

        if self._difficulty > 0.2:
            # --- Tier 1: Counting and mirror patterns ---
            # Counting sequences
            patterns.extend([
                "1 2 3 4 5 6 7 8 9 10",
                "a b c d e f g h",
                "2 4 6 8 10 12 14 16",
            ])
            # Mirror / palindrome seeds
            patterns.extend([
                "abcba abcba abcba",
                "abccba abccba",
                "12321 12321 12321",
            ])

        if self._difficulty > 0.4:
            # --- Tier 2: Multi-char patterns, math sequences ---
            patterns.extend([
                "abcdabcdabcd",
                "aabbccaabbcc",
                "abcdeabcde",
            ])
            # Fibonacci-like
            patterns.extend([
                "1 1 2 3 5 8 13 21 34",
                "0 1 1 2 3 5 8 13",
            ])
            # Powers
            patterns.extend([
                "1 2 4 8 16 32 64 128",
                "1 3 9 27 81 243",
            ])
            # Nested repetitions
            patterns.extend([
                "aab aab aab bba bba bba",
                "xy xy xy yx yx yx",
            ])

        if self._difficulty > 0.6:
            # --- Tier 3: Phrases and structural patterns ---
            patterns.extend([
                "the cat sat on the mat",
                "one two three one two three",
                "hello world hello world",
            ])
            # Call-and-response
            patterns.extend([
                "knock knock who is there",
                "ready set go ready set go",
                "tick tock tick tock tick tock",
            ])
            # Nested structures
            patterns.extend([
                "a bb ccc dddd ccc bb a",
                "open close open close open close",
                "up down up up down down up down",
            ])
            # Simple cause-effect
            patterns.extend([
                "push fall push fall push fall",
                "ask answer ask answer ask answer",
            ])

        if self._difficulty > 0.8:
            # --- Tier 4: Sentences with grammar ---
            patterns.extend([
                "if it rains then we stay inside",
                "the big dog chased the small cat",
                "I see you and you see me",
            ])
            # More complex grammar
            patterns.extend([
                "when the sun rises the birds sing",
                "she said hello and he said goodbye",
                "first we walk then we run then we rest",
            ])
            # Longer narrative sequences
            patterns.extend([
                "the seed grows into a plant and the plant makes new seeds",
                "water flows down from the mountain to the river to the sea",
            ])
            # Question-answer patterns
            patterns.extend([
                "what is this it is a ball what is that it is a box",
                "where is the cat the cat is here where is the dog the dog is there",
            ])

        return patterns

    def reset(self) -> list[int]:
        """Reset with a new random pattern.

        Returns:
            Initial token sequence (first few tokens of pattern).
        """
        pattern = random.choice(self._patterns)
        self._current_sequence = self._tokenizer.encode(pattern)
        self._position = min(3, len(self._current_sequence) - 1)
        self._step_count = 0
        return self._current_sequence[:self._position]

    def step(self, action: int) -> tuple[int, dict[str, Any]]:
        """Reveal the next actual token.

        Args:
            action: EVA's predicted next token.

        Returns:
            Tuple of (actual_token, info_dict).
        """
        if self._position >= len(self._current_sequence):
            # Pattern exhausted — start a new one
            self.reset()

        actual = self._current_sequence[self._position]
        correct = action == actual
        self._position += 1
        self._step_count += 1

        # Track accuracy for adaptive difficulty
        if self._adaptive:
            self._recent_total += 1
            if correct:
                self._recent_correct += 1
            if self._recent_total >= self._adapt_window:
                accuracy = self._recent_correct / self._recent_total
                self._adapt_difficulty(accuracy)
                self._recent_correct = 0
                self._recent_total = 0

        info = {
            "correct": correct,
            "position": self._position,
            "pattern_length": len(self._current_sequence),
            "difficulty": self._difficulty,
        }

        return actual, info

    def _adapt_difficulty(self, accuracy: float) -> None:
        """Adjust difficulty based on recent prediction accuracy.

        Increases difficulty when accuracy is high (EVA is ready),
        decreases slightly when accuracy is very low (struggling).

        Args:
            accuracy: Recent prediction accuracy (0.0 - 1.0).
        """
        old_difficulty = self._difficulty
        if accuracy > 0.7:
            # EVA is doing well — increase challenge
            self._difficulty = min(1.0, self._difficulty + 0.05)
        elif accuracy < 0.2 and self._difficulty > 0.1:
            # EVA is struggling — ease off slightly
            self._difficulty = max(0.0, self._difficulty - 0.02)

        if self._difficulty != old_difficulty:
            self._patterns = self._generate_patterns()
            logger.info(
                "Nursery difficulty adapted: %.2f -> %.2f "
                "(accuracy=%.2f)",
                old_difficulty,
                self._difficulty,
                accuracy,
            )

    def get_current_sequence(self) -> list[int]:
        """Return tokens seen so far."""
        return self._current_sequence[:self._position]

    def increase_difficulty(self, amount: float = 0.1) -> None:
        """Increase environment difficulty.

        Args:
            amount: How much to increase difficulty.
        """
        self._difficulty = min(1.0, self._difficulty + amount)
        self._patterns = self._generate_patterns()

    @property
    def difficulty(self) -> float:
        """Current difficulty level."""
        return self._difficulty
