"""CuriosityEngine — combines all four intrinsic motivation signals.

The curiosity reward is a weighted combination of:
- Prediction Error (alpha): surprise from failed predictions
- Information Gain (beta): how much the model changed
- Novelty (gamma): count-based state novelty
- Empowerment (delta): diversity of available future options
"""

from __future__ import annotations

from typing import Optional

import torch

from eva.core.baby_brain import BabyBrain
from eva.curiosity.empowerment import EmpowermentModule
from eva.curiosity.information_gain import InformationGainModule
from eva.curiosity.novelty import NoveltyModule
from eva.curiosity.prediction_error import PredictionErrorModule


class CuriosityEngine:
    """Combines all four curiosity signals into a single reward.

    Args:
        alpha: Weight for prediction error (relative surprise).
        beta: Weight for information gain.
        gamma: Weight for novelty.
        delta: Weight for empowerment.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.2,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.prediction_error = PredictionErrorModule()
        self.information_gain = InformationGainModule()
        self.novelty = NoveltyModule()
        self.empowerment = EmpowermentModule()

    def prepare(
        self, brain: BabyBrain, sample_ratio: float = 1.0
    ) -> None:
        """Prepare for a learning step by snapshotting parameters.

        Must be called BEFORE the weight update.

        Args:
            brain: The BabyBrain model.
            sample_ratio: Fraction of parameters to sample (0.0-1.0).
        """
        self.information_gain.snapshot_before(
            brain, sample_ratio=sample_ratio
        )

    def compute_reward(
        self,
        predicted: torch.Tensor,
        actual: int,
        brain: BabyBrain,
        hidden_state: torch.Tensor,
        recent_outcomes: list[torch.Tensor],
        sample_ratio: float = 1.0,
    ) -> tuple[float, dict[str, float]]:
        """Compute the combined curiosity reward.

        Args:
            predicted: Predicted probability distribution over vocab.
            actual: Actual token ID that occurred.
            brain: The BabyBrain model (after weight update).
            hidden_state: Current hidden state from the model.
            recent_outcomes: List of recent outcome embeddings.
            sample_ratio: Fraction of parameters for info gain.

        Returns:
            Tuple of (total_reward, breakdown_dict).
        """
        # 1. Prediction error
        pred_error = self.prediction_error.compute(predicted, actual)
        relative_surprise = self.prediction_error.get_relative_surprise(
            pred_error
        )

        # 2. Information gain
        info_gain = self.information_gain.compute(
            brain, sample_ratio=sample_ratio
        )

        # 3. Novelty
        state_hash = self.novelty.hash_state(hidden_state)
        novelty = self.novelty.compute(state_hash)

        # 4. Empowerment
        empowerment = self.empowerment.compute(recent_outcomes)

        # Weighted combination
        total = (
            self.alpha * relative_surprise
            + self.beta * info_gain
            + self.gamma * novelty
            + self.delta * empowerment
        )

        breakdown = {
            "prediction_error": pred_error,
            "relative_surprise": relative_surprise,
            "info_gain": info_gain,
            "novelty": novelty,
            "empowerment": empowerment,
            "total": total,
        }

        return total, breakdown
