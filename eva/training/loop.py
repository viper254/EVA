"""Training Loop — curiosity-driven learning with emotional modulation.

The core training loop where EVA learns from its environment:
1. Observe stimulus from environment
2. Predict next token
3. Compute curiosity reward from prediction error
4. Update emotions based on experience
5. Modulate learning rate and exploration
6. Update weights (with gradient accumulation)
7. Store experience in episodic memory
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from eva.core.baby_brain import BabyBrain
from eva.core.config import EVAConfig
from eva.core.tokenizer import EVATokenizer
from eva.curiosity.reward import CuriosityEngine
from eva.emotions.affect import AffectiveState
from eva.emotions.developmental import CrisisDetector, DevelopmentalEmotions
from eva.emotions.homeostasis import Homeostasis
from eva.emotions.modulation import EmotionalModulation
from eva.environment.base import BaseEnvironment
from eva.guidance.caregiver import AICaregiver
from eva.guidance.presence import PresenceDynamics
from eva.memory.episodic import Episode, EpisodicMemory
from eva.training.curriculum import DevelopmentalCurriculum

logger = logging.getLogger(__name__)


def _get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    """Create a learning rate scheduler with optional warmup.

    Args:
        optimizer: The optimizer.
        scheduler_type: "cosine", "linear", or "none".
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.

    Returns:
        LambdaLR scheduler, or None if scheduler_type is "none".
    """
    if scheduler_type == "none":
        return None

    def lr_lambda(current_step: int) -> float:
        # Linear warmup phase
        if current_step < warmup_steps:
            return max(0.01, current_step / max(1, warmup_steps))
        # Decay phase
        progress = (current_step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        if scheduler_type == "cosine":
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif scheduler_type == "linear":
            return max(0.01, 1.0 - progress)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TrainingLoop:
    """Main training loop for EVA development.

    Orchestrates curiosity-driven learning with emotional modulation,
    caregiver interaction, and developmental curriculum progression.

    Args:
        brain: The EVA neural network.
        config: EVA configuration.
        environment: The learning environment.
        tokenizer: The tokenizer.
    """

    def __init__(
        self,
        brain: BabyBrain,
        config: EVAConfig,
        environment: BaseEnvironment,
        tokenizer: EVATokenizer,
    ) -> None:
        self.brain = brain
        self.config = config
        self.environment = environment
        self.tokenizer = tokenizer

        # Device
        self.device = brain.device

        # Training hyperparameters
        self._grad_accum_steps = getattr(
            config.training, "gradient_accumulation_steps", 1
        )
        self._max_grad_norm = getattr(config.training, "max_grad_norm", 1.0)
        self._info_gain_sample_ratio = getattr(
            config.training, "info_gain_sample_ratio", 1.0
        )

        # Curiosity system
        self.curiosity = CuriosityEngine(
            alpha=getattr(config.curiosity, "alpha", 0.3),
            beta=getattr(config.curiosity, "beta", 0.3),
            gamma=getattr(config.curiosity, "gamma", 0.2),
            delta=getattr(config.curiosity, "delta", 0.2),
        )

        # Emotional system
        self.affect = AffectiveState()
        self.homeostasis = Homeostasis()
        self.modulation = EmotionalModulation()
        self.developmental_emotions = DevelopmentalEmotions(
            config.developmental_emotions
        )
        self.crisis_detector = CrisisDetector()

        # Memory
        max_mem_size = getattr(config.memory, "max_size", 10000)
        self.memory = EpisodicMemory(max_size=max_mem_size)

        # Guidance
        self.caregiver = AICaregiver(
            response_contingency=getattr(
                config.guidance.ai_scaffold, "response_contingency", 0.8
            ),
            socratic_probability=getattr(
                config.guidance.ai_scaffold, "socratic_probability", 0.6
            ),
        )
        self.presence = PresenceDynamics()

        # Curriculum
        self.curriculum = DevelopmentalCurriculum(
            starting_phase=getattr(config.training, "phase", "prenatal")
        )

        # Optimizer
        lr = getattr(config.training, "learning_rate", 1e-4)
        self.optimizer = torch.optim.Adam(brain.parameters(), lr=lr)

        # State tracking
        self._step: int = 0
        self._steps_since_social: int = 0
        self._steps_active: int = 0
        self._recent_outcomes: list[torch.Tensor] = []

    def train(
        self,
        num_steps: int,
        checkpoint_every: int = 1000,
        log_every: int = 10,
        checkpoint_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run the training loop for a number of steps.

        Args:
            num_steps: Total training steps.
            checkpoint_every: Save checkpoint every N steps.
            log_every: Log metrics every N steps.
            checkpoint_path: Path prefix for checkpoints.

        Returns:
            Dictionary of training statistics.
        """
        self.brain.train()
        sequence = self.environment.reset()
        total_reward = 0.0
        total_loss = 0.0
        correct_predictions = 0

        # Set up LR scheduler
        scheduler_type = getattr(
            self.config.training, "lr_scheduler", "none"
        )
        warmup_steps = getattr(self.config.training, "warmup_steps", 0)
        scheduler = _get_lr_scheduler(
            self.optimizer, scheduler_type, warmup_steps, num_steps
        )

        # Gradient accumulation counter
        accum_count = 0

        logger.info(
            "Training started: %d steps, phase=%s, device=%s, "
            "grad_accum=%d, scheduler=%s, warmup=%d",
            num_steps,
            self.curriculum.current_phase,
            self.device,
            self._grad_accum_steps,
            scheduler_type,
            warmup_steps,
        )

        for step_i in tqdm(range(num_steps), desc="Training"):
            self._step += 1
            self._steps_active += 1
            self._steps_since_social += 1

            # Get current context
            context = self.environment.get_current_sequence()
            if len(context) < 2:
                sequence = self.environment.reset()
                context = self.environment.get_current_sequence()
                if len(context) < 2:
                    continue

            # Prepare input tensor on correct device
            input_ids = torch.tensor(
                [context], dtype=torch.long, device=self.device
            )

            # Snapshot params before update (for information gain)
            # Use sampling to reduce overhead
            self.curiosity.prepare(
                self.brain,
                sample_ratio=self._info_gain_sample_ratio,
            )

            # Forward pass with mixed precision
            with torch.amp.autocast(
                device_type=self.device.type, dtype=torch.float16
            ):
                predicted_dist = self.brain.predict_next(input_ids)
                hidden = self.brain.get_hidden_state()

            # EVA's prediction
            predicted_token = predicted_dist.argmax(dim=-1).item()

            # Environment reveals actual token
            actual_token, env_info = self.environment.step(predicted_token)
            correct = env_info.get("correct", False)
            if correct:
                correct_predictions += 1

            # Compute loss
            target = torch.tensor([actual_token], device=self.device)
            log_probs = torch.log(predicted_dist.float().squeeze(0) + 1e-10)
            loss = F.nll_loss(log_probs.unsqueeze(0), target)

            # Scale loss for gradient accumulation
            scaled_loss = loss / self._grad_accum_steps
            total_loss += loss.item()

            # Backward pass — accumulate gradients
            scaled_loss.backward()
            accum_count += 1

            # Apply emotional modulation to LR
            lr_mult = self.modulation.get_learning_rate_multiplier(
                self.affect, self.homeostasis
            )

            # Step optimizer only every grad_accum_steps
            if accum_count >= self._grad_accum_steps:
                base_lr = getattr(
                    self.config.training, "learning_rate", 1e-4
                )
                for pg in self.optimizer.param_groups:
                    pg["lr"] = base_lr * lr_mult

                torch.nn.utils.clip_grad_norm_(
                    self.brain.parameters(), self._max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Step scheduler after optimizer
                if scheduler is not None:
                    scheduler.step()

                accum_count = 0

            # Compute curiosity reward
            reward, reward_breakdown = self.curiosity.compute_reward(
                predicted_dist.detach(),
                actual_token,
                self.brain,
                hidden,
                self._recent_outcomes,
                sample_ratio=self._info_gain_sample_ratio,
            )
            total_reward += reward

            # Track outcomes for empowerment
            if hidden is not None:
                outcome_emb = hidden.mean(dim=1).squeeze(0).detach()
                self._recent_outcomes.append(outcome_emb)
                if len(self._recent_outcomes) > 50:
                    self._recent_outcomes.pop(0)

            # Update emotions
            prediction_success = 1.0 if correct else 0.0
            caregiver_recency = 1.0 / (
                1.0 + self._steps_since_social * 0.01
            )
            self.affect.update(
                prediction_success=prediction_success,
                prediction_error=reward_breakdown["prediction_error"],
                action_success=correct_predictions / max(1, self._step),
                caregiver_recency=caregiver_recency,
                caregiver_contingency=0.8,
            )
            cb_config = getattr(
                self.config.emotions, "circuit_breakers", None
            )
            if cb_config:
                self.affect.apply_circuit_breakers(cb_config)

            # Update homeostasis
            self.homeostasis.update(
                curiosity_reward=reward,
                steps_active=self._steps_active,
                steps_since_social=self._steps_since_social,
            )

            # Crisis detection
            self.crisis_detector.update(self.affect.valence)

            # Curriculum progression
            self.curriculum.update_competence(
                "prediction", prediction_success
            )
            self.curriculum.step()

            # Store in memory
            importance = self.modulation.get_memory_importance(self.affect)
            state_emb = (
                hidden.mean(dim=1).squeeze(0).detach()
                if hidden is not None
                else torch.zeros(self.brain.d_model, device=self.device)
            )
            episode = Episode(
                state_embedding=state_emb,
                action=predicted_token,
                outcome=actual_token,
                surprise=reward_breakdown["prediction_error"],
                emotional_importance=importance,
                source_tag="self",
                timestamp=self._step,
            )
            self.memory.store(episode)

            # Rest period (memory consolidation)
            if self.homeostasis.needs_rest():
                self.memory.consolidate()
                self._steps_active = 0

            # Logging
            if self._step % log_every == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Step %d | loss=%.4f | reward=%.4f | "
                    "valence=%.2f | arousal=%.2f | phase=%s | "
                    "lr=%.2e | lr_mult=%.2f",
                    self._step,
                    loss.item(),
                    reward,
                    self.affect.valence,
                    self.affect.arousal,
                    self.curriculum.current_phase,
                    current_lr,
                    lr_mult,
                )

            # Checkpoint
            if checkpoint_path and self._step % checkpoint_every == 0:
                self._save_checkpoint(checkpoint_path)

        stats = {
            "total_steps": self._step,
            "total_reward": total_reward,
            "avg_loss": total_loss / max(1, num_steps),
            "accuracy": correct_predictions / max(1, num_steps),
            "final_phase": self.curriculum.current_phase,
            "affect": self.affect.to_dict(),
            "homeostasis": self.homeostasis.get_drives(),
            "memory_size": self.memory.size(),
            "crises_survived": self.crisis_detector.crises_survived,
        }

        logger.info("Training complete: %s", stats)
        return stats

    def _save_checkpoint(self, path_prefix: str) -> None:
        """Save a training checkpoint."""
        path = f"{path_prefix}_step{self._step}.pt"
        torch.save(
            {
                "step": self._step,
                "brain_state_dict": self.brain.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "affect": self.affect.to_dict(),
                "curriculum": self.curriculum.to_dict(),
                "homeostasis": self.homeostasis.get_drives(),
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint."""
        checkpoint = torch.load(
            path, weights_only=False, map_location=self.device
        )
        self.brain.load_state_dict(checkpoint["brain_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._step = checkpoint.get("step", 0)
        logger.info("Checkpoint loaded: %s (step %d)", path, self._step)

    @property
    def step_count(self) -> int:
        return self._step
