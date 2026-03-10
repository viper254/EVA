"""EVA training modules — training loop and curriculum."""

from eva.training.curriculum import DevelopmentalCurriculum
from eva.training.loop import TrainingLoop

__all__ = ["DevelopmentalCurriculum", "TrainingLoop"]
