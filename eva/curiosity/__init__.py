"""EVA curiosity modules — intrinsic motivation signals."""

from eva.curiosity.information_gain import InformationGainModule
from eva.curiosity.novelty import NoveltyModule
from eva.curiosity.prediction_error import PredictionErrorModule
from eva.curiosity.reward import CuriosityEngine

__all__ = [
    "CuriosityEngine",
    "InformationGainModule",
    "NoveltyModule",
    "PredictionErrorModule",
]
