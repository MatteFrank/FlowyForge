"""Task plugin namespace."""

from flowyforge.task_plugins.calo_generation import CaloGenerationTask
from flowyforge.task_plugins.classification import ClassificationTask
from flowyforge.task_plugins.masked_modeling import MaskedModelingTask
from flowyforge.task_plugins.trigger import TriggerTask

__all__ = [
    "CaloGenerationTask",
    "ClassificationTask",
    "MaskedModelingTask",
    "TriggerTask",
]

