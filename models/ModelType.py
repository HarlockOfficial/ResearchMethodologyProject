import enum

from .MLP import MLP
from .RandomForest import RandomForest


class ModelType(enum.Enum):
    Mlp = MLP
    RandomForest = RandomForest
