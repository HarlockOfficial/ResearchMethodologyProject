import enum

from .ADA import ADA
from .BAG import BAG
from .DT import DT
from .GB import GB
from .KNN import KNN
from .LR import LR
from .MLP import MLP
from .RandomForest import RandomForest


class ModelType(enum.Enum):
    Mlp = MLP
    RandomForest = RandomForest
    Gb = GB
    Knn = KNN
    Dt = DT
    Ada = ADA
    Bag = BAG
    Lr = LR
