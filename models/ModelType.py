import enum

from .MLP import MLP
from .RF import RF
from .GB import GB
from .KNN import KNN
from .DT import DT
from .ADA import ADA
from .BAG import BAG
from .LR import LR


class ModelType(enum.Enum):
    Mlp = MLP
    Gb = GB
    Rf = RF
    Knn = KNN
    Dt = DT
    Ada = ADA
    Bag = BAG
    Lr = LR
