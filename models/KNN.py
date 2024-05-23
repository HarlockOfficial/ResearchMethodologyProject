from sklearn.neighbors import KNeighborsClassifier

from models.MachineLearningClassifier import MachineLearningClassifier


class KNN(MachineLearningClassifier):
    def __init__(self, **kwargs):
        super().__init__(KNeighborsClassifier(**kwargs))
