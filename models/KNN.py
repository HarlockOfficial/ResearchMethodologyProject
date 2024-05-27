from sklearn.neighbors import KNeighborsClassifier

from models.MachineLearningClassifier import MachineLearningClassifier


class KNN(MachineLearningClassifier):
    def __init__(self, *, n_neighbours=5, **kwargs):
        super().__init__(KNeighborsClassifier(n_neighbors=n_neighbours))
