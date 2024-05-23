from models.AbstractClassifier import AbstractClassifier
from sklearn.neighbors import KNeighborsClassifier


class KNN(AbstractClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_classifier()

    def create_classifier(self):
        self.clf = KNeighborsClassifier(**self.kwargs)
