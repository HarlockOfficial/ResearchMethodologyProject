from sklearn.tree import DecisionTreeClassifier
from models.AbstractClassifier import AbstractClassifier


class DT(AbstractClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_classifier()

    def create_classifier(self):
        self.clf = DecisionTreeClassifier(**self.kwargs)
