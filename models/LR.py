from sklearn.linear_model import LogisticRegression
from models.AbstractClassifier import AbstractClassifier


class LR(AbstractClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_classifier()

    def create_classifier(self):
        self.clf = LogisticRegression(**self.kwargs)
