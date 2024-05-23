from sklearn.ensemble import AdaBoostClassifier
from models.AbstractClassifier import AbstractClassifier


class ADA(AbstractClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_classifier()

    def create_classifier(self):
        self.clf = AdaBoostClassifier(**self.kwargs)
