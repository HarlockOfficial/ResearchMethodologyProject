from sklearn.ensemble import GradientBoostingClassifier
from models.AbstractClassifier import AbstractClassifier


class GB(AbstractClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_classifier()

    def create_classifier(self):
        self.clf = GradientBoostingClassifier(**self.kwargs)
