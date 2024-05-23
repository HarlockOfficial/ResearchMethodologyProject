from sklearn.ensemble import BaggingClassifier
from models.AbstractClassifier import AbstractClassifier


class BAG(AbstractClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_classifier()

    def create_classifier(self):
        self.clf = BaggingClassifier(**self.kwargs)
