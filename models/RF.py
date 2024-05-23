from models.AbstractClassifier import AbstractClassifier
from sklearn.ensemble import RandomForestClassifier


class RF(AbstractClassifier):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.create_classifier()

    def create_classifier(self):
        self.clf = RandomForestClassifier(**self.kwargs)
