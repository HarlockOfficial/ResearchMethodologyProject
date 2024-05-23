from sklearn.ensemble import BaggingClassifier

from models.MachineLearningClassifier import MachineLearningClassifier


class BAG(MachineLearningClassifier):
    def __init__(self, **kwargs):
        super().__init__(BaggingClassifier(**kwargs))
