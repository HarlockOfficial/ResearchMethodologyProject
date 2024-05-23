from sklearn.ensemble import AdaBoostClassifier

from models.MachineLearningClassifier import MachineLearningClassifier


class ADA(MachineLearningClassifier):
    def __init__(self, **kwargs):
        super().__init__(AdaBoostClassifier(**kwargs))
