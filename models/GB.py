from sklearn.ensemble import GradientBoostingClassifier

from models.MachineLearningClassifier import MachineLearningClassifier


class GB(MachineLearningClassifier):
    def __init__(self, **kwargs):
        super().__init__(GradientBoostingClassifier(**kwargs))
