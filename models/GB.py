from sklearn.ensemble import GradientBoostingClassifier

from models.MachineLearningClassifier import MachineLearningClassifier


class GB(MachineLearningClassifier):
    def __init__(self, *, n_estimators=100, learning_rate=1.0, max_depth=None, random_state=42, **kwargs):
        super().__init__(GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state))
