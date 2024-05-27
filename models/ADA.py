from sklearn.ensemble import AdaBoostClassifier

from models.MachineLearningClassifier import MachineLearningClassifier


class ADA(MachineLearningClassifier):
    def __init__(self, *, algorithm='SAMME', n_estimators=100, random_state=42, **kwargs):
        super().__init__(AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state, algorithm=algorithm))
