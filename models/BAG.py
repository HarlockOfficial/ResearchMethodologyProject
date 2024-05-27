from sklearn.ensemble import BaggingClassifier

from models.MachineLearningClassifier import MachineLearningClassifier


class BAG(MachineLearningClassifier):
    def __init__(self, *, n_estimators=10, random_state=0, **kwargs):
        super().__init__(BaggingClassifier(n_estimators=n_estimators, random_state=random_state))
