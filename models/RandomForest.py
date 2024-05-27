from sklearn.ensemble import RandomForestClassifier

from models.MachineLearningClassifier import MachineLearningClassifier


class RandomForest(MachineLearningClassifier):
    def __init__(self, *, n_estimators=100, max_depth=None, random_state=42, **kwargs):
        super(RandomForest, self).__init__(RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state))
