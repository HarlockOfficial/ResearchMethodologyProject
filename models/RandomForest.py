from sklearn.ensemble import RandomForestClassifier

from models.MachineLearningClassifier import MachineLearningClassifier


class RandomForest(MachineLearningClassifier):
    def __init__(self, **kwargs):
        super(RandomForest, self).__init__(RandomForestClassifier(**kwargs))
