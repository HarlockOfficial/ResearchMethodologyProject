from sklearn.linear_model import LogisticRegression

from models.MachineLearningClassifier import MachineLearningClassifier


class LR(MachineLearningClassifier):
    def __init__(self, **kwargs):
        super().__init__(LogisticRegression(**kwargs))
