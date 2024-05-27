from sklearn.linear_model import LogisticRegression

from models.MachineLearningClassifier import MachineLearningClassifier


class LR(MachineLearningClassifier):
    def __init__(self, *, random_state=42, **kwargs):
        super().__init__(LogisticRegression(random_state=random_state))
