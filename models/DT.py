from sklearn.tree import DecisionTreeClassifier

from models.MachineLearningClassifier import MachineLearningClassifier


class DT(MachineLearningClassifier):
    def __init__(self, **kwargs):
        super().__init__(DecisionTreeClassifier(**kwargs))
