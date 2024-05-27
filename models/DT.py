from sklearn.tree import DecisionTreeClassifier

from models.MachineLearningClassifier import MachineLearningClassifier


class DT(MachineLearningClassifier):
    def __init__(self, *, random_state=42, **kwargs):
        super().__init__(DecisionTreeClassifier(random_state=random_state))
