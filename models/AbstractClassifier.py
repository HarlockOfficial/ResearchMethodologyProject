from abc import ABC, abstractmethod
from sklearn.metrics import classification_report
import pathlib
import pickle


class AbstractClassifier(ABC):
    def __init__(self):
        self.clf = None
        self.clf_report = None

    @abstractmethod
    def create_classifier(self):
        pass

    def fit(self, X, y):
        self.clf.fit(X, y)

    def evaluate(self, X, y):
        y_pred = self.clf.predict(X)
        self.clf_report = classification_report(y, y_pred)

    def save(self, path):
        if not pathlib.Path(path).exists():
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        with open(f'{path}/model.pkl', 'wb') as f:
            pickle.dump(self.clf, f)

        with open(f'{path}/report.txt', 'w') as f:
            f.write(self.clf_report)
