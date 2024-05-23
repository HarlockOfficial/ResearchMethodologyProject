from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_validate

from models.AbstractClassifier import AbstractClassifier
from my_logger import MyLogger


class MachineLearningClassifier(AbstractClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return self.model.predict(x)

    def fit(self, df, *, criterion, target_column_name, **kwargs):
        x = df.drop(columns=[target_column_name])
        y = df[target_column_name].values
        scores = cross_validate(self.model, x, y, cv=10, return_estimator=True)
        self.model = scores['estimator'][-1]
        scores = scores['test_score']
        MyLogger().get_logger().debug(f"Train Step Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        y_pred = self(x)
        MyLogger().get_logger().debug(f'Train Dataset Prediction Accuracy: {accuracy_score(y, y_pred)}')
        MyLogger().get_logger().debug(f'Train Dataset Classification Report: {classification_report(y, y_pred)}')
        return -1, scores.mean()

    def evaluate(self, df, *, criterion, target_column_name, **kwargs):
        x = df.drop(columns=[target_column_name])
        y = df[target_column_name].values
        y_pred = self(x)
        accuracy = accuracy_score(y, y_pred)
        MyLogger().get_logger().debug(f'Test Dataset Prediction Accuracy: {accuracy}')
        self.model_report = classification_report(y, y_pred)
        MyLogger().get_logger().debug(f'Test Dataset Classification Report: {self.model_report}')
        return -1, accuracy
