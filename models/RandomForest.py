import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, cross_validate
from torch import nn

from my_logger import MyLogger


class RandomForest(nn.Module):
    def __init__(self, *, n_trees, max_depth, **kwargs):
        super(RandomForest, self).__init__()
        self.forest = RandomForestClassifier(n_trees, max_depth=max_depth, random_state=42)

    def forward(self, x):
        return self.forest.predict(x)

    def fit(self, df: pd.DataFrame, *, criterion, target_column_name:str, **kwargs):
        x = df.drop(columns=[target_column_name])
        y = df[target_column_name].values
        scores = cross_validate(self.forest, x, y, cv=10, return_estimator=True)
        self.forest = scores['estimator'][-1]
        scores = scores['test_score']
        MyLogger().get_logger().debug(f"Train Step Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
        y_pred = self(x)
        MyLogger().get_logger().debug(f'Train Dataset Prediction Accuracy: {accuracy_score(y, y_pred)}')
        MyLogger().get_logger().debug(f'Train Dataset Classification Report: {classification_report(y, y_pred)}')
        return -1, scores.mean()

    def evaluate(self, df: pd.DataFrame, *, criterion, target_column_name:str, **kwargs):
        x = df.drop(columns=[target_column_name])
        y = df[target_column_name].values
        y_pred = self(x)
        accuracy = accuracy_score(y, y_pred)
        MyLogger().get_logger().debug(f'Test Dataset Prediction Accuracy: {accuracy}')
        MyLogger().get_logger().debug(f'Test Dataset Classification Report: {classification_report(y, y_pred)}')
        return -1, accuracy
