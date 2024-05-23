from abc import ABC, abstractmethod

import pandas as pd
import torch
from sklearn.metrics import classification_report
import pathlib
import pickle

from torch import nn


class AbstractClassifier(nn.Module, ABC):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model_report = None

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame, optimizer: torch.optim.Optimizer, criterion, target_column_name:str, batch_size:int = 32, device: str = 'cpu', **kwargs):
        pass

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, criterion, target_column_name: str, batch_size:int = 32, device: str = 'cpu', **kwargs):
        pass

    def save(self, path):
        if not pathlib.Path(path).exists():
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        with open(f'{path}/model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

        with open(f'{path}/report.txt', 'w') as f:
            f.write(self.model_report)
