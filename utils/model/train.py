import pandas as pd
import torch
from torch import nn

from my_logger import MyLogger


def eval_model(model: nn.Module, df: pd.DataFrame, criterion, target_column_name: str, batch_size:int = 32, device: str = 'cpu'):
    """
    Evaluate the model.

    Args:
        model The model.
        df The dataset.
    """
    return model.evaluate(df, criterion, target_column_name, batch_size, device)

def train(model: nn.Module, dataset_train: pd.DataFrame, dataset_test: list[pd.DataFrame], epochs:int = 10, batch_size:int = 32, device: str = 'cpu') -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Train the model.

    Args:
        model The model.

        dataset_train The training dataset.
        dataset_test The testing dataset.
    """
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    optimizer = None
    if len(list(model.parameters())) > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.functional.binary_cross_entropy
    for i in range(epochs):
        loss, accuracy = model.fit(df=dataset_train, optimizer=optimizer, criterion=criterion, target_column_name='Outcome', batch_size=batch_size, device=device)
        MyLogger().get_logger().info(f"epoch {i+1} loss {loss} accuracy {accuracy}")
        train_loss.append(loss)
        train_accuracy.append(accuracy)
    for df in dataset_test:
        loss, accuracy = model.evaluate(df=df, criterion=criterion, target_column_name='Outcome', batch_size=batch_size, device=device)
        MyLogger().get_logger().info(f"test loss {loss} accuracy {accuracy}")
        test_loss.append(loss)
        test_accuracy.append(accuracy)

    return train_loss, train_accuracy, test_loss, test_accuracy
