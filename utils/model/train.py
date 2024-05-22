import pandas as pd
import torch
from torch import nn

from my_logger import MyLogger


def fit(model: nn.Module, df: pd.DataFrame, optimizer:torch.optim.Optimizer, criterion, target_column_name:str, batch_size:int = 32, device: str = 'cpu') -> tuple[float, float]:
    """
    Fit the model.

    Args:
        model The model.
        df The dataset.
    """
    model.train()
    model.to(device).to(torch.float32)

    loss_list = []
    accuracy_list = []

    for i in range(0, len(df), batch_size):
        batch = df[i:i+batch_size]
        X = batch.drop(columns=[target_column_name])
        y = batch[target_column_name]
        X = torch.tensor(X.values).to(device).to(torch.float32)
        y = torch.tensor(y.values).to(device).to(torch.float32)
        optimizer.zero_grad()
        y_pred = model(X)
        y_pred = y_pred.squeeze()
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        accuracy = (y_pred.argmax() == y).sum().item() / len(y)
        accuracy_list.append(accuracy)
    return sum(loss_list) / len(loss_list), sum(accuracy_list) / len(accuracy_list)


def eval_model(model: nn.Module, df: pd.DataFrame, criterion, target_column_name: str, batch_size:int = 32, device: str = 'cpu'):
    """
    Evaluate the model.

    Args:
        model The model.
        df The dataset.
    """
    model.eval()
    model.to(device).to(torch.float32)

    loss_list = []
    accuracy_list = []

    for i in range(0, len(df), batch_size):
        batch = df[i:i+batch_size]
        X = batch.drop(columns=[target_column_name])
        y = batch[target_column_name]
        X = torch.tensor(X.values).to(device).to(torch.float32)
        y = torch.tensor(y.values).to(device).to(torch.float32)
        y_pred = model(X)
        y_pred = y_pred.squeeze()
        loss = criterion(y_pred, y)
        loss_list.append(loss.item())
        accuracy = (y_pred.argmax() == y).sum().item() / len(y)
        accuracy_list.append(accuracy)
    return sum(loss_list) / len(loss_list), sum(accuracy_list) / len(accuracy_list)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.functional.binary_cross_entropy
    for i in range(epochs):
        loss, accuracy = fit(model, dataset_train, optimizer, criterion, 'Outcome', batch_size, device)
        MyLogger().get_logger().info(f"epoch {i} loss {loss} accuracy {accuracy}")
        train_loss.append(loss)
        train_accuracy.append(accuracy)
    for df in dataset_test:
        loss, accuracy = eval_model(model, df, criterion, 'Outcome', batch_size, device)
        MyLogger().get_logger().info(f"test loss {loss} accuracy {accuracy}")
        test_loss.append(loss)
        test_accuracy.append(accuracy)

    return train_loss, train_accuracy, test_loss, test_accuracy
