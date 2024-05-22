import pandas as pd
import torch
import torch.nn as nn

class MLP(nn.Sequential):
    def __init__(self, *, input_dim:int, output_dim:int, hidden_dims:list[int], activation:str='ReLU', dropout:float=0.0, **kwargs):
        super(MLP, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(1, len(dims)):
            self.add_module(f'Linear_{i}', nn.Linear(dims[i-1], dims[i]))
            if i < len(dims)-1:
                self.add_module(f'Activation_{i}', getattr(nn, activation)())
                self.add_module(f'Dropout_{i}', nn.Dropout(dropout))
        self.add_module('Sigmoid', nn.Sigmoid())

    def fit(self,  df: pd.DataFrame, optimizer:torch.optim.Optimizer, criterion, target_column_name:str, batch_size:int = 32, device: str = 'cpu', **kwargs):
        self.train()
        self.to(device).to(torch.float32)

        loss_list = []
        accuracy_list = []

        for i in range(0, len(df), batch_size):
            batch = df[i:i + batch_size]
            X = batch.drop(columns=[target_column_name])
            y = batch[target_column_name]
            X = torch.tensor(X.values).to(device).to(torch.float32)
            y = torch.tensor(y.values).to(device).to(torch.float32)
            optimizer.zero_grad()
            y_pred = self(X)
            y_pred = y_pred.squeeze()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            accuracy = (y_pred.argmax() == y).sum().item() / len(y)
            accuracy_list.append(accuracy)
        return sum(loss_list) / len(loss_list), sum(accuracy_list) / len(accuracy_list)

    def evaluate(self, df: pd.DataFrame, criterion, target_column_name: str, batch_size:int = 32, device: str = 'cpu', **kwargs):
        """
            Evaluate the model.

            Args:
                model The model.
                df The dataset.
            """
        self.eval()
        self.to(device).to(torch.float32)

        loss_list = []
        accuracy_list = []

        for i in range(0, len(df), batch_size):
            batch = df[i:i + batch_size]
            X = batch.drop(columns=[target_column_name])
            y = batch[target_column_name]
            X = torch.tensor(X.values).to(device).to(torch.float32)
            y = torch.tensor(y.values).to(device).to(torch.float32)
            y_pred = self(X)
            y_pred = y_pred.squeeze()
            loss = criterion(y_pred, y)
            loss_list.append(loss.item())
            accuracy = (y_pred.argmax() == y).sum().item() / len(y)
            accuracy_list.append(accuracy)
        return sum(loss_list) / len(loss_list), sum(accuracy_list) / len(accuracy_list)
