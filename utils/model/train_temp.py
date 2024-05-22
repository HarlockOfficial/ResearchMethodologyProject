import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from my_logger import MyLogger


def train_one_epoch(model: nn.Module, training_loader: DataLoader, optimizer:torch.optim.Optimizer, criterion, device: str = 'cpu'):

    last_loss = 0.0

    correct_values_counter = 0

    for i, data in enumerate(training_loader):

        #X = data[:-1]
        #y = data[-1]

        #X = torch.tensor(X).to(device).to(torch.float32)
        #y = torch.tensor([y]).to(device).to(torch.float32)

        X, y = data

        X = X.to(device).to(torch.float32)
        y = y.to(device).to(torch.float32)

        optimizer.zero_grad()

        outputs = model(X)

        outputs = outputs.squeeze()

        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()

        if i == len(training_loader) - 1:
            last_loss = loss.item()

        for i in range(len(outputs)):
            if outputs[i] < 0.5:
                outputs[i] = 0
            else:
                outputs[i] = 1
            if y[i] == outputs[i]:
                correct_values_counter += 1

    return last_loss, correct_values_counter / len(training_loader.dataset)


def eval_model(model: nn.Module, testing_loader: DataLoader, criterion, device: str = 'cpu'):
    model.eval()
    model.to(device).to(torch.float32)

    correct_values_counter = 0

    last_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(testing_loader):
            #X = data[:-1]
            #y = data[-1]

            #X = torch.tensor(X).to(device).to(torch.float32)
            #y = torch.tensor([y]).to(device).to(torch.float32)

            X, y = data

            X = X.to(device).to(torch.float32)
            y = y.to(device).to(torch.float32)

            outputs = model(X)

            outputs = outputs.squeeze()

            loss = criterion(outputs, y)

            if i == len(testing_loader) - 1:
                last_loss = loss.item()

            for i in range(len(outputs)):
                if outputs[i] < 0.5:
                    outputs[i] = 0
                else:
                    outputs[i] = 1
                if y[i] == outputs[i]:
                    correct_values_counter += 1

    return last_loss, correct_values_counter / len(testing_loader.dataset)

def train(model: nn.Module, dataset_train: pd.DataFrame, dataset_test: pd.DataFrame, epochs:int = 10, batch_size:int = 32, device: str = 'cpu'):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.functional.binary_cross_entropy
    target_column_name = 'Outcome'
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []
    training_dataset = dataset_train.iloc[:, :-1]
    training_dataset = torch.utils.data.TensorDataset(torch.tensor(training_dataset.values).float(), torch.tensor(dataset_train[target_column_name].values).float())
    training_loader = DataLoader(training_dataset, batch_size=batch_size)
    testing_dataset = dataset_test.iloc[:, :-1]
    testing_dataset = torch.utils.data.TensorDataset(torch.tensor(testing_dataset.values).float(), torch.tensor(dataset_test[target_column_name].values).float())
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size)
    for epoch in range(epochs):
        model.train()
        #tr_l, tr_a = train_one_epoch(model, dataset_train, optimizer, criterion, batch_size, device)
        tr_l, tr_a = train_one_epoch(model, training_loader, optimizer, criterion, device)
        train_loss_list.append(tr_l)
        train_accuracy_list.append(tr_a)
        #te_l, te_a = eval_model(model, dataset_test, criterion, target_column_name, batch_size, device)
        te_l, te_a = eval_model(model, testing_loader, criterion, device)
        test_loss_list.append(te_l)
        test_accuracy_list.append(te_a)
        print(f'Epoch {epoch + 1} - Train loss: {tr_l:.3f} - Test loss: {te_l:.3f} - Train accuracy: {tr_a:.3f} - Test accuracy: {te_a:.3f}')

    return train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list
