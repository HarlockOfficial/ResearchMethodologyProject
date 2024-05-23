import pathlib
import pickle
import sys

import pandas as pd
import torch.cuda
from matplotlib import pyplot as plt
from torch import nn

import utils
from models import ModelType


def load_dataset(dataset_path: str) -> tuple:
    """
    Load the dataset from a file.

    Args:
        dataset_path: str The path to the dataset.

    Returns:
        tuple The dataset.
    """
    base_path = pathlib.Path(dataset_path).parent / 'processed'

    if not base_path.exists():
        dataset = utils.dataset_load(dataset_path)
        utils.dataset_describe(dataset)

        dataset_train, dataset_test = utils.dataset_preprocess(dataset)
        index = 0
        for dataset_lists_lists in dataset_train:
            for dataset_lists in dataset_lists_lists:
                for datasets in dataset_lists:
                    for df in datasets:
                        path = base_path / f'train{index}'
                        if not path.exists():
                            path.parent.mkdir(parents=True, exist_ok=True)
                        utils.dataset_serialize(df, str(path))
                        index += 1

        index = 0
        for dataset_lists_lists in dataset_test:
            for dataset_lists in dataset_lists_lists:
                for datasets in dataset_lists:
                    for df in datasets:
                        path = base_path / f'test{index}'
                        if not path.exists():
                            path.parent.mkdir(parents=True, exist_ok=True)
                        utils.dataset_serialize(df, str(path))
                        index += 1

    dataset_train = []
    dataset_test = []
    for path in base_path.glob('train*'):
        dataset_train.append(utils.dataset_deserialize(str(path)))
    for path in base_path.glob('test*'):
        dataset_test.append(utils.dataset_deserialize(str(path)))
    return dataset_train, dataset_test


def main(dataset_path: str, selected_model: ModelType, base_model_path:str=None) -> None:
    if base_model_path is None:
        base_model_path = pathlib.Path(dataset_path).parent / 'models' / selected_model.name

    dataset_train, dataset_test = load_dataset(dataset_path)

    # Last column is the target value
    input_dim = dataset_train[0].shape[1] - 1
    output_dim = 1

    epochs_list = [100, 1000, 2500, 5000, 10000]
    for epochs in epochs_list:
        for index, df_train in enumerate(dataset_train):
            model = utils.init_model(selected_model,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=[8, 16, 12, 8, 4],
                activation='ReLU',
                dropout=0.001,
                n_trees=100,
                max_depth=None
            )
            train_model(df_train, dataset_test, model, base_model_path, epochs)

def train_model(dataset_train: pd.DataFrame, dataset_test: list[pd.DataFrame], model: nn.Module, base_model_path: pathlib.Path, epochs:int):
    utils.model_summary(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    train_loss, train_accuracy, test_loss, test_accuracy = utils.model_train(model, dataset_train, dataset_test,
                                                                             epochs=epochs, batch_size=32, device=device)
    model_path = base_model_path / f'{epochs}'
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
    utils.model_save(model, f'{model_path}/model')

    with open(f'{model_path}/train_loss.pkl', 'wb') as f:
        pickle.dump(train_loss, f)
    with open(f'{model_path}/train_accuracy.pkl', 'wb') as f:
        pickle.dump(train_accuracy, f)
    with open(f'{model_path}/test_loss.pkl', 'wb') as f:
        pickle.dump(test_loss, f)
    with open(f'{model_path}/test_accuracy.pkl', 'wb') as f:
        pickle.dump(test_accuracy, f)

    plt.figure()
    plt.plot(train_loss, label='train loss')
    plt.plot(train_accuracy, label='train accuracy')
    plt.legend()
    plt.savefig(f'{model_path}/train.png')

    plt.figure()
    plt.plot(test_loss, label='test loss')
    plt.plot(test_accuracy, label='test accuracy')
    plt.legend()
    plt.savefig(f'{model_path}/test.png')


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    selected_model = sys.argv[2]
    selected_model = ModelType[selected_model.title()]
    main(dataset_path, selected_model)