import pathlib
import pickle
import sys

import pandas as pd
import torch.cuda
from matplotlib import pyplot as plt
from torch import nn

import utils
from models import ModelType
from my_logger import MyLogger


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

        for dataset_lists in dataset_train:
            for df, imputation, outlier in dataset_lists:
                path = base_path / imputation / outlier / 'train'
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                utils.dataset_serialize(df, str(path))

        for dataset_lists in dataset_test:
            for df, imputation, outlier in dataset_lists:
                path = base_path / imputation / outlier / 'test'
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                utils.dataset_serialize(df, str(path))

    dataset_train = []
    dataset_test = []
    for path in base_path.glob('*/*/train*'):
        dataset_train.append((utils.dataset_deserialize(str(path)), "/".join(str(path).split('/')[-3:-1])))
    for path in base_path.glob('*/*/test*'):
        dataset_test.append((utils.dataset_deserialize(str(path)), "/".join(str(path).split('/')[-3:-1])))
    return dataset_train, dataset_test


def main(dataset_path: str, selected_models: ModelType | list[ModelType], base_model_path:str=None) -> None:
    # TODO later use base_model_path to save and load models (to avoid retraining models every time)
    dataset_train, dataset_test = load_dataset(dataset_path)

    # Last column is the target value
    input_dim = dataset_train[0][0].shape[1] - 1
    output_dim = 1

    epochs_list = [100, 1000]
    for selected_model in selected_models:
        base_model_path = pathlib.Path(dataset_path).parent / 'models' / selected_model.name
        for epochs in epochs_list:
            MyLogger().get_logger().info(f'Training model {selected_model.name} for {epochs} epochs')
            for index, (df_train, df_path) in enumerate(dataset_train):
                model = utils.init_model(selected_model,
                    # Neural network params
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dims=[8, 16, 12, 8, 4],
                    activation='ReLU',
                    dropout=0.001,
                    # Machine Learning Models params
                    max_depth=None,
                    random_state=42,
                    n_estimators=100
                )
                train_model(df_train, dataset_test, model, base_model_path / df_path , epochs)

def train_model(dataset_train: pd.DataFrame, dataset_test: list[pd.DataFrame], model: nn.Module, base_model_path: pathlib.Path, epochs:int):
    utils.model_summary(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    train_loss, train_accuracy, test_loss, test_accuracy = utils.model_train(model, dataset_train, dataset_test,
                                                                             epochs=epochs, batch_size=32, device=device)
    model_path = base_model_path / f'{epochs}'
    MyLogger().get_logger().info(f'Saving model data and results to {model_path}')
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