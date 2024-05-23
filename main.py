import pathlib
import pickle
import sys

import torch.cuda
from matplotlib import pyplot as plt

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

    dataset = utils.dataset_load(dataset_path)
    utils.dataset_describe(dataset)

    return utils.dataset_preprocess(dataset)


def train_mlp(dataset_train, dataset_test, model, base_model_path, epochs):
    utils.model_summary(model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    train_loss, train_accuracy, test_loss, test_accuracy = utils.model_train(model,
                                                                             dataset_train.reset_index(drop=True),
                                                                             dataset_test.reset_index(drop=True),
                                                                             epochs=epochs, batch_size=16,
                                                                             device=device)

    model_path = base_model_path / f'{epochs}'
    if not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)
    utils.model_save(model, f'{model_path}/model')

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


def train_ml(dataset_train, dataset_test, model, base_model_path, epochs):
    labels_column = 'Outcome'
    model.fit(dataset_train.drop(columns=labels_column), dataset_train[labels_column])
    model.evaluate(dataset_test.drop(columns=labels_column), dataset_test[labels_column])
    path = f'{base_model_path}/{epochs}/model'
    model.save(path)


def main(dataset_path: str, selected_model: ModelType, base_model_path: str = None):
    if base_model_path is None:
        base_model_path = pathlib.Path(dataset_path).parent / 'models' / selected_model.name

    dataset_train, dataset_test = load_dataset(dataset_path)

    input_dim = dataset_train.shape[1] - 1
    output_dim = 1

    epochs_list = [100, 1000]
    for epochs in epochs_list:
        # train gradient booster
        if selected_model == ModelType.Mlp:
            model = utils.init_model(selected_model,
                                     input_dim=input_dim,
                                     output_dim=output_dim,
                                     hidden_dims=[8, 16, 12, 8, 4],
                                     activation='ReLU',
                                     dropout=0.001
                                     )
            train_mlp(dataset_train, dataset_test, model, base_model_path, epochs)
        else:
            # TODO: tune hyperparameters
            model = utils.init_model(selected_model)
            train_ml(dataset_train, dataset_test, model, base_model_path, epochs)


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    selected_model = sys.argv[2]
    selected_model = ModelType[selected_model.title()]
    main(dataset_path, selected_model)
