import pandas as pd

from sklearn.model_selection import train_test_split


def split_dataset(dataset: list[tuple[pd.DataFrame, str]], train_percentage: float) \
        -> tuple[list[tuple[pd.DataFrame, str]], list[tuple[pd.DataFrame, str]]]:
    """
    Split the dataset into training and testing datasets.

    Args
        dataset: list of dataframes The dataset to split.
        train_percentage: float The percentage of the dataset to use for training.

    Returns
        list of dataframes: The training and testing datasets.
    """
    assert 0 < train_percentage < 1, "train_percentage must be between 0 and 1"

    ds_trains, ds_tests = [], []
    for ds, imputation_method_name in dataset:
        ds_train, ds_test = train_test_split(ds, train_size=train_percentage)
        ds_trains.append((ds_train, imputation_method_name))
        ds_tests.append((ds_test, imputation_method_name))

    return ds_trains, ds_tests
