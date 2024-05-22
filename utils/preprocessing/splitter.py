import pandas as pd

from sklearn.model_selection import train_test_split


def split_dataset(dataset: pd.DataFrame, train_percentage: float):

    assert 0 < train_percentage < 1, "train_percentage must be between 0 and 1"

    ds_train, ds_test = train_test_split(dataset, train_size=train_percentage, random_state=42)

    return ds_train, ds_test
