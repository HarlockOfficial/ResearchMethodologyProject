import enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import FunctionProxy


class ScalingMethod(enum.Enum):
    StandardScaler = FunctionProxy(lambda dataset: StandardScaler().fit_transform(dataset))

    def __call__(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return self.value(dataset)


def feature_scaling(dataset: list[list[tuple[pd.DataFrame, str, str]]], methods: list[ScalingMethod] = None) -> list[list[tuple[pd.DataFrame, str, str]]]:
    """
    Scale the features in the dataset.

    Args:
        dataset: list[pd.DataFrame] The dataset to scale.
        methods: list[ScalingMethod] The methods to use for scaling the dataset.

    Returns:
        pd.DataFrame: The dataset with scaled features.
    """
    if methods is None:
        methods = [ScalingMethod.StandardScaler]

    output_datasets = []
    assert len(methods) == 1, "Only one scaling method is supported"
    for method in methods:
        for dataset_list in dataset:
            output_datasets.append([])
            for ds, imputation, outliers in dataset_list:
                columns = ds.columns
                ds_X, ds_y = ds[ds.columns[:-1]], ds[ds.columns[-1]]
                ds_X = method(ds_X)
                ds_y = ds_y.to_numpy().reshape(-1, 1)
                np_arr = np.hstack((ds_X, ds_y))
                ds = pd.DataFrame(np_arr, columns=columns)
                output_datasets[-1].append((ds, imputation, outliers))

    return output_datasets