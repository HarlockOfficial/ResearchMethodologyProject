import enum

import pandas as pd
from sklearn.preprocessing import StandardScaler

class ScalingMethod(enum.Enum):
    StandardScaler = lambda dataset: StandardScaler().fit_transform(*dataset)

def feature_scaling(dataset: list[list[list[pd.DataFrame]]], methods: list[ScalingMethod] = None) -> list[list[list[list[pd.DataFrame]]]]:
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
    for method in methods:
        output_datasets.append([])
        for dataset_lists in dataset:
            output_datasets[-1].append([])
            for dataset_list in dataset_lists:
                output_datasets[-1][-1].append([])
                for ds in dataset_list:
                    ds_X, ds_y = ds[ds.columns[:-1]], ds[ds.columns[-1]]
                    ds_X = method((ds_X, ds_y))
                    ds_X = pd.DataFrame(ds_X, columns=ds.columns[:-1])
                    ds = pd.concat([ds_X, ds_y], axis=1)
                    output_datasets[-1][-1][-1].append(ds)

    return output_datasets