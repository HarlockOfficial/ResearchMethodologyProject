import enum

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler

import pandas as pd

from utils import FunctionProxy


class ClassBalanceMethod(enum.Enum):
    SMOTETOMEK = FunctionProxy(lambda dataset: SMOTETomek(random_state=42).fit_resample(*dataset))
    RANDOMOVERSAMPLING = FunctionProxy(lambda dataset: RandomOverSampler(random_state=42).fit_resample(*dataset))

    def __call__(self, dataset):
        return self.value(dataset)


def class_balance(original_datasets: list[list[tuple[pd.DataFrame, str, str]]],
                  methods: list[ClassBalanceMethod] = None) -> list[list[tuple[pd.DataFrame, str, str]]]:
    """
    Balance the classes in the dataset.

    Args:
        original_datasets: list[pd.DataFrame] The dataset to balance.
        methods: list[ClassBalanceMethod] The methods to use in sequence on the data for balancing the dataset.

    Returns:
        pd.DataFrame: The dataset with balanced classes.
    """
    if methods is None:
        methods = [ClassBalanceMethod.RANDOMOVERSAMPLING, ClassBalanceMethod.SMOTETOMEK]

    output_datasets = []
    for dataset_list in original_datasets:
        output_datasets.append([])
        for dataset, imputation, outliers in dataset_list:
            ds_X, ds_y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
            for method in methods:
                ds_X, ds_y = method((ds_X, ds_y))
            ds = pd.concat([ds_X, ds_y], axis=1)
            output_datasets[-1].append((ds, imputation, outliers))
    return output_datasets
