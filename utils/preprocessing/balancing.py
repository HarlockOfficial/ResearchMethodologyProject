import enum

from imblearn.combine import SMOTETomek
import pandas as pd

class ClassBalanceMethod(enum.Enum):
    SMOTETOMEK = lambda dataset: SMOTETomek(random_state=42).fit_resample(*dataset)

def class_balance(original_datasets: list[list[pd.DataFrame]],
                  methods: list[ClassBalanceMethod] = None) -> list[list[list[pd.DataFrame]]]:
    """
    Balance the classes in the dataset.

    Args:
        original_datasets: list[pd.DataFrame] The dataset to balance.
        methods: list[ClassBalanceMethod] The methods to use for balancing the dataset.

    Returns:
        pd.DataFrame: The dataset with balanced classes.
    """
    if methods is None:
        methods = [ClassBalanceMethod.SMOTETOMEK]

    output_datasets = []
    for methods in methods:
        output_datasets.append([])
        for dataset_list in original_datasets:
            output_datasets[-1].append([])
            for dataset in dataset_list:
                ds_X, ds_y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
                ds_X, ds_y = methods((ds_X, ds_y))
                ds = pd.concat([ds_X, ds_y], axis=1)
                output_datasets[-1][-1].append(ds)
    return output_datasets
