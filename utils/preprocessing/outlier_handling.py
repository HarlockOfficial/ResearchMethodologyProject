import enum

import pandas as pd

def zscore_outliers(dataset: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """
    Handle outliers in the dataset using Z-score.

    Args:
        dataset: pd.DataFrame The dataset to handle outliers.
        threshold: float The threshold to use for Z-score.

    Returns:
        pd.DataFrame: The dataset with outliers handled.
    """
    z_scores = (dataset - dataset.mean()) / dataset.std()
    dataset = (dataset[(z_scores.abs() < threshold).all(axis=1)])
    return dataset

def iqr_outliers(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Handle outliers in the dataset using IQR.

    Args:
        dataset: pd.DataFrame The dataset to handle outliers.

    Returns:
        pd.DataFrame: The dataset with outliers handled.
    """
    quantiles = dataset.quantile([0.25, 0.75])
    iqr = quantiles.loc[0.75] - quantiles.loc[0.25]
    lower_bound = quantiles.loc[0.25] - 1.5 * iqr
    upper_bound = quantiles.loc[0.75] + 1.5 * iqr
    orig_df = dataset.copy()
    dataset = dataset[(orig_df >= lower_bound) & (orig_df <= upper_bound)]
    dataset = dataset.dropna()
    return dataset

class OutlierHandlingMethod(enum.Enum):
    # TODO: check threshold, is an hyperparameter
    ZSCORE = lambda dataset: zscore_outliers(dataset, 3.0)
    IQR = lambda dataset: iqr_outliers(dataset)


def handle_outliers(original_datasets: list[pd.DataFrame], methods: list[OutlierHandlingMethod] = None) -> list[list[pd.DataFrame]]:
    """
    Handle outliers in the dataset.

    Args:
        original_datasets: list[pd.DataFrame] The dataset to handle outliers.
        methods: list[OutlierHandlingMethod] The methods to use for handling outliers.

    Returns:
        pd.DataFrame: The dataset with outliers handled.
    """
    if methods is None:
        methods = [OutlierHandlingMethod.ZSCORE, OutlierHandlingMethod.IQR]
    datasets = []

    for method in methods:
        datasets.append([])
        for dataset in original_datasets:
            ds = method(dataset)
            datasets[-1].append(ds)
    # chose best solution
    return datasets
