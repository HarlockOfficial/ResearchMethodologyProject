import enum

import pandas as pd

def zscore_outliers(dataset: pd.DataFrame, threshold: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
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

def iqr_outliers(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def handle_outliers(original_dataset: pd.DataFrame, method: OutlierHandlingMethod = OutlierHandlingMethod.IQR):

    ds = method(original_dataset)

    return ds
