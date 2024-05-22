import enum

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

def sklearn_value_imputation(strategy: str, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with mean value of the column

    Args
        dataset: pd.DataFrame: input dataset

    Returns
        pd.DataFrame: dataset with imputed missing values
    """
    imp = SimpleImputer(strategy=strategy)
    dataset_imputed = pd.DataFrame(imp.fit_transform(dataset), columns=dataset.columns)
    return dataset_imputed

def iterative_imputation(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with iterative imputation

    Args
        dataset: pd.DataFrame: input dataset

    Returns
        pd.DataFrame: dataset with imputed missing values
    """
    imp = IterativeImputer()
    dataset_imputed = pd.DataFrame(imp.fit_transform(dataset), columns=dataset.columns)
    return dataset_imputed

def knn_imputation(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with KNN imputation

    Args
        dataset: pd.DataFrame: input dataset

    Returns
        pd.DataFrame: dataset with imputed missing values
    """
    imp = KNNImputer()
    dataset_imputed = pd.DataFrame(imp.fit_transform(dataset), columns=dataset.columns)
    return dataset_imputed


class ImputationMethod(enum.Enum):
    MEAN = lambda dataset: sklearn_value_imputation('mean', dataset)
    MEDIAN = lambda dataset: sklearn_value_imputation('median', dataset)
    ITERATIVE = lambda dataset: iterative_imputation(dataset)
    KNN = lambda dataset: knn_imputation(dataset)


def compute_imputation(original_dataset: pd.DataFrame, imputation_methods:list[ImputationMethod] = None) -> list[pd.DataFrame]:
    """
    Compute imputation of missing values in the dataset

    Args
        dataset: pd.DataFrame: input dataset

    Returns
        pd.DataFrame: dataset with imputed missing values
    """
    if imputation_methods is None:
        imputation_methods = [ImputationMethod.MEDIAN]
    output_datasets = []
    for imputation_method in imputation_methods:
        dataset = imputation_method(original_dataset)
        output_datasets.append(dataset)

    return output_datasets
