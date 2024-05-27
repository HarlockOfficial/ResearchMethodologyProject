import enum

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

from utils import FunctionProxy


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
    MEAN = FunctionProxy(lambda dataset: sklearn_value_imputation('mean', dataset))
    MEDIAN = FunctionProxy(lambda dataset: sklearn_value_imputation('median', dataset))
    ITERATIVE = FunctionProxy(lambda dataset: iterative_imputation(dataset))
    KNN = FunctionProxy(lambda dataset: knn_imputation(dataset))

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)



def compute_imputation(original_dataset: pd.DataFrame, imputation_methods:list[ImputationMethod] = None) -> list[tuple[pd.DataFrame, str]]:
    """
    Compute imputation of missing values in the dataset

    Args
        dataset: pd.DataFrame: input dataset

    Returns
        pd.DataFrame: dataset with imputed missing values
    """
    if imputation_methods is None:
        imputation_methods = [ImputationMethod.MEDIAN, ImputationMethod.MEAN, ImputationMethod.ITERATIVE, ImputationMethod.KNN]
    output_datasets = []
    for imputation_method in imputation_methods:
        dataset = imputation_method(original_dataset)
        output_datasets.append((dataset, imputation_method.name))

    return output_datasets
