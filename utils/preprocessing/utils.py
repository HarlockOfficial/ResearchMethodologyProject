import pandas as pd

from utils.preprocessing import compute_imputation, handle_outliers, split_dataset, class_balance, feature_scaling


def dataset_preprocess(dataset: pd.DataFrame) -> tuple[list, list]:
    dataset = compute_imputation(dataset)
    dataset_train, dataset_test = split_dataset(dataset, 0.8)

    dataset_train = handle_outliers(dataset_train)
    dataset_test = handle_outliers(dataset_test)

    dataset_train = class_balance(dataset_train)

    dataset_train = feature_scaling(dataset_train)
    dataset_test = feature_scaling(dataset_test)
    return dataset_train, dataset_test
