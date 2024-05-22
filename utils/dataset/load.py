import pandas as pd


def load_dataset(dataset_path):
    """
    Load dataset from a given path

    Args:
        dataset_path (str): path to csv dataset file

    Returns:
        pandas.DataFrame: dataset
    """
    return pd.read_csv(dataset_path)
