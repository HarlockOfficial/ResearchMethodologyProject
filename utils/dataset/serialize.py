import pickle

import pandas as pd


def serialize_dataset(dataset: pd.DataFrame, file_name: str) -> None:
    """
    Save the dataset to a file.

    Args:
        dataset: tuple[pd.DataFrame, pd.DataFrame] The dataset to save.
        file_name: str The type of the dataset.
    """
    if not file_name.endswith(".pkl"):
        file_name = f"{file_name}.pkl"
    with open (f"{file_name}", "wb") as f:
        pickle.dump(dataset, f)

def deserialize_dataset(file_name: str) -> pd.DataFrame:
    """
    Load the dataset from a file.

    Args:
        file_name: str The path to the dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame] The dataset.
    """
    if not file_name.endswith(".pkl"):
        file_name = f"{file_name}.pkl"
    with open(f"{file_name}", "rb") as f:
        return pickle.load(f)
