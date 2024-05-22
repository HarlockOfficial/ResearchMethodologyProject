import logging

import pandas as pd

from my_logger import MyLogger


def describe_dataset(dataset: pd.DataFrame, loglevel: int = logging.DEBUG):
    """
    Describe the dataset
    
    Args:
        dataset: pd.DataFrame: The dataset to describe
        loglevel: str: The log level to use
    
    Returns:
        None
    """
    MyLogger().get_logger().log(loglevel, dataset.describe())
