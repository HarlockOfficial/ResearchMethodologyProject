import logging

from torch import nn

from my_logger import MyLogger


def model_summary(model: nn.Module, loglevel: int = logging.DEBUG):
    """
    Print the model summary.

    Args:
        model The model.
    """
    MyLogger().get_logger().log(loglevel, model)
