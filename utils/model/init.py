from torch import nn

from models import ModelType


def init_model(selected_model: ModelType, **kwargs) -> nn.Module:
    """
    Initialize the model.

    Args:
        selected_model: str The selected model.

    Returns:
        nn.Module The model.
    """
    return selected_model.value(**kwargs)
