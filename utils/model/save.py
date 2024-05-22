import pickle

import torch
from torch import nn


def model_save(model: nn.Module, file_name: str) -> None:
    """
    Save the model to a file.

    Args:
        model: nn.Module The model to save.
        file_name: str The file name.
    """
    torch.save(model.state_dict(), f"{file_name}.pt")

    with open(f"{file_name}.pkl", "wb") as f:
        pickle.dump(model, f)
