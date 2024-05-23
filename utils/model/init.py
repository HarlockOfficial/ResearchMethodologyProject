from models import ModelType


def init_model(selected_model: ModelType, **kwargs):
    return selected_model.value(**kwargs)
