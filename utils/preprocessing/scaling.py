import enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class ScalingMethod(enum.Enum):
    StandardScaler = lambda dataset: StandardScaler().fit_transform(dataset)


def feature_scaling(dataset: pd.DataFrame, method: ScalingMethod = ScalingMethod.StandardScaler):

    columns = dataset.columns
    ds_X, ds_y = dataset[dataset.columns[:-1]], dataset[dataset.columns[-1]]
    ds_X = method(ds_X)
    ds_y = ds_y.to_numpy().reshape(-1, 1)
    np_arr = np.hstack((ds_X, ds_y))
    ds = pd.DataFrame(np_arr, columns=columns)

    return ds