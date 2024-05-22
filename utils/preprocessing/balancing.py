import enum

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler

import pandas as pd

class ClassBalanceMethod(enum.Enum):
    SMOTETOMEK = lambda dataset: SMOTETomek(random_state=42).fit_resample(*dataset)
    RANDOMOVERSAMPLING = lambda dataset: RandomOverSampler(random_state=42).fit_resample(*dataset)

def class_balance(original_dataset: pd.DataFrame):

    ds_X, ds_y = original_dataset.iloc[:, :-1], original_dataset.iloc[:, -1]
    ds_X, ds_y = ClassBalanceMethod.RANDOMOVERSAMPLING((ds_X, ds_y))
    ds_X, ds_y = ClassBalanceMethod.SMOTETOMEK((ds_X, ds_y))
    ds = pd.concat([ds_X, ds_y], axis=1)

    return ds
