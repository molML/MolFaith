from enum import Enum
from typing import Tuple

import numpy as np
from torch.utils.data import  DataLoader


type Data = DataLoader | Tuple[np.array, np.array]

# Todo: Only use DataSplitType instead of DataCSVFileNames -> Makes it more consistent
class DataCategoryType(str, Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class DataCSVFileNames(str, Enum):
    TRAIN = 'train.csv'
    VAL = 'val.csv'
    TEST = 'test.csv'


class SupportedFingerprints(str, Enum):
    ECFP = 'ecfp'
    FCFP = 'fcfp'
    COUNT_BASED = 'count_based'
    SORT_AND_SLICE = 'sort_and_slice'
