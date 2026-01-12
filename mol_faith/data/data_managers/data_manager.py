from abc import abstractmethod

from mol_faith.data.types import Data

class DataManager:
    def __init__(self, data_path: str):
        self.data_path = data_path

    @abstractmethod
    def load_data(self, target_column_name: str, data_split_type: str) -> Data:
        raise NotImplementedError
