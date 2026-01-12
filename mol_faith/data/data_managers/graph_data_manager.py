import enum
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles
from torch_geometric.utils.smiles import x_map

from mol_faith.data.data_managers.data_manager import DataManager
from mol_faith.data.types import DataCategoryType

type FeatureProviders = PytorchGeometricFeatures


class SupportedFeatureProviders(str, enum.Enum):
    PYTORCH_GEOMETRIC = 'pytorch_geometric'


class PytorchGeometricFeatures:
    def __init__(self):
        self.num_node_features = 9
        self.num_edge_features = 3

    def __call__(self, smiles: str) -> Data:
        return from_smiles(smiles)


class GraphDataManager(DataManager):
    def __init__(self, data_path: str, batch_size: int, feature_provider: FeatureProviders,
                 scale_features: bool = False):
        super().__init__(data_path)
        self.feature_provider = feature_provider
        self.scale_features = scale_features
        self.batch_size = batch_size

    def load_data(self, target_column_name: str, data_category_type: str) -> DataLoader:
        data = pd.read_csv(os.path.join(self.data_path, f'{data_category_type}.csv'))

        # SMILES should be in the first column!
        graph_data = [self.feature_provider(smiles) for smiles in data.iloc[:, 0]]

        if self.scale_features:
            min_max_scaling_features(all_data=graph_data)

        add_label_to_graph_data(graph_data=graph_data, label=data[target_column_name])

        cast_data_to_float(graph_data=graph_data)

        if data_category_type == DataCategoryType.TRAIN.value:
            data_loader = DataLoader(graph_data, batch_size=self.batch_size, shuffle=True)
        else:
            data_loader = DataLoader(graph_data, batch_size=self.batch_size)

        return data_loader


def add_label_to_graph_data(graph_data: List[Data], label: pd.DataFrame) -> None:
    for idx, graph in enumerate(graph_data):
        # Make it shape (1, ) instead of 0 shape -> Compatible to model output
        graph.y = np.expand_dims(label.iloc[idx], axis=0)


def create_inverse_probs_sampler(df_data: pd.DataFrame) -> WeightedRandomSampler:
    binned_target = pd.cut(df_data.iloc[:, 1], bins=10, labels=False)
    sampling_probs = binned_target.value_counts(normalize=True)
    inverse_sampling_probs = 1 - sampling_probs
    weights = np.array([inverse_sampling_probs[bin_label] for bin_label in binned_target])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=False)
    return sampler


def cast_data_to_float(graph_data: List[Data]) -> None:
    for data in graph_data:
        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.y = torch.tensor(data.y).float()


def min_max_scaling_features(all_data: List[Data]):
    for data in all_data:
        data.x = data.x.to(torch.float64)
        for node in data.x:
            for idx, feature_values in enumerate(x_map.values()):
                if type(feature_values[0]) == int:
                    node[idx] = (node[idx] - np.min(feature_values)) / (np.max(feature_values) - np.min(feature_values))
                else:
                    indices = np.arange(len(feature_values))
                    node[idx] = (node[idx] - np.min(indices)) / (np.max(indices) - np.min(indices))


def create_graph_dataloader(path_to_data_folder: str, batch_size: int, split_num: int, data_split_type: str,
                            target_column_name: str, scale_features: Optional[bool] = False) -> DataLoader:
    path_to_data_folder = os.path.join(path_to_data_folder, f'split_{split_num}')
    data = pd.read_csv(os.path.join(path_to_data_folder, f'{data_split_type}.csv'))

    # SMILES should be in the first column!
    graph_data = [from_smiles(smiles) for smiles in data.iloc[:, 0]]

    if scale_features:
        min_max_scaling_features(all_data=graph_data)

    add_label_to_graph_data(graph_data=graph_data, label=data[target_column_name])

    cast_data_to_float(graph_data=graph_data)

    if data_split_type == DataCategoryType.TRAIN.value:
        data_loader = DataLoader(graph_data, batch_size=batch_size, shuffle=True)
    else:
        data_loader = DataLoader(graph_data, batch_size=batch_size)

    return data_loader
