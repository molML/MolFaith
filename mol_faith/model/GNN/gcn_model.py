from torch import nn
import torch_geometric.nn as geom_nn

from mol_faith.model.configs.dnn_configs import GCNConfig
from mol_faith.model.GNN.base_gnn import BasicGNNRegressor


class GCN(BasicGNNRegressor):
    def __init__(self, config: GCNConfig):
        super().__init__(config=config)

    @staticmethod
    def _create_gnn_layer(in_channels: int, out_channels: int) -> nn.Module:
        return geom_nn.GCNConv(in_channels=in_channels, out_channels=out_channels)

