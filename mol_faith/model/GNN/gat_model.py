from torch import nn
import torch_geometric.nn as geom_nn

from mol_faith.model.configs.dnn_configs import GATConfig
from mol_faith.model.GNN.base_gnn import BasicGNNRegressor


class GAT(BasicGNNRegressor):
    def __init__(self, config: GATConfig):
        super().__init__(config=config)

    def reset_parameters(self):
        super().reset_parameters()

    def _create_gnn_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        return geom_nn.GATv2Conv(in_channels=in_channels, out_channels=out_channels, heads=self.config.heads,
                                 concat=self.config.concat, negative_slope=self.config.negative_slope,
                                 dropout=self.config.dropout_gat_conv)

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.c_in <= 0:
            raise ValueError(f"Input channels must be positive, got {self.config.c_in}")
        if self.config.c_hidden <= 0:
            raise ValueError(f"Hidden channels must be positive, got {self.config.c_hidden}")
        if self.config.num_layers <= 0:
            raise ValueError(f"Number of layers must be positive, got {self.config.num_layers}")
        if self.config.c_hidden % self.config.heads != 0:
            raise ValueError(f"Parameter d_model ({self.config.c_hidden}) "
                             f"must be divisible by number of heads ({self.config.heads})")