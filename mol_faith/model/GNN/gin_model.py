from torch import nn
from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import GINConv

from mol_faith.model.configs.dnn_configs import GINConfig
from mol_faith.model.GNN.base_gnn import BasicGNNRegressor


class GIN(BasicGNNRegressor):
    def __init__(self, config: GINConfig):
        super().__init__(config=config)

    def _create_gnn_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        mlp = self.create_mlp(in_channels=in_channels, out_channels=out_channels)
        return GINConv(mlp)

    def create_mlp(self, in_channels: int, out_channels: int):
        mlp = MLP(
            # According to pytorch geometric implementation
            [in_channels, out_channels, out_channels],
            act=self.config.mlp_act,
            act_first=self.config.mlp_act_first,
            norm=self.config.mlp_norm,
            norm_kwargs=self.config.mlp_norm_kwargs,
        )
        return mlp