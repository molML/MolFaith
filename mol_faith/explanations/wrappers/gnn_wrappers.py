import torch
from torch import nn
from torch_geometric.data import Data

from mol_faith.model.GNN.base_gnn import BasicGNNRegressor


class CaptumModelEmbeddingWrapper(nn.Module):
    """
    Wrapper that takes embedding inputs instead of raw node features.
    """

    def __init__(self, original_model: BasicGNNRegressor):
        super().__init__()
        self.model = original_model

    def forward(self, embeddings, edge_index, batch_index=None):
        assert embeddings.shape[0] == 1, "Dimension 0 of input should be 1"

        # Handle Captum's extra batch dimension - squeeze if present as in PyG it doesn't exist
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)

        x = embeddings

        # Pass through GNN layers
        x = self.model.apply_gnn_layers(x, edge_index, batch_index)

        # Global pooling
        x = self.model.pool(x, batch_index)

        # Prediction head
        x = self.model.head(x)

        return x.squeeze(-1)

    def encode_features(self, data: torch.Tensor):
        return self.model.encode_features(raw_features=data)


class CaptumModelWrapper(nn.Module):
    """
    This wrapper works with RAW inputs and creates embeddings in the forward pass.
    """

    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, node_features, edge_index, batch_index):
        assert node_features.shape[0] == 1, "Dimension 0 of input should be 1"

        # Handle Captum's extra batch dimension - squeeze if present as in PyG it doesn't exist
        if node_features.dim() == 3:
            node_features = node_features.squeeze(0)

        # Create Data object that the model expects
        data = Data(
            x=node_features,
            edge_index=edge_index,
            batch=batch_index
        )

        output = self.model(data)

        return output.squeeze(-1)


class PyGWrapper(nn.Module):
    """
    This wrapper works with RAW inputs and creates embeddings in the forward pass. It has specific forward args to be
    compatible with the internals of PyG
    """

    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, x, edge_index, batch_index):
        # Create Data object that the model expects
        data = Data(
            x=x,
            edge_index=edge_index,
            batch=batch_index
        )

        output = self.model(data)

        return output.squeeze(-1)


class PyGEmbeddingWrapper(nn.Module):
    """
    Wrapper that takes embedding inputs instead of raw node features.
    """

    def __init__(self, original_model: BasicGNNRegressor):
        super().__init__()
        self.model = original_model

    def forward(self, x, edge_index, batch_index=None):
        # Pass through GNN layers
        x = self.model.apply_gnn_layers(x, edge_index, batch_index)

        # Global pooling
        x = self.model.pool(x, batch_index)

        # Prediction head
        x = self.model.head(x)

        return x.squeeze(-1)

    def encode_features(self, data: torch.Tensor):
        return self.model.encode_features(raw_features=data)
