import torch
from torch import nn

from mol_faith.data.data_managers.smiles_data_manager import SMILESTokenizer
from mol_faith.model.CNN.cnn_model import BasicCNNRegressor


class CaptumModelEmbeddingWrapper(nn.Module):
    """
    Wrapper that takes embedding inputs instead of raw node features.
    """

    def __init__(self, original_model: BasicCNNRegressor):
        super().__init__()
        self.model = original_model

    def forward(self, embeddings):
        x = embeddings

        # Pass through GNN layers
        x = self.model.apply_cnn_branches(x)

        x = self.model.apply_attention(x)

        x = self.model.head(x)

        return x.squeeze(-1)

    def encode_features(self, data: torch.Tensor):
        return self.model.encode_features(token_ids=data)

    def get_tokenizer(self) -> SMILESTokenizer:
        return self.model.tokenizer

    @staticmethod
    def get_model_type() -> str:
        return 'CNN'

class CaptumModelWrapper(nn.Module):
    """
    Wrapper that takes embedding inputs instead of raw node features.
    """

    def __init__(self, original_model: BasicCNNRegressor):
        super().__init__()
        self.model = original_model

    def forward(self, token_ids):
        assert token_ids.shape[0] == 1, "Dimension 0 of input should be 1"

        # Handle Captum's extra batch dimension - squeeze if present as in PyG it doesn't exist
        if token_ids.dim() == 3:
            token_ids = token_ids.squeeze(0)

        x = self.model(token_ids)

        return x.squeeze(-1)
