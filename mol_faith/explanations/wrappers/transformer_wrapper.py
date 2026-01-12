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

        # Apply transformer encoding
        x = self.model.apply_transformer_encoding(x)

        # Apply pooling
        x = self.model.apply_pooling(x)

        # Prediction head: [B, d_model] -> [B, 1]
        x = self.model.head(x)

        return x.squeeze(-1)

    def encode_features(self, data: torch.Tensor):
        return self.model.encode_features(token_ids=data)

    def get_tokenizer(self) -> SMILESTokenizer:
        return self.model.tokenizer

    @staticmethod
    def get_model_type() -> str:
        return 'Transformer'
