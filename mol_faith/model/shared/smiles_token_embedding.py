import torch
from torch import nn as nn
from typing import Literal


class SMILESTokenEmbedding(nn.Module):
    """Token embedding layer with masking support for SMILES sequences."""

    def __init__(self, vocab_size: int, embedding_dim: int, mask_token_id: int,
                 model_type: Literal["CNN", "Transformer"]):
        super().__init__()
        self._validate_parameters(vocab_size=vocab_size, embedding_dim=embedding_dim, mask_token_id=mask_token_id,
                                  model_type=model_type)
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mask_token_id = mask_token_id

    @staticmethod
    def _validate_parameters(vocab_size: int, embedding_dim: int, mask_token_id: int,
                             model_type: Literal["CNN", "Transformer"]) -> None:
        """Validate embedding parameters."""
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if not 0 <= mask_token_id < vocab_size:
            raise ValueError(f"mask_token_id must be in [0, vocab_size), got {mask_token_id}")
        if model_type not in ["CNN", "Transformer"]:
            raise ValueError('SMILESTokenizerEmbedding only supports CNN and Transformer models!')

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Token indices of shape [batch_size, sequence_length]
        Returns:
            Embedded tokens of shape [batch_size, embedding_dim, sequence_length]
        """
        # Direct embedding lookup
        embeddings = self.embedding(token_ids)

        if self.model_type == "CNN":
            # Transpose for conv1d: [B, L, E] -> [B, E, L]
            return embeddings.transpose(1, 2)
        else:
            # Keep sequence format: [B, L, E]
            return embeddings

    def get_mask_embedding(self) -> torch.Tensor:
        """Get the learned MASK token embedding."""
        return self.embedding.weight[self.mask_token_id]

    def reset_parameters(self) -> None:
        """Reset embedding parameters."""
        self.embedding.reset_parameters()
