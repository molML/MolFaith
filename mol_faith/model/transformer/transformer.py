import math
import torch
import torch.nn as nn
from typing import Optional

from mol_faith.data.data_managers.smiles_data_manager import SMILESTokenizer
from mol_faith.model.configs.dnn_configs import TransformerConfig
from mol_faith.model.shared.prediction_head import PredictionHead
from mol_faith.model.shared.smiles_token_embedding import SMILESTokenEmbedding


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input sequences.
    From https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        # Shape [max_len, 1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # Shape [d_model//2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Replace d//2 values with the corresponding even/odd features
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, max_seq_len, d_model]
        Returns:
            Position-encoded tensor of same shape
        """
        # Broadcasting the batch dimension
        # [batch_size, max_seq_len, d_model] + [max_seq_len, d_model]
        return x + self.pe


class BasicTransformerRegressor(nn.Module):
    """Transformer regressor for SMILES with integrated token-level masking support."""

    def __init__(self, config: TransformerConfig, tokenizer: SMILESTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        assert len(self.tokenizer.token_to_id) > 0, 'Tokenizer was not loaded with vocabulary! Please load the vocab!'
        self._validate_config()

        self.token_embedding = self._build_token_embedding()
        self.pos_encoding = self._build_positional_encoding()
        self.encoder = self._build_transformer_encoder()
        self.head = self._build_prediction_head()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.d_model is None:
            raise ValueError(f"Parameter d_model is mandatory and needs to be set!")
        if self.config.dim_feedforward is None:
            raise ValueError(f"Parameter dim_feedforward is mandatory and needs to be set!")
        if self.config.d_model <= 0:
            raise ValueError(f"Parameter d_model must be positive, got {self.config.d_model}")
        if self.config.heads <= 0:
            raise ValueError(f"Parameter n_heads must be positive, got {self.config.heads}")
        if self.config.d_model % self.config.heads != 0:
            raise ValueError(f"Parameter d_model ({self.config.d_model}) must be divisible by n_heads ({self.config.heads})")
        if self.config.dim_feedforward <= 0:
            raise ValueError(f"Parameter dim_feedforward must be positive, got {self.config.dim_feedforward}")

    def _build_token_embedding(self) -> SMILESTokenEmbedding:
        """Build token embedding layer."""
        return SMILESTokenEmbedding(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.config.d_model,
            mask_token_id=self.tokenizer.mask_token_id,
            model_type="Transformer"
        )

    def _build_positional_encoding(self) -> PositionalEncoding:
        """Build positional encoding layer."""
        return PositionalEncoding(
            d_model=self.config.d_model,
            max_seq_len=self.config.max_seq_len
        )

    def _build_transformer_encoder(self) -> nn.TransformerEncoder:
        """Build transformer encoder."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dp_rate,
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)

    def _build_prediction_head(self) -> PredictionHead:
        """Build the prediction head."""
        return PredictionHead(
            d_in=self.config.d_model,
            n_out=1,
            hidden=self.config.c_hidden_head,
            dropout=self.config.dp_rate_head
        )

    @staticmethod
    def _validate_token_mask(token_mask: torch.Tensor, sequence_length: int) -> None:
        """
        Validate token mask tensor format and content.

        Args:
            token_mask: Binary mask tensor [batch_size, sequence_length]
            sequence_length: Expected sequence length
        Raises:
            ValueError: If token mask format is invalid
        """
        if not isinstance(token_mask, torch.Tensor):
            raise ValueError(f"token_mask must be a torch.Tensor, got {type(token_mask)}")

        if token_mask.dim() != 2:
            raise ValueError(f"token_mask must be 2D [batch_size, sequence_length], got shape {token_mask.shape}")

        if token_mask.size(1) != sequence_length:
            raise ValueError(f"token_mask sequence length {token_mask.size(1)} doesn't match input {sequence_length}")

        # Check if values are binary (0 or 1)
        unique_values = torch.unique(token_mask)
        if not torch.all((unique_values == 0) | (unique_values == 1)):
            raise ValueError(f"token_mask must contain only 0s and 1s, got unique values: {unique_values.tolist()}")

    def _apply_token_masking(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply token-level masking by replacing masked tokens with MASK token ID.

        Args:
            token_ids: Original token IDs [batch_size, sequence_length]
            token_mask: Binary mask tensor [batch_size, sequence_length] with 1=mask, 0=keep
        Returns:
            Token IDs with MASK tokens applied at masked positions
        """
        # Clone to avoid modifying original tensor
        masked_token_ids = token_ids.clone()

        # Replace masked positions with MASK token ID
        # token_mask == 1 means mask this token
        masked_token_ids[token_mask == 1] = self.tokenizer.mask_token_id

        return masked_token_ids

    def encode_features(self, token_ids: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode token IDs to embeddings with optional token masking.

        Args:
            token_ids: SMILES token indices of shape [batch_size, sequence_length]
            token_mask: Optional binary tensor for masking tokens
        Returns:
        Token embeddings of shape [batch_size, sequence_length, d_model]
        """
        # Apply token masking if provided
        if token_mask is not None:
            # Masker creates fore evaluation only 1D Tokens, for now even though it is ugly we create the batch dimension
            if token_mask.dim() == 1:
                token_mask = torch.unsqueeze(token_mask, 0)
            self._validate_token_mask(token_mask=token_mask, sequence_length=token_ids.size(1))
            token_ids = self._apply_token_masking(token_ids, token_mask)

        # Token embedding: [B, L] -> [B, L, E]
        x = self.token_embedding(token_ids)
        return x

    def apply_transformer_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer encoding to embeddings.

        Args:
            x: Token embeddings of shape [batch_size, sequence_length, d_model]
        Returns:
            Transformer encoded features of shape [batch_size, sequence_length, d_model]
        """
        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer processing: [B, L, E] -> [B, L, E]
        x = self.encoder(x)

        return x

    @staticmethod
    def apply_pooling(x: torch.Tensor) -> torch.Tensor:
        """
        Apply global pooling to transformer output.

        Args:
            x: Transformer output of shape [batch_size, sequence_length, d_model]
        Returns:
            Pooled features of shape [batch_size, d_model]
        """
        # Global average pooling: [B, L, E] -> [B, E]
        return x.mean(dim=1)

    def forward(self, token_ids: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the Transformer regressor.

        Args:
            token_ids: SMILES token indices of shape [batch_size, sequence_length]
            token_mask: Optional binary tensor of shape [batch_size, sequence_length]
                       indicating which tokens to mask (1=mask, 0=keep)
        Returns:
            Regression predictions of shape [batch_size]
        """
        # Encode features with optional token-level masking
        x = self.encode_features(token_ids=token_ids, token_mask=token_mask)

        # Apply transformer encoding
        x = self.apply_transformer_encoding(x)

        # Apply pooling
        x = self.apply_pooling(x)

        # Prediction head: [B, d_model] -> [B, 1]
        x = self.head(x)

        return x.squeeze(-1)

    def reset_parameters(self) -> None:
        """Reset all model parameters."""
        self.token_embedding.reset_parameters()

        # Reset transformer parameters
        for module in self.encoder.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

        # Reset prediction head parameters
        for module in self.head.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def save_params(self, path: str) -> None:
        """Save model parameters to disk."""
        torch.save(self.state_dict(), path)

    def load_params(self, path: str) -> None:
        """Load model parameters from disk."""
        state_dict = torch.load(path, weights_only=True)
        self.load_state_dict(state_dict, strict=False)
