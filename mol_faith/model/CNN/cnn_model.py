import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from mol_faith.data.data_managers.smiles_data_manager import SMILESTokenizer
from mol_faith.model.configs.dnn_configs import CNNConfig
from mol_faith.model.shared.prediction_head import PredictionHead
from mol_faith.model.shared.smiles_token_embedding import SMILESTokenEmbedding


class SqueezeExcitationBlock(nn.Module):
    """Squeeze-and-Excitation block for 1D feature maps."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = min(reduction, channels)

        if self.reduction < 1:
            self.reduction = 1

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = self._build_fc_layers()

    def _build_fc_layers(self) -> nn.Sequential:
        """Build fully connected layers for squeeze-excitation."""
        reduced_channels = max(1, self.channels // self.reduction)
        return nn.Sequential(
            nn.Linear(self.channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, self.channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, channels, sequence_length]
        Returns:
            Channel-reweighted tensor of same shape
        """
        batch_size, channels, _ = x.shape

        # Global average pooling: [B, C, L] -> [B, C, 1] -> [B, C]
        weights = self.global_pool(x).view(batch_size, channels)

        # Squeeze-and-excitation: [B, C] -> [B, C] -> [B, C, 1]
        weights = self.fc(weights).unsqueeze(-1)

        # Apply channel attention
        return x * weights

    def reset_parameters(self) -> None:
        """Reset all parameters in the block."""
        for module in self.fc.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()


class ConvolutionBranch(nn.Module):
    """Single convolution branch with batch normalization and global max pooling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dropout_rate: float = 0.0):
        super().__init__()
        self._validate_parameters(in_channels, out_channels, kernel_size, dropout_rate)

        self.conv_block = self._build_conv_block(in_channels, out_channels, kernel_size)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    @staticmethod
    def _validate_parameters(in_channels: int, out_channels: int,
                             kernel_size: int, dropout_rate: float) -> None:
        """Validate branch parameters."""
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

    @staticmethod
    def _build_conv_block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
        """Build convolution block with normalization and activation."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, in_channels, sequence_length]
        Returns:
            Global max-pooled features of shape [batch_size, out_channels]
        """
        # Apply convolution: [B, in_ch, L] -> [B, out_ch, L]
        x = self.conv_block(x)

        # Apply dropout after convolution
        if self.dropout is not None:
            x = self.dropout(x)

        # Global max pooling: [B, out_ch, L] -> [B, out_ch]
        return F.max_pool1d(x, x.size(-1)).squeeze(-1)

    def reset_parameters(self) -> None:
        """Reset all parameters in the branch."""
        for module in self.conv_block.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()


class BasicCNNRegressor(nn.Module):
    """CNN regressor for SMILES with integrated token-level masking support."""

    def __init__(self, config: CNNConfig, tokenizer: SMILESTokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        assert len(self.tokenizer.token_to_id) > 0, 'Tokenizer was not loaded with vocabulary!'
        self._validate_config()

        # Core model components
        self.token_embedding = self._build_token_embedding()
        self.encoder = self._build_encoder()
        self.attention = self._build_attention() if self._use_attention() else None
        self.head = self._build_prediction_head()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.embedding_dim <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {self.config.embedding_dim}")
        if not self.config.kernel_sizes:
            raise ValueError("At least one kernel size must be specified")
        if self.config.c_hidden <= 0:
            raise ValueError(f"Filters per kernel must be positive, got {self.config.c_hidden}")

    def _use_attention(self) -> bool:
        """Check if squeeze-excitation attention should be used."""
        return getattr(self.config, 'use_se_attention', False)

    def _build_token_embedding(self) -> SMILESTokenEmbedding:
        """Build token embedding layer."""
        return SMILESTokenEmbedding(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.config.embedding_dim,
            mask_token_id=self.tokenizer.mask_token_id,
            model_type="CNN"
        )

    def _build_encoder(self) -> nn.ModuleList:
        """Build convolution branches for different kernel sizes."""
        branches = nn.ModuleList()

        for kernel_size in self.config.kernel_sizes:
            branch = ConvolutionBranch(in_channels=self.config.embedding_dim,
                                       out_channels=self.config.c_hidden,
                                       kernel_size=kernel_size,
                                       dropout_rate=self.config.dp_rate)
            branches.append(branch)

        return branches

    def _build_attention(self) -> SqueezeExcitationBlock:
        """Build attention mechanism."""
        total_features = len(self.config.kernel_sizes) * self.config.c_hidden
        return SqueezeExcitationBlock(
            channels=total_features,
            reduction=getattr(self.config, 'se_reduction', 16)
        )

    def _build_prediction_head(self) -> PredictionHead:
        """Build the prediction head."""
        total_features = len(self.config.kernel_sizes) * self.config.c_hidden
        return PredictionHead(d_in=total_features,
                              n_out=1,
                              hidden=self.config.c_hidden_head,
                              dropout=self.config.dp_rate_head)

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
        # If token_mask does not include any tokens (edge case) nothing will happen and we us the normal output
        masked_token_ids[token_mask == 1] = self.tokenizer.mask_token_id

        return masked_token_ids

    def encode_features(self, token_ids: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode token IDs to embeddings with optional token masking.
        Args:
            token_ids: SMILES token indices of shape [batch_size, sequence_length]
            token_mask: Optional binary tensor for masking tokens
        Returns:
            Token embeddings of shape [batch_size, embedding_dim, sequence_length]
        """
        # Apply token masking if provided
        if token_mask is not None:
            # Masker creates fore evaluation only 1D Tokens, for now even though it is ugly we create the batch dimension
            if token_mask.dim() == 1:
                token_mask = torch.unsqueeze(token_mask, 0)
            self._validate_token_mask(token_mask, token_ids.size(1))
            token_ids = self._apply_token_masking(token_ids, token_mask)

        # Token embedding: [B, L] -> [B, E, L]
        x = self.token_embedding(token_ids)
        return x

    def apply_cnn_branches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution branches in parallel.
        Args:
            x: Token embeddings of shape [batch_size, embedding_dim, sequence_length]
        Returns:
            Concatenated branch features of shape [batch_size, total_features]
        """
        branch_features = []
        for branch in self.encoder:
            # Each branch: [B, E, L] -> [B, filters_per_kernel]
            features = branch(x)
            branch_features.append(features)

        # Concatenate all branch features: [B, total_features]
        x = torch.cat(branch_features, dim=1)
        return x

    def apply_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply squeeze-and-excitation attention if enabled.
        Args:
            x: Features of shape [batch_size, total_features]
        Returns:
            Attention-weighted features of shape [batch_size, total_features]
        """
        if self.attention is not None:
            x = self.attention(x.unsqueeze(-1)).squeeze(-1)
        return x

    def forward(self, token_ids: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the CNN regressor.

        Args:
            token_ids: SMILES token indices of shape [batch_size, sequence_length]
            token_mask: Optional binary tensor of shape [batch_size, sequence_length]
                       indicating which tokens to mask (1=mask, 0=keep)
        Returns:
            Regression predictions of shape [batch_size]
        """
        # Encode features with optional token-level masking
        x = self.encode_features(token_ids=token_ids, token_mask=token_mask)

        # Pass through CNN branches
        x = self.apply_cnn_branches(x)

        # Apply attention if enabled
        x = self.apply_attention(x)

        # Prediction head: [B, total_features] -> [B] -> 1
        x = self.head(x)

        return x.squeeze(-1)

    def reset_parameters(self) -> None:
        """Reset all model parameters."""
        self.token_embedding.reset_parameters()

        for branch in self.encoder:
            branch.reset_parameters()

        if self.attention is not None:
            self.attention.reset_parameters()

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
