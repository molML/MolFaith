import torch
import torch_geometric.nn as geom_nn
from abc import abstractmethod
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.resolver import normalization_resolver
from torch_geometric.utils.smiles import x_map
from typing import Optional, Dict

from mol_faith.model.configs.dnn_configs import DNNModelConfig
from mol_faith.model.shared.prediction_head import PredictionHead


class AtomFeatureEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        # +1 for MASK token
        # We have the normal categories and add an additional category for masking
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)
        self.vocab_size = vocab_size

    def forward(self, feature_indices: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            feature_indices: Tensor of shape [num_atoms], categorical indices (e.g., [6, 7, 8, 6])
            mask: Binary tensor [num_atoms], 1 where feature is masked, 0 otherwise

        Returns:
            Embedded features of shape [num_atoms, embedding_dim]
        """
        # Only apply masking if we have indeed an atom feature to mask! mask=[0,0,0,0] -> Skip and use normal input!
        if mask is not None and mask.any():
            masked_indices = feature_indices.clone()
            # MASK token at the end
            # E.g. vocab_size = 22 (we just add +1 for the embedding input size directly and not
            # override the original vocab_size)
            # indices (original): [0, ..., 21] and with masking: [0, ..., 22] -> Masking index = 22 (vocab size)
            masked_indices[mask == 1] = self.vocab_size
            return self.embedding(masked_indices)
        return self.embedding(feature_indices)


class AtomFeatureEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 32):
        super().__init__()

        # Use PyG's official feature mapping
        self.x_map = x_map

        # Create embedding layers for each feature
        self.embeddings = nn.ModuleDict()

        for feature_name, value_list in self.x_map.items():
            vocab_size = len(value_list)
            self.embeddings[feature_name] = AtomFeatureEmbedding(vocab_size, embedding_dim)

        # Total embedding dimension after concatenation
        self.total_embedding_dim = len(self.x_map) * embedding_dim
        self.num_features = len(self.x_map)

    def forward(self, atom_features: Dict[str, torch.Tensor],
                feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            atom_features: Dict with feature tensors, each shape [num_atoms]
            feature_mask: Binary tensor [num_atoms, num_features], 1 where features are masked, 0 otherwise
                         Shape: [4, 9] for 4 atoms with 9 features each
                         Example: [[0, 1, 0, ...],   # Atom 0: mask feature 1
                                  [0, 0, 0, ...],   # Atom 1: no masking
                                  [1, 1, 0, ...],   # Atom 2: mask features 0,1
                                  [0, 0, 0, ...]]   # Atom 3: no masking

        Returns:
            Embedded features of shape [num_atoms, total_embedding_dim]
        """
        embedded_list = []

        for feature_idx, feature_name in enumerate(self.x_map.keys()):
            embedding_layer = self.embeddings[feature_name]
            feature_indices = atom_features[feature_name]  # Shape: [num_atoms]

            # Extract mask for this specific feature across all atoms
            if feature_mask is not None:
                feature_specific_mask = feature_mask[:, feature_idx]  # Shape: [num_atoms]
            else:
                feature_specific_mask = None

            # Apply embedding with feature-specific masking [num_atoms, embedding_dim]
            embedded = embedding_layer(feature_indices, mask=feature_specific_mask)
            embedded_list.append(embedded)

        # Concatenate all feature embeddings [num_atoms, num_features*embedding_dim]
        return torch.cat(embedded_list, dim=1)

    def reset_parameters(self) -> None:
        """Reset all embedding parameters."""
        for embedding_layer in self.embeddings.values():
            embedding_layer.embedding.reset_parameters()


class GNNBlock(nn.Module):
    def __init__(self, gnn_layer: nn.Module, norm: Optional[nn.Module] = None,
                 dropout: Optional[nn.Module] = None) -> None:
        super().__init__()

        self.gnn_layer = gnn_layer
        self.norm = norm
        self.dropout = dropout
        self.activation = nn.ReLU()

        self._supports_batch_norm = self._check_batch_norm_support()

    def _check_batch_norm_support(self) -> bool:
        """Check if normalization layer supports batch parameter."""
        if not self.norm or not hasattr(self.norm, 'forward'):
            return False

        import inspect
        norm_params = inspect.signature(self.norm.forward).parameters
        return 'batch' in norm_params

    def reset_parameters(self) -> None:
        """Reset all layer parameters."""
        self.gnn_layer.reset_parameters()
        if self.norm:
            self.norm.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: GNN -> Norm -> Activation -> Dropout
        """
        # Graph convolution
        x = self.gnn_layer(x, edge_index)

        # Normalization
        if self.norm:
            x = self._apply_normalization(x, batch_index)

        # Activation
        x = self.activation(x)

        # Dropout
        if self.dropout:
            x = self.dropout(x)

        return x

    def _apply_normalization(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Apply normalization with or without batch parameter."""
        if self._supports_batch_norm:
            return self.norm(x, batch_index)
        return self.norm(x)


class BasicGNNRegressor(nn.Module):
    def __init__(self, config: DNNModelConfig):
        super().__init__()
        self.config = config

        # Initialize atom feature encoder
        self.atom_encoder = AtomFeatureEncoder(embedding_dim=config.embedding_dim)

        # Update config to use embedding dimension as input
        self._original_c_in = self.config.c_in
        self.config.c_in = self.atom_encoder.total_embedding_dim

        self._validate_config()

        self.encoder = self._build_encoder()
        self.head = self._build_prediction_head()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.c_in <= 0:
            raise ValueError(f"Input channels must be positive, got {self.config.c_in}")
        if self.config.c_hidden <= 0:
            raise ValueError(f"Hidden channels must be positive, got {self.config.c_hidden}")
        if self.config.num_layers <= 0:
            raise ValueError(f"Number of layers must be positive, got {self.config.num_layers}")

    def get_hidden_dimension(self) -> int:
        """
        Calculate the actual output dimension from GAT layers.

        Returns:
            int: Actual output dimensions considering concatenation behavior

        Examples:
            >>> # concat=True, c_hidden=64, heads=4
            >>> self.get_hidden_dimension()
            256

            >>> # concat=False, c_hidden=64, heads=4
            >>> self.get_hidden_dimension()
            64
        """
        if hasattr(self.config, 'concat') and self.config.concat:
            return self.config.c_hidden * self.config.heads
        else:
            return self.config.c_hidden

    def _build_encoder(self) -> nn.ModuleList:
        """Build the GNN encoder layers."""
        layers = nn.ModuleList()

        in_channels = self.config.c_in
        for layer_idx in range(self.config.num_layers):
            gnn_layer = self._create_gnn_layer(in_channels, self.config.c_hidden)

            # Adjust the input channels after one message passing (it depends on the used message passing layer)
            in_channels = self.get_hidden_dimension()

            norm = self._create_normalization_layer(in_channels) if self.config.norm_func else None
            dropout = self._create_dropout_layer() if self.config.dp_rate else None

            block = GNNBlock(
                gnn_layer=gnn_layer,
                norm=norm,
                dropout=dropout
            )
            layers.append(block)

        return layers

    def _build_prediction_head(self) -> PredictionHead:
        """Build the prediction head."""
        return PredictionHead(
            d_in=self.get_hidden_dimension(),
            n_out=1,
            hidden=self.config.c_hidden_head,
            dropout=self.config.dp_rate_head
        )

    @abstractmethod
    def _create_gnn_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        raise NotImplementedError("Subclasses must implement _create_gnn_layer")

    def _create_normalization_layer(self, in_channels: int) -> nn.Module:
        return normalization_resolver(self.config.norm_func, in_channels)

    def _create_dropout_layer(self) -> nn.Module:
        return nn.Dropout(self.config.dp_rate)

    def reset_parameters(self) -> None:
        # Reset atom encoder parameters
        self.atom_encoder.reset_parameters()

        # Reset GNN encoder parameters
        for layer in self.encoder:
            layer.reset_parameters()

        # Reset head parameters
        for module in self.head.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, data: Data, feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the GNN regressor with feature-level masking support.

        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features [total_nodes, num_features] OR dict of features
                - edge_index: Edge connectivity [2, num_edges]
                - batch: Batch indices [total_nodes]
            feature_mask: Binary tensor [num_atoms, num_features], 1 where features are masked, 0 otherwise
                         Shape: [4, 9] for 4 atoms with 9 features each
                         Example: [[0, 1, 0, ...],   # Atom 0: mask feature 1
                                  [0, 0, 0, ...],   # Atom 1: no masking
                                  [1, 1, 0, ...],   # Atom 2: mask features 0,1
                                  [0, 0, 0, ...]]   # Atom 3: no masking

        Returns:
            Regression predictions [batch_size] - one prediction per graph
        """
        edge_index = data.edge_index
        batch_index = getattr(data, 'batch', torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device))

        # Encode features with feature-level masking
        x = self.encode_features(data.x, feature_mask)

        # Pass through GNN layers
        x = self.apply_gnn_layers(x, edge_index, batch_index)

        # Global pooling
        x = self.pool(x, batch_index)

        # Prediction head
        x = self.head(x)

        return x.squeeze(-1)

    def encode_features(self, raw_features, feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode raw features using embeddings with feature-level masking.

        Args:
            raw_features: Either tensor [total_nodes, num_features] or dict
            feature_mask: Optional binary tensor [total_nodes, num_features]

        Returns:
            Embedded features [total_nodes, total_embedding_dim]
        """
        # Convert features to dictionary format if needed
        if isinstance(raw_features, dict):
            atom_features = raw_features
        else:
            atom_features = prepare_atom_features_dict(raw_features)

        # Apply feature embeddings with masking
        return self.atom_encoder(atom_features, feature_mask)

    def apply_gnn_layers(self, x: torch.Tensor, edge_index: torch.Tensor,
                         batch_index: torch.Tensor) -> torch.Tensor:
        """Apply GNN encoding layers."""
        for layer in self.encoder:
            x = layer(x, edge_index, batch_index)
        return x

    @staticmethod
    def pool(x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Global pooling operation."""
        return geom_nn.global_mean_pool(x, batch_index)

    def save_params(self, path: str) -> None:
        """Save model parameters to disk."""
        torch.save(self.state_dict(), path)

    def load_params(self, path: str) -> None:
        """Load model parameters from disk with backward compatibility."""
        state_dict = torch.load(path, weights_only=True)

        # Remove deprecated keys for backward compatibility
        deprecated_keys = {"norm.weight", "norm.bias", "norm.mean_scale"}
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k not in deprecated_keys
        }

        # Load with strict=False to handle missing embedding keys
        self.load_state_dict(filtered_state_dict, strict=False)


def prepare_atom_features_dict(raw_features: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Convert raw feature tensor to dictionary format expected by the model.

    Args:
        raw_features: Tensor of shape [total_nodes, num_features]
    Returns:
        Dict mapping feature names to tensors, each with shape [total_nodes]
    """
    atom_features = {}
    for i, feature_name in enumerate(x_map.keys()):
        atom_features[feature_name] = raw_features[:, i].long()

    return atom_features
