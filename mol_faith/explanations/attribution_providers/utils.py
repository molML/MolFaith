import numpy as np
import torch
from typing import Protocol, Any


class BaseAttributionProvider(Protocol):
    def explain(self, data: Any) -> np.ndarray:
        """Get explanations for input data
        
        Args:
            data: Input data to explain. Could be:
                - PyG Data object with x, edge_index, y attributes
                - Tensor for fingerprint data
                
        Returns:
            torch.Tensor: Attribution scores
        """
        raise NotImplementedError


    @staticmethod
    def _aggregate_without_feature_mask_gnn(attributions: torch.Tensor,
                                            original_shape) -> np.ndarray:
        """
        Aggregate attributions when no feature mask was used.
        Sum across embedding dimensions for each feature.
        """
        num_nodes, num_features = original_shape

        if attributions.shape[1] == num_features:
            # Already at feature level
            return np.abs(attributions.detach().cpu().numpy())

        # Need to aggregate from embedding level
        embedding_dim_per_feature = attributions.shape[1] // num_features
        result = np.zeros((num_nodes, num_features))

        for i in range(num_features):
            start_idx = i * embedding_dim_per_feature
            end_idx = (i + 1) * embedding_dim_per_feature
            feature_attr = attributions[:, start_idx:end_idx]
            result[:, i] = torch.sum(torch.abs(feature_attr), dim=1).detach().cpu().numpy()

        return result