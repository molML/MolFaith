import numpy as np
import torch
from torch_geometric.explain import Explainer, GNNExplainer
from typing import Any, Dict, Optional

from mol_faith.explanations.attribution_providers.graphmask import GraphMaskExplainer
from mol_faith.explanations.attribution_providers.utils import BaseAttributionProvider
from mol_faith.explanations.types import ExplanationMethod, ExplanationMethodArgs


class PyTorchGeometricProvider(BaseAttributionProvider):
    def __init__(self, model: Any, method: str, method_args: Optional[ExplanationMethodArgs] = None):
        self.model = model
        self.method = method
        self.method_args = method_args or {}
        self.explainer = self._get_explainer()

    @property
    def get_default_args(self) -> Dict:
        """Get default arguments for each method."""
        return {
            ExplanationMethod.GNN_EXPLAINER: {
                "explainer_args": {
                    "epochs": 200,
                    "lr": 0.001,
                }
            },
            ExplanationMethod.GRAPH_MASK: {
                "explainer_args": {
                    "num_layers": self.model.model.config.num_layers,
                    "epochs": 200,
                    "lr": 0.001,
                    "log": False
                }
            }
        }

    def _get_explainer(self):
        # Merge default args with user-provided args
        default_args = self.get_default_args.get(self.method, {})

        explainer_args = {
            **default_args.get("explainer_args", {}),
            **self.method_args.get("explainer_args", {})
        }

        if self.method == ExplanationMethod.GNN_EXPLAINER:
            algorithm = GNNExplainer(**explainer_args)
        elif self.method == ExplanationMethod.GRAPH_MASK:
            algorithm = GraphMaskExplainer(**explainer_args)
        else:
            raise ValueError(f"Unsupported PyG method: {self.method}")

        return Explainer(
            self.model,
            algorithm=algorithm,
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type=None,
            model_config=dict(
                mode="regression",
                task_level="graph",
                return_type="raw"
            )
        )

    def explain(self, data: Any) -> np.ndarray:
        """Get explanations using PyTorch Geometric methods.
        
        Args:
            data: PyG Data object containing:
                - x: Node features
                - edge_index: Graph connectivity
                - y: Target value
                - batch: Batch indices (optional)
                
        Returns:
            torch.Tensor: Node mask explanations
        """
        if not hasattr(data, 'edge_index'):
            raise ValueError("Data must have edge_index attribute")

        if not hasattr(data, 'y'):
            raise ValueError("Data must have y attribute")

        if not hasattr(data, 'x'):
            raise ValueError("Data must have x attribute")

        batch_index = getattr(data, 'batch', torch.zeros(data.x.size(0), dtype=torch.long))

        # Need to do this or else we get some problem in the pyg code (.backwards is called twice)
        with torch.no_grad():
            encoded_features = self.model.encode_features(data.x)

        attributions = self.explainer(
            encoded_features,
            data.edge_index,
            batch_index=batch_index
        )

        attributions = attributions.node_mask.detach().cpu()
        return self._aggregate_without_feature_mask_gnn(attributions=attributions, original_shape=data.x.shape)
