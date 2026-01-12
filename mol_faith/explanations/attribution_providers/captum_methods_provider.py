import logging
import numpy as np
import torch
from captum._utils.models.linear_model import SkLearnLinearModel
from captum.attr import IntegratedGradients, LayerGradCam, KernelShap, Lime, Attribution, \
    ShapleyValueSampling
from captum.attr._core.lime import get_exp_kernel_similarity_function
from torch import Size
from torch_geometric.data import Data
from torch_geometric.explain.algorithm.captum import to_captum_input
from torch_geometric.utils.smiles import x_map
from typing import Any, Optional, Union, Type

from mol_faith.data.data_managers.smiles_data_manager import SMILESTokenizer
from mol_faith.explanations.attribution_providers.guided_gradcam import GuidedGradCam
from mol_faith.explanations.attribution_providers.utils import BaseAttributionProvider
from mol_faith.explanations.types import ExplanationMethod, ExplanationMethodArgs

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CaptumAttributionProvider(BaseAttributionProvider):
    """
    Generic Captum attribution provider that works with any pre-wrapped model.
    """

    def __init__(self, model: Any, method: ExplanationMethod, method_args: Optional[ExplanationMethodArgs] = None):
        self.model = model
        self.method = method
        self.method_args = method_args or {}
        self.explainer = self._initialize_explainer()

    @property
    def _is_graph_model(self):
        # Bit ugly as we always use wrapper but keep it for now. ToDo: Make it more beautiful
        if self.model.model.__class__.__name__ == 'GCN':
            return True
        else:
            return False

    def _initialize_grad_cam_based_explainer(self, explainer_class: Union[Type[GuidedGradCam], Type[LayerGradCam]],
                                             method_name: str) -> Optional[Union[GuidedGradCam, LayerGradCam]]:
        """Common logic for GradCAM-style explainers (GradCAM, GuidedGradCAM, etc.)."""
        target_layer = self.method_args.get('explainer_args', {}).get('target_layer')
        target_layers = self.method_args.get('explainer_args', {}).get('target_layers')

        if target_layers is not None:
            # Multiple layers - create multiple explainers
            self.multi_layer_explainers = []
            for layer in target_layers:
                explainer = explainer_class(self.model, layer, self._is_graph_model)
                self.multi_layer_explainers.append(explainer)
            return None  # No single explainer
        elif target_layer is not None:
            # Single layer
            return explainer_class(self.model, target_layer, self._is_graph_model)
        else:
            raise ValueError(f'{method_name} requires either target_layer or target_layers in method_args')

    def _initialize_explainer(self) -> Optional[Attribution]:
        """Initialize the appropriate explainer."""
        if self.method == ExplanationMethod.GRAD_CAM:
            return self._initialize_grad_cam_based_explainer(LayerGradCam, 'GradCAM')
        elif self.method == ExplanationMethod.GUIDED_GRAD_CAM:
            return self._initialize_grad_cam_based_explainer(GuidedGradCam, 'GuidedGradCAM')
        elif self.method == ExplanationMethod.INTEGRATED_GRADIENTS:
            return IntegratedGradients(self.model)
        elif self.method == ExplanationMethod.SHAP_SAMPLING:
            return ShapleyValueSampling(self.model)
        elif self.method == ExplanationMethod.KERNEL_SHAP:
            return KernelShap(self.model)
        elif self.method == ExplanationMethod.LIME:
            exp_eucl_distance = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)
            return Lime(self.model, similarity_func=exp_eucl_distance,
                        interpretable_model=SkLearnLinearModel("linear_model.Ridge"))
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def explain(self, data: Any) -> Optional[np.ndarray]:
        """Generate explanations using Captum methods."""
        try:
            if hasattr(data, 'x'):
                return self._explain_gnn(data)
            else:
                return self._explain_sequence_model(data)

        except Exception as e:
            logger.error(f"Explanation failed for method {self.method}: {str(e)}")
            logger.error(f"Data type: {type(data).__name__}")
            if hasattr(data, 'shape'):
                logger.error(f"Data shape: {data.shape}")
            elif hasattr(data, 'x'):  # PyG Data object
                logger.error(f"Graph data - nodes: {data.x.shape[0]}, features: {data.x.shape[1]}")

            raise RuntimeError(f"Failed to generate {self.method.value} explanation: {str(e)}") from e

    @staticmethod
    def _create_baseline_for_gnn(data: Data, baseline_strategy: str = 'mask_tokens') -> torch.Tensor:
        """Create baseline input for GNN data."""
        num_atoms, num_features = data.x.shape
        device = data.x.device

        if baseline_strategy == 'mask_tokens':
            baseline_tensor = torch.zeros((num_atoms, num_features), dtype=torch.long, device=device)
            # Set each feature column to its respective MASK token index
            for feature_idx, (feature_name, value_list) in enumerate(x_map.items()):
                if feature_idx < num_features:
                    vocab_size = len(value_list)
                    mask_token_idx = vocab_size  # MASK token is at vocab_size
                    baseline_tensor[:, feature_idx] = mask_token_idx
            return baseline_tensor
        elif baseline_strategy == 'zeros':
            return torch.zeros((num_atoms, num_features), dtype=data.x.dtype, device=device)
        elif baseline_strategy == 'random':
            return torch.randint_like(data.x, high=10)
        else:
            raise ValueError(f"Unknown baseline strategy: {baseline_strategy}")

    def _create_baseline_for_sequence_model(self, data: torch.Tensor,
                                            baseline_strategy: str = 'mask_tokens') -> torch.Tensor:
        """Create baseline input for CNN/sequence data."""
        if baseline_strategy == 'mask_tokens':
            tokenizer: SMILESTokenizer = self.model.get_tokenizer()
            non_padding_mask = data != tokenizer.pad_token_id
            baseline_tensor = data.clone()
            baseline_tensor[non_padding_mask] = tokenizer.mask_token_id
            return baseline_tensor
        elif baseline_strategy == 'zeros':
            return torch.zeros_like(data)
        elif baseline_strategy == 'random':
            return torch.randn_like(data)
        else:
            raise ValueError(f"Unknown baseline strategy for CNN: {baseline_strategy}")

    def _create_feature_mask_for_gnn(self, embedded_data: torch.Tensor, original_shape: Size) -> torch.Tensor:
        """Create feature mask for GNN embedded data where embedding dimensions are grouped by feature."""
        num_nodes, num_features = original_shape
        embedding_dim = self.model.model.config.embedding_dim

        feature_mask = []
        group_id = 0

        for node_idx in range(num_nodes):
            for feature_idx in range(num_features):
                # Each feature's embedding dimensions get the same group ID
                feature_mask.extend([group_id] * embedding_dim)
                group_id += 1

        return torch.tensor(feature_mask).reshape(embedded_data.shape).to(embedded_data.device)

    def create_feature_mask_for_sequence_model(self, data: torch.Tensor) -> torch.Tensor:
        """Create feature mask for CNN/sequence data or Transformer data."""
        model_type = self.model.get_model_type()

        if model_type == 'CNN':
            # CNN format: [batch_size, embedding_dim, seq_len]
            batch_size, embedding_dim, seq_len = data.shape
            # Create base sequence: [0, 1, 2, ..., seq_len-1]
            token_indices = torch.arange(seq_len, device=data.device)
            # Expand to [embedding_dim, seq_len] by repeating across embedding dimension
            token_mask = token_indices.unsqueeze(0).expand(embedding_dim, seq_len)
            # Expand to [batch_size, embedding_dim, seq_len] by repeating across batch dimension
            feature_mask = token_mask.unsqueeze(0).expand(batch_size, embedding_dim, seq_len)
            return feature_mask
        elif model_type == 'Transformer':
            # Transformer format: [batch_size, seq_len, embedding_dim]
            batch_size, seq_len, embedding_dim = data.shape
            # Create base sequence: [0, 1, 2, ..., seq_len-1]
            token_indices = torch.arange(seq_len, device=data.device)
            # Expand to [seq_len, embedding_dim] by repeating across embedding dimension
            token_mask = token_indices.unsqueeze(1).expand(seq_len, embedding_dim)
            # Expand to [batch_size, seq_len, embedding_dim] by repeating across batch dimension
            feature_mask = token_mask.unsqueeze(0).expand(batch_size, seq_len, embedding_dim)
            return feature_mask
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _aggregate_with_feature_mask_gnn(self, attributions: torch.Tensor,
                                         original_shape: Size) -> np.ndarray:
        """
        Aggregate attributions when feature mask was used for GNN data.
        Since feature mask groups embedding dimensions by feature, we take the first value from each group.
        """
        num_nodes, num_features = original_shape
        embedding_dim = self.model.model.config.embedding_dim

        try:
            reshaped = attributions.view(num_nodes, num_features, embedding_dim)
            # Take the first value from each feature group (since they're all the same due to feature mask)
            feature_attributions = reshaped[:, :, 0]  # Shape: (num_nodes, num_features)
            return np.abs(feature_attributions.cpu().numpy())
        except RuntimeError as e:
            logger.error(
                f"Failed to reshape attributions {attributions.shape} to ({num_nodes}, {num_features}, {embedding_dim})")
            raise ValueError(f"Attribution shape incompatible with expected dimensions") from e

    @staticmethod
    def _expand_attributions_on_feature_level(attributions: torch.Tensor,
                                              original_shape: Size) -> np.ndarray:
        n_nodes, n_features = original_shape

        if attributions.dim() == 1:
            # Shape [n_nodes] -> reshape to [n_nodes, 1]
            attributions = attributions.unsqueeze(1)
        elif attributions.shape != (n_nodes, 1):
            raise ValueError(f"Unsupported attribution shape: {attributions.shape}")

        node_feature_attributions = attributions / n_features
        expanded_attributions = node_feature_attributions.repeat(1, n_features)
        expanded_attributions = torch.abs(expanded_attributions)

        return expanded_attributions.detach().cpu().numpy()

    def _aggregate_with_feature_mask_sequence_model(self, attributions: torch.Tensor) -> np.ndarray:
        """
        Aggregate attributions when feature mask was used for sequence data.
        Since feature mask groups embedding dimensions by token, we take the first value from each group.
        """
        model_type = self.model.get_model_type()

        if model_type == 'CNN':
            # CNN format: attributions shape [embedding_dim, seq_len]
            # Take the first value from each token group (since they're all the same due to feature mask)
            token_attributions = attributions[0, :]
            return np.abs(token_attributions.cpu().numpy())
        elif model_type == 'Transformer':
            # Transformer format: attributions shape [seq_len, embedding_dim]
            # Take the first value from each token group (across embedding dimension)
            token_attributions = attributions[:, 0]
            return np.abs(token_attributions.cpu().numpy())
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _aggregate_without_feature_mask_sequence_model(self, attributions: torch.Tensor) -> np.ndarray:
        """Aggregate attributions for sequence data without feature mask."""
        model_type = self.model.get_model_type()

        if model_type == 'CNN':
            # CNN format: attributions shape [embedding_dim, seq_len]
            if attributions.ndim == 2:
                # Sum across embedding dimension (axis=0) to get token-level attributions
                return np.sum(np.abs(attributions.detach().cpu().numpy()), axis=0)
            else:
                raise ValueError(f'Expected CNN attributions with shape [embedding_dim, seq_len] '
                                 f'but got {attributions.shape}.')
        elif model_type == 'Transformer':
            # Transformer format: attributions shape [seq_len, embedding_dim]
            if attributions.ndim == 2:
                # Sum across embedding dimension (axis=1) to get token-level attributions
                return np.sum(np.abs(attributions.detach().cpu().numpy()), axis=1)
            else:
                raise ValueError(f'Expected Transformer attributions with shape [seq_len, embedding_dim] '
                                 f'but got {attributions.shape}.')
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _explain_gnn(self, data: Data) -> np.ndarray:
        """
        Unified explanation method for GNN data with baseline support.
        Handles all Captum methods with proper baseline, feature mask, and aggregation.
        """
        method_args = self.method_args.get('attribute_args', {}).copy()
        # Step 1: Create baseline (GradCam and GuidedGradCAM can not use baseline argument)
        if self.method not in [ExplanationMethod.GRAD_CAM, ExplanationMethod.GUIDED_GRAD_CAM]:
            baseline_input = self._create_baseline_for_gnn(data, 'mask_tokens')
            baseline_embedding = self.model.encode_features(baseline_input)
            baseline_embedding = (baseline_embedding.unsqueeze(0),)  # Format for Captum
            method_args['baselines'] = baseline_embedding

        # Step 2: Encode input features
        encoded_features = self.model.encode_features(data.x)

        # Step 3: Determine if we need feature mask (only for LIME and KernelSHAP)
        needs_feature_mask = self.method in [ExplanationMethod.KERNEL_SHAP, ExplanationMethod.LIME,
                                             ExplanationMethod.SHAP_SAMPLING]
        if needs_feature_mask:
            feature_mask = self._create_feature_mask_for_gnn(encoded_features, data.x.shape)
            method_args['feature_mask'] = feature_mask

        # Step 4: Add method-specific arguments
        if self.method == ExplanationMethod.INTEGRATED_GRADIENTS:
            # We need this so that GNN and captum works together. It is due to the batch dimension handling from captum!
            method_args['internal_batch_size'] = 1

        # Step 5: Prepare inputs for Captum
        inputs, additional_args = to_captum_input(encoded_features, data.edge_index, 'node', data.batch)

        # Step 6: Get attributions
        attributions = self.explainer.attribute(
            inputs,
            additional_forward_args=additional_args,
            **method_args
        )

        # For GradCAM we do not have a batch dimension anymore but need to expand the attributions to feature level
        # We get shape [n_nodes, 1] but we need to have shape [n_nodes, n_features] in order to correctly evaluate
        # F-Fidelity where the attributions should be on the same level as the input
        if self.method == ExplanationMethod.GRAD_CAM:
            return self._expand_attributions_on_feature_level(attributions, data.x.shape)

        attributions = attributions[0].squeeze(0)  # Remove batch dimension

        # Step 8: Aggregate attributions to have same shape as the input [n_nodes, n_features]
        if needs_feature_mask:
            return self._aggregate_with_feature_mask_gnn(attributions, data.x.shape)
        else:
            return self._aggregate_without_feature_mask_gnn(attributions, data.x.shape)

    def _explain_cnn_gradcam_multi_layer(self, data: torch.Tensor) -> np.ndarray:
        """
        Handle multi-layer GradCAM for CNN.
        """
        if self.multi_layer_explainers is None:
            raise ValueError("Multi-layer explainers not initialized!")

        method_args = self.method_args.get('attribute_args', {}).copy()

        # Step 2: Encode input features
        encoded_features = self.model.encode_features(data)

        # Step 3: Get attributions from each layer
        all_attributions = []

        for explainer in self.multi_layer_explainers:
            # Get raw attributions for this layer
            attrs = explainer.attribute(encoded_features, **method_args)
            attrs = attrs[0].squeeze(0)  # Remove batch dimension

            all_attributions.append(attrs.detach().cpu().numpy())

        # Step 3: Stack and aggregate across layers
        stacked_attrs = np.stack(all_attributions, axis=0)

        if self.method == ExplanationMethod.GRAD_CAM:
            # For GradCAM we get attributions for each input token
            # stacked_attrs shape [num_conv_layers, seq_len] -> [seq_len]
            return np.sum(np.abs(stacked_attrs), axis=0)
        else:
            # For Guided GradCAM we need to aggregate to token level
            # stacked_attrs shape [num_conv_layers, embed_dim, seq_len] -> [embed_dim, seq_len]
            attributions = np.sum(np.abs(stacked_attrs), axis=0)
            # [embed_dim, seq_len] -> [seq_len]
            return self._aggregate_without_feature_mask_sequence_model(torch.Tensor(attributions))

    def _explain_sequence_model(self, data: torch.Tensor) -> np.ndarray:
        """
        Unified explanation method for CNN/sequence data with baseline support.
        """
        # If we have a gradcam based xAI method, we need to extract attributions differently as we have
        # multiple CNN layers that we have to get attributions from! We just sum the attributions from each conv layer.
        # Additionally, these methods do not support baseline argument
        # For transformer model we do not have gradCAM based methods so we do not need to handle it!
        if self.method in [ExplanationMethod.GRAD_CAM, ExplanationMethod.GUIDED_GRAD_CAM]:
            return self._explain_cnn_gradcam_multi_layer(data)

        method_args = self.method_args.get('attribute_args', {}).copy()

        # Step 1: Create baseline. For gradcam that does not support baseline we do not execute this part
        baseline_input = self._create_baseline_for_sequence_model(data, 'mask_tokens')
        baseline_embedding = self.model.encode_features(baseline_input)
        method_args['baselines'] = baseline_embedding

        # Step 2: Encode input features
        encoded_features = self.model.encode_features(data)

        # Step 3: Add feature mask if needed
        needs_feature_mask = self.method in [ExplanationMethod.KERNEL_SHAP, ExplanationMethod.LIME,
                                             ExplanationMethod.SHAP_SAMPLING]
        if needs_feature_mask:
            feature_mask = self.create_feature_mask_for_sequence_model(encoded_features)
            method_args['feature_mask'] = feature_mask

        # Step 5: Get attributions
        attributions = self.explainer.attribute(encoded_features, **method_args)
        attributions = attributions[0].squeeze(0)

        # Step 6: Aggregate results
        if needs_feature_mask:
            return self._aggregate_with_feature_mask_sequence_model(attributions)
        else:
            return self._aggregate_without_feature_mask_sequence_model(attributions)
