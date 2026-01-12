import torch
from abc import ABC, abstractmethod
from enum import Enum
from torch_geometric.data import Data
from typing import Tuple, Optional

from mol_faith.data.data_managers.smiles_data_manager import SMILESTokenizer


class EvaluationStrategy(Enum):
    """Strategies for selecting explanation features"""
    MASS_THRESHOLD = "mass_threshold"
    TOP_PERCENTAGE = "top_percentage"


class BaseFidelityMasker(ABC):
    def __init__(self, beta: float = 0.1, alpha_plus_orig: float = 0.5,
                 alpha_minus_orig: float = 0.5, n_samples: int = 50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.beta = beta
        self.alpha_plus_orig = alpha_plus_orig
        self.alpha_minus_orig = alpha_minus_orig
        self.n_samples = n_samples

    @staticmethod
    def _detect_edge_case(s_ratio: float) -> Optional[str]:
        """Detect edge cases based on s_ratio thresholds"""
        if s_ratio < 0.05:
            return 'too_sparse'
        elif s_ratio > 0.95:
            return 'too_dense'
        else:
            return None

    def _attribution_mass_selection(self, attribution_scores: torch.Tensor, mass_threshold: float) -> Tuple[
        torch.Tensor, int]:
        """Select features based on attribution mass (cumulative importance)"""
        if torch.any(attribution_scores < 0):
            negative_count = torch.sum(attribution_scores < 0).item()
            min_value = attribution_scores.min().item()
            raise ValueError(f"Attribution mass selection requires non-negative scores. "
                             f"Found {negative_count} negative values (min: {min_value:.6f}). "
                             f"Consider: 1) Using torch.abs(scores), 2) Filter out negative attributions")

        total_importance = attribution_scores.sum()

        if total_importance == 0:
            raise ValueError(f'Encountered example with no attribution mass: {attribution_scores}')

        relative_importance = attribution_scores / total_importance

        sorted_indices = torch.argsort(relative_importance, descending=True)

        cumulative_mass = 0.0
        selected_indices = []

        for idx in sorted_indices:
            selected_indices.append(idx.item())
            cumulative_mass += relative_importance[idx].item()
            if cumulative_mass >= mass_threshold:
                break

        explanation_mask = torch.zeros_like(attribution_scores, dtype=torch.bool, device=self.device)
        explanation_mask[selected_indices] = True

        return explanation_mask, len(selected_indices)

    def _top_k_percent_selection(self, attribution_scores: torch.Tensor, percentage: float, td: int) -> Tuple[
        torch.Tensor, int]:
        """
        Select features based on top percentage of features by attribution magnitude

        Args:
            attribution_scores: Attribution scores for all features (flattened)
            percentage: Percentage of features to select (0.0 to 1.0)
            td: Total number of features

        Returns:
            explanation_mask: Boolean mask indicating selected features
            s: Number of selected features
        """
        if not (0.0 <= percentage <= 1.0):
            raise ValueError(f"Percentage must be between 0.0 and 1.0, got: {percentage}")

        # Calculate number of features to select (at least 1)
        num_features_to_select = max(1, int(percentage * td))

        # Get indices of top-k features by attribution magnitude (absolute value for handling negative attributions)
        _, top_indices = torch.topk(torch.abs(attribution_scores), num_features_to_select, largest=True)

        # Create explanation mask
        explanation_mask = torch.zeros_like(attribution_scores, dtype=torch.bool, device=self.device)
        explanation_mask[top_indices] = True

        return explanation_mask, num_features_to_select

    def _calculate_fidelity_parameters(self, s: int, td: int, threshold_value: float,
                                       strategy: EvaluationStrategy) -> Tuple[
        Optional[float], Optional[float], Optional[dict]]:
        """Calculate F-Fidelity parameters and enhanced metadata"""
        # Skip extreme cases
        if s == 0 or s == td:
            return None, None, None

        # Calculate F-Fidelity parameters
        s_ratio = s / td
        edge_case_type = self._detect_edge_case(s_ratio)
        alpha_plus = min(self.alpha_plus_orig, self.beta * td / s)
        alpha_minus = min(self.alpha_minus_orig, self.beta * td / (td - s))

        # Enhanced metadata
        metadata = {
            # Core F-Fidelity parameters
            'alpha_plus': alpha_plus,
            'alpha_minus': alpha_minus,
            'alpha_plus_origin': self.alpha_plus_orig,
            'alpha_minus_origin': self.alpha_minus_orig,
            's_value': s,
            'td_value': td,
            's_ratio': s_ratio,

            # Edge case detection
            'edge_case_type': edge_case_type,
            'is_edge_case': edge_case_type is not None,

            # Constraint analysis
            'constraint_activation': {
                'alpha_plus_limited_by_beta': alpha_plus < self.alpha_plus_orig,
                'alpha_minus_limited_by_beta': alpha_minus < self.alpha_minus_orig,
            },

            # Evaluation feasibility
            'evaluation_feasibility': {
                'can_evaluate_fid_plus': s > 0,
                'can_evaluate_fid_minus': s < td,
            },

            # Attribution analysis
            'selection_strategy': strategy.value,
            'threshold_value_used': threshold_value,
        }

        return alpha_plus, alpha_minus, metadata

    @abstractmethod
    def create_finetuning_masks(self, batch_data: torch.Tensor) -> torch.Tensor:
        """Create random masks for fine-tuning phase"""
        raise NotImplementedError

    @abstractmethod
    def create_evaluation_masks(self, data, attribution_scores: torch.Tensor, threshold_value: float = 0.8,
                                strategy: EvaluationStrategy = EvaluationStrategy.MASS_THRESHOLD) -> Optional[dict]:
        """Create masks for XAI evaluation based on attribution scores"""
        raise NotImplementedError


class SequenceFFidelityMasker(BaseFidelityMasker):
    """
    F-Fidelity masking for sequence data (SMILES) with padding token support.

    Mask convention: 1 = mask, 0 = keep
    """

    def __init__(self, tokenizer: SMILESTokenizer, beta: float,
                 alpha_plus_orig: float = 0.5, alpha_minus_orig: float = 0.5, n_samples: int = 50):
        super().__init__(beta, alpha_plus_orig, alpha_minus_orig, n_samples)
        self.tokenizer = tokenizer

    def _validate_sequence_inputs(self, data: torch.Tensor, attribution_scores: torch.Tensor):
        """Validate sequence inputs for evaluation"""
        if data.shape[0] != 1:
            raise ValueError(f"Expected batch dimension to be 1, got shape: {data.shape[0]}")
        if len(attribution_scores.shape) != 1:
            raise ValueError(f"Expected 1D attribution scores, got shape: {attribution_scores.shape}")
        if data.shape[1] != len(attribution_scores):
            raise ValueError(f"Length mismatch: data={data.shape[1]}, scores={len(attribution_scores)}")
        if not torch.all((data >= 0) & (data < self.tokenizer.vocab_size)):
            raise ValueError("Invalid token indices found in sequence data")

    def _validate_batch_inputs(self, batch_data: torch.Tensor):
        """Validate batch inputs for fine-tuning"""
        if len(batch_data.shape) != 2:
            raise ValueError(f"Expected 2D batch [batch_size, seq_len], got shape: {batch_data.shape}")
        if not torch.all((batch_data >= 0) & (batch_data < self.tokenizer.vocab_size)):
            raise ValueError("Invalid token indices found in batch data")

    def create_finetuning_masks(self, batch_data: torch.Tensor) -> torch.Tensor:
        """Create random masks for fine-tuning phase using fixed removal count."""
        self._validate_batch_inputs(batch_data)

        batch_size, seq_len = batch_data.shape
        masks = torch.zeros_like(batch_data, dtype=torch.float, device=self.device)

        for i in range(batch_size):
            # Get real tokens for this sequence
            real_token_mask = (batch_data[i] != self.tokenizer.pad_token_id)
            real_token_indices = torch.where(real_token_mask)[0]

            # Calculate exact number of elements to mask (If input is too small e.g 0, mask at least 1)
            num_tokens_to_mask = max(int(self.beta * len(real_token_indices)), 1)

            # Randomly select exactly num_tokens_to_mask indices
            selected_indices = real_token_indices[
                torch.randperm(len(real_token_indices), device=self.device)[:num_tokens_to_mask]
            ]
            masks[i, selected_indices] = 1

        return masks

    def create_evaluation_masks(self, data: torch.Tensor, attribution_scores: torch.Tensor,
                                threshold_value: float = 0.8,
                                strategy: EvaluationStrategy = EvaluationStrategy.MASS_THRESHOLD) -> Optional[dict]:
        """Create masks for XAI evaluation based on importance scores (single example only)."""
        # Validate inputs
        self._validate_sequence_inputs(data, attribution_scores)

        # Remove batch dimension to get actual sequence length
        seq_len = len(torch.squeeze(data))

        # Clean attributions and get sequence info of the actual molecule
        cleaned_attributions, td, non_padding_token_mask = self._process_attributions_and_get_td(data,
                                                                                                 attribution_scores)

        # Get the attribution scores for the molecule ignoring the padding part of the sequence
        non_padding_token_attributions = cleaned_attributions[non_padding_token_mask]

        # Select explanation mask based on strategy
        if strategy == EvaluationStrategy.MASS_THRESHOLD:
            non_padding_explanation_mask, s = self._attribution_mass_selection(non_padding_token_attributions,
                                                                               threshold_value)
        elif strategy == EvaluationStrategy.TOP_PERCENTAGE:
            # We need td to calculate the top k% of attributions
            non_padding_explanation_mask, s = self._top_k_percent_selection(non_padding_token_attributions,
                                                                            threshold_value, td)
        else:
            raise ValueError(f"Unsupported evaluation strategy: {strategy}")

        # Calculate shared F-Fidelity parameters
        alpha_plus, alpha_minus, metadata = self._calculate_fidelity_parameters(s, td, threshold_value, strategy)
        if metadata is None:  # Extreme cases: Skip evaluation
            return None

        # Add sequence-specific metadata
        metadata.update({
            'modality': 'sequence',
            'has_padding': True,
            'total_sequence_length': seq_len,
            'actual_molecule_length': td,
            'padding_tokens': seq_len - td,
            'padding_ratio': (seq_len - td) / seq_len,
        })

        fid_plus_masks = self._create_sequence_fid_plus_masks(non_padding_explanation_mask, alpha_plus,
                                                              seq_len=seq_len)
        fid_minus_masks = self._create_sequence_fid_minus_masks(non_padding_explanation_mask, alpha_minus,
                                                                seq_len=seq_len)

        return {
            'fid_plus_masks': fid_plus_masks,
            'fid_minus_masks': fid_minus_masks,
            'metadata': metadata
        }

    def _process_attributions_and_get_td(self, sequence: torch.Tensor,
                                         attribution_scores: torch.Tensor) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Clean attributions for padding tokens and calculate actual sequence length"""
        sequence = torch.squeeze(sequence)
        non_padding_token_mask = (sequence != self.tokenizer.pad_token_id)
        cleaned_attributions = attribution_scores.clone()
        cleaned_attributions[~non_padding_token_mask] = 0.0
        td = non_padding_token_mask.sum().item()
        return cleaned_attributions, td, non_padding_token_mask

    def _create_sequence_fid_plus_masks(self, non_padding_explanation_mask: torch.Tensor, alpha_plus: float,
                                        seq_len: int) -> torch.Tensor:
        """Create Fid+ masks for sequences: mask some important features

        Args:
            non_padding_explanation_mask: Boolean mask indicating which tokens are part of the explanation (no padding)
            alpha_plus: Probability of masking each explanation token
            seq_len: Full sequence length including padding

        Returns:
            Masks with shape [n_samples, seq_len] where 1=mask, 0=keep
        """
        explanation_indices = torch.where(non_padding_explanation_mask)[0]

        # Create random selection for each sample
        random_values = torch.rand((self.n_samples, len(explanation_indices)), device=self.device)
        mask_selection = random_values <= alpha_plus

        # Create final masks - since padding is only at the end, explanation_indices
        # are already in the correct position for the full sequence
        final_masks = torch.zeros((self.n_samples, seq_len), device=self.device, dtype=torch.float)

        for i in range(self.n_samples):
            selected_indices = explanation_indices[mask_selection[i]]
            final_masks[i, selected_indices] = 1

        return final_masks

    def _create_sequence_fid_minus_masks(self, non_padding_explanation_mask: torch.Tensor, alpha_minus: float,
                                         seq_len: int) -> torch.Tensor:
        """Create Fid- masks for sequences: mask some unimportant features

        Args:
            non_padding_explanation_mask: Boolean mask indicating which tokens are part of the explanation (no padding)
            alpha_minus: Probability of masking each non-explanation token
            seq_len: Full sequence length including padding

        Returns:
            Masks with shape [n_samples, seq_len] where 1=mask, 0=keep
        """
        non_explanation_mask = ~non_padding_explanation_mask
        non_explanation_indices = torch.where(non_explanation_mask)[0]

        # Create random selection for each sample - mask non-explanation tokens
        random_values = torch.rand((self.n_samples, len(non_explanation_indices)), device=self.device)
        mask_selection = random_values <= alpha_minus

        # Create final masks - only mask the randomly selected non-explanation tokens
        final_masks = torch.zeros((self.n_samples, seq_len), device=self.device, dtype=torch.float)

        for i in range(self.n_samples):
            selected_indices = non_explanation_indices[mask_selection[i]]
            final_masks[i, selected_indices] = 1

        return final_masks


class GraphFFidelityMasker(BaseFidelityMasker):
    """
    F-Fidelity masking for graph data with importance conservation.

    Mask convention: 1 = mask, 0 = keep
    """

    def __init__(self, beta,
                 alpha_plus_orig: float = 0.5,
                 alpha_minus_orig: float = 0.5,
                 n_samples: int = 50):
        super().__init__(beta, alpha_plus_orig, alpha_minus_orig, n_samples)

    @staticmethod
    def _validate_graph_inputs(data: Data, attribution_scores: torch.Tensor):
        """Validate graph inputs for evaluation"""
        if not hasattr(data, 'x') or data.x is None:
            raise ValueError("Graph data missing node features (data.x)")

        n_nodes, n_features = data.x.shape
        valid_shapes = [(n_nodes,), (n_nodes, n_features)]

        if attribution_scores.shape not in valid_shapes:
            raise ValueError(f"Attribution scores shape {attribution_scores.shape} not in {valid_shapes}")

        if not torch.isfinite(attribution_scores).all():
            raise ValueError("Attribution scores contain non-finite values (NaN or Inf)")

    @staticmethod
    def _validate_batch_graph_inputs(batch_data: Data):
        """Validate batch graph inputs for fine-tuning"""
        if not hasattr(batch_data, 'x') or batch_data.x is None:
            raise ValueError("Batch graph data missing node features (batch_data.x)")
        if len(batch_data.x.shape) != 2:
            raise ValueError(f"Expected 2D node features [n_nodes, n_features], got: {batch_data.x.shape}")
        if not hasattr(batch_data, 'batch') or batch_data.batch is None:
            raise ValueError(f"Expect to have a batch input in the finetuning stage! Batch attribute is not set in "
                             f"data; {batch_data}!")

    def create_finetuning_masks(self, batch_data: Data) -> torch.Tensor:
        """Create random masks for fine-tuning phase using fixed removal count per graph."""
        self._validate_batch_graph_inputs(batch_data)

        node_features = batch_data.x
        total_nodes, n_features = node_features.shape
        mask = torch.zeros_like(node_features, dtype=torch.float, device=self.device)

        # Process each graph independently
        batch_indices = batch_data.batch
        unique_graphs = torch.unique(batch_indices)

        for graph_idx in unique_graphs:
            # Get nodes belonging to this specific graph
            graph_node_mask = (batch_indices == graph_idx)
            graph_node_indices = torch.where(graph_node_mask)[0]
            graph_num_nodes = len(graph_node_indices)
            graph_total_elements = graph_num_nodes * n_features

            # Calculate number of elements to mask for this specific graph
            num_elements_to_mask = max(int(self.beta * graph_total_elements), 1)

            # Random selection within this graph's elements
            selected_flat_indices = torch.randperm(
                graph_total_elements, device=self.device
            )[:num_elements_to_mask]

            # Convert flat indices to 2D coordinates
            node_offsets = selected_flat_indices // n_features
            feature_indices = selected_flat_indices % n_features

            # Map to global node indices
            global_node_indices = graph_node_indices[node_offsets]

            # Set mask values
            mask[global_node_indices, feature_indices] = 1

        return mask

    def create_evaluation_masks(self, data: Data, attribution_scores: torch.Tensor,
                                threshold_value: float = 0.8,
                                strategy: EvaluationStrategy = EvaluationStrategy.MASS_THRESHOLD) -> Optional[dict]:

        """Create masks for XAI evaluation based on importance scores (single example only)."""
        # Validate inputs
        self._validate_graph_inputs(data, attribution_scores)

        node_features = data.x
        n_nodes, n_features = node_features.shape
        td = n_nodes * n_features

        # Handle different importance score granularity
        if len(attribution_scores.shape) == 1 and attribution_scores.shape[0] == n_nodes:
            # Distribute node importance across features (importance conservation)
            attribution_scores = attribution_scores.unsqueeze(-1) / n_features
            attribution_scores = attribution_scores.repeat(1, n_features)
            attribution_scores = attribution_scores.view(-1)
            granularity_type = 'node_level'
        elif len(attribution_scores.shape) == 2 and attribution_scores.shape == (n_nodes, n_features):
            attribution_scores = attribution_scores.view(-1)
            granularity_type = 'node_feature_level'
        else:
            raise ValueError(
                f"Attribution scores should be [n_nodes] or [n_nodes, n_features], got: {attribution_scores.shape}")

        # Select explanation mask based on strategy
        if strategy == EvaluationStrategy.MASS_THRESHOLD:
            explanation_mask, s = self._attribution_mass_selection(attribution_scores,
                                                                   threshold_value)
        elif strategy == EvaluationStrategy.TOP_PERCENTAGE:
            # We need td to calculate the top k% of attributions
            explanation_mask, s = self._top_k_percent_selection(attribution_scores,
                                                                threshold_value, td)
        else:
            raise ValueError(f"Unsupported evaluation strategy: {strategy}")

        # Calculate F-Fidelity parameters
        alpha_plus, alpha_minus, metadata = self._calculate_fidelity_parameters(s, td, threshold_value, strategy)
        if metadata is None:  # Extreme cases
            return None

        # Add graph-specific metadata
        metadata.update({
            'modality': 'graph',
            'has_padding': False,
            'total_sequence_length': None,
            'n_nodes': n_nodes,
            'n_features': n_features,
            'total_node_features': td,
            'granularity_type': granularity_type,
            'importance_conservation_applied': granularity_type == 'node_level',
        })

        # Create stochastic evaluation masks using graph-specific logic
        fid_plus_masks = self._create_graph_fid_plus_masks(explanation_mask, alpha_plus, (n_nodes, n_features))
        fid_minus_masks = self._create_graph_fid_minus_masks(explanation_mask, alpha_minus, (n_nodes, n_features))

        return {
            'fid_plus_masks': fid_plus_masks,
            'fid_minus_masks': fid_minus_masks,
            'metadata': metadata
        }

    def _create_graph_fid_plus_masks(self, explanation_mask: torch.Tensor, alpha_plus: float,
                                     mask_shape: Tuple[int, int]) -> torch.Tensor:
        """Create Fid+ masks for graphs: mask some important features"""
        explanation_indices = torch.where(explanation_mask)[0]

        # Create random values for all samples and explanation indices at once
        # Shape [n_samples, s]
        random_values = torch.rand((self.n_samples, len(explanation_indices)), device=self.device)
        mask_selection = random_values <= alpha_plus

        # Create final masks
        final_masks = torch.zeros((self.n_samples,) + mask_shape, device=self.device, dtype=torch.float)

        for i in range(self.n_samples):
            selected_indices = explanation_indices[mask_selection[i]]
            flat_mask = torch.zeros(mask_shape[0] * mask_shape[1], device=self.device, dtype=torch.float)
            flat_mask[selected_indices] = 1
            final_masks[i] = flat_mask.view(mask_shape)

        return final_masks

    def _create_graph_fid_minus_masks(self, explanation_mask: torch.Tensor, alpha_minus: float,
                                      mask_shape: Tuple[int, int]) -> torch.Tensor:
        """Create Fid- masks for graphs: mask some unimportant features"""
        non_explanation_mask = ~explanation_mask
        non_explanation_indices = torch.where(non_explanation_mask)[0]

        # Create random values for all samples and non-explanation indices at once
        random_values = torch.rand((self.n_samples, len(non_explanation_indices)), device=self.device)
        mask_selection = random_values <= alpha_minus

        # Create final masks - only mask the randomly selected non-explanation features
        final_masks = torch.zeros((self.n_samples,) + mask_shape, device=self.device, dtype=torch.float)

        for i in range(self.n_samples):
            selected_indices = non_explanation_indices[mask_selection[i]]
            flat_mask = torch.zeros(mask_shape[0] * mask_shape[1], device=self.device, dtype=torch.float)
            flat_mask[selected_indices] = 1
            final_masks[i] = flat_mask.view(mask_shape)

        return final_masks
