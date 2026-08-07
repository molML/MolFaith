import logging
import numpy as np
import os
import pandas as pd
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import Mol
from torch_geometric.data import Data
from typing import List, Optional, Dict, Any, Union, Tuple

from mol_faith.data.data_managers.graph_data_manager import PytorchGeometricFeatures, SupportedFeatureProviders
from mol_faith.data.data_managers.smiles_data_manager import SMILESTokenizer
from mol_faith.data.types import DataCategoryType
from mol_faith.explanations.explainer.base_explainer import Explainer
from mol_faith.explanations.explainer.gnn_explainer import GNNExplainer
from mol_faith.explanations.explainer.sequence_explainer import SequenceExplainer
from mol_faith.explanations.explainer.utils import Attributions
from mol_faith.explanations.types import ExplanationMethod, ExplanationMethodArgs
from mol_faith.model.CNN.cnn_model import BasicCNNRegressor
from mol_faith.model.GNN.base_gnn import BasicGNNRegressor
from mol_faith.model.GNN.gat_model import GAT
from mol_faith.model.GNN.gcn_model import GCN
from mol_faith.model.GNN.gin_model import GIN
from mol_faith.model.configs.dnn_configs import get_config, DNNConfig, DNNModelNames
from mol_faith.model.transformer.transformer import BasicTransformerRegressor
from mol_faith.target_extraction.benzene_substructure_extraction import BenzeneSubstructureExtractor
from mol_faith.target_extraction.hbond_donor_extraction import HBondDonorExtractor
from mol_faith.target_extraction.largest_conjugated_system_extraction import LargestConjugatedSystemExtractor
from mol_faith.target_extraction.shared import FilteredAttributions, map_tokens_to_atoms, \
    aggregate_token_attributions_to_atoms
from mol_faith.training.dnn_training.checkpoint_handler import CheckpointHandler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

type InputData = Union[torch.Tensor, Data]
type SupportedModels = Union[BasicGNNRegressor, BasicCNNRegressor, BasicTransformerRegressor]

AVAILABLE_EXTRACTORS = {
    'benzene': BenzeneSubstructureExtractor,
    'largest_conjugated_system': LargestConjugatedSystemExtractor,
    'hbond_donor': HBondDonorExtractor
}


@dataclass
class GTEvalResult:
    smiles: str
    target: float
    explanation_method: str
    data_split_idx: Optional[int]
    has_target_substructure: bool
    metadata: Dict[str, Any]


class BaseFractionEvaluator(ABC):
    def __init__(self,
                 model_path: str,
                 explanation_method: ExplanationMethod,
                 substructure_extractor_name: str,
                 output_folder: str,
                 explanation_methods_args: Optional[ExplanationMethodArgs] = None,
                 random_seed: int = 42,
                 skip_existing: bool = True):
        self.model_path = model_path
        self.explanation_method = explanation_method
        self.substructure_extractor_name = substructure_extractor_name
        self.output_folder = output_folder
        self.explanation_methods_args = explanation_methods_args
        self.random_seed = random_seed
        self.skip_existing = skip_existing
        self._set_seeds()
        self.is_multi_split = self._detect_multi_split_structure()

        # Initialize substructure extractor (config will be set later)
        if substructure_extractor_name not in AVAILABLE_EXTRACTORS:
            raise ValueError(f"Unsupported substructure extractor: {substructure_extractor_name}. "
                             f"Available options: {list(AVAILABLE_EXTRACTORS.keys())}")
        self.substructure_extractor_class = AVAILABLE_EXTRACTORS[substructure_extractor_name]
        self.substructure_extractor = None

    def _set_seeds(self):
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        np.random.seed(self.random_seed)

    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def _load_model_and_config(self, model_path: str) -> Tuple[SupportedModels, DNNConfig]:
        """Load fine-tuned model and config specific to the architecture"""
        pass

    @abstractmethod
    def _get_explainer(self, model_path: str) -> Explainer:
        """Get appropriate explainer for this model type"""
        pass

    @abstractmethod
    def _create_data(self, smiles: str, target: float, config) -> InputData:
        """Create data appropriate for this model type"""
        pass

    @abstractmethod
    def _get_random_attributions(self, data: InputData) -> np.ndarray:
        """Generate random attributions matching input shape"""
        pass

    def _get_substructure_extractor(self, config: DNNConfig):
        """Get or create substructure extractor with optional config for sequence models"""
        if self.substructure_extractor is None:
            self.substructure_extractor = self.substructure_extractor_class(config)
        return self.substructure_extractor

    def _detect_multi_split_structure(self) -> bool:
        """Detect if model_path contains multiple data splits"""
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")

        data_split_folders = [f for f in os.listdir(self.model_path)
                              if f.startswith('data_split_') and os.path.isdir(os.path.join(self.model_path, f))]

        if data_split_folders:
            logger.info(f"Detected multi-split structure with {len(data_split_folders)} splits")
            return True
        elif os.path.exists(os.path.join(self.model_path, 'config.yaml')):
            logger.info("Detected single model structure")
            return False
        else:
            raise ValueError(f"Invalid model path structure. Expected either data_split_x folders or config.yaml")

    def _get_data_split_paths(self) -> List[tuple]:
        """Get list of (data_split_idx, model_path) tuples"""
        if not self.is_multi_split:
            return [(None, self.model_path)]

        data_split_paths = []
        for folder in os.listdir(self.model_path):
            if folder.startswith('data_split_') and os.path.isdir(os.path.join(self.model_path, folder)):
                split_idx = int(folder.split('_')[-1])
                split_path = os.path.join(self.model_path, folder)
                data_split_paths.append((split_idx, split_path))

        return sorted(data_split_paths)

    def _check_existing_results(self, data_split_idx: Optional[int]) -> bool:
        """Check if results already exist for given data split"""
        output_dir = os.path.join(
            self.output_folder,
            self.explanation_method.value,
            self.substructure_extractor_name
        )

        if data_split_idx is not None:
            output_dir = os.path.join(output_dir, f'data_split_{data_split_idx}')

        output_file = os.path.join(output_dir, 'target_fraction_results.csv')

        exists = os.path.exists(output_file)
        if exists:
            logger.info(f"Found existing results for data_split={data_split_idx} at {output_file}")

        return exists

    @staticmethod
    def _get_attributions(smiles: str, target: float, explainer: Explainer) -> np.ndarray:
        """Get attributions from explainer"""
        attributions = explainer.get_single_attributions_scores(smiles=smiles, target_value=target)
        return attributions.atom_attributions

    @staticmethod
    def _load_test_dataset(config: DNNConfig) -> pd.DataFrame:
        """Load test dataset"""
        test_data_path = os.path.join(config.training.dataset_folder_path, f'{DataCategoryType.TEST.value}.csv')
        if not os.path.exists(test_data_path):
            raise ValueError(f"Test dataset not found: {test_data_path}")
        return pd.read_csv(test_data_path)

    @staticmethod
    def _filter_padding_attributions(smiles: str, attributions: np.ndarray, config: DNNConfig) -> np.ndarray:
        """
        Filter out padding token attributions from sequence attributions.
        Args:
            smiles: SMILES string of the molecule
            attributions: Attribution array for sequence tokens [seq_len]
            config: Configuration object with max_seq_len
        Returns:
            filtered_attributions: Attribution array with padding tokens removed (shorter array)
        """
        tokenizer = SMILESTokenizer.load_vocab(config.model.vocab_path)
        input_tensor = tokenizer.encode(
            smiles=smiles,
            max_seq_len=config.model.max_seq_len,
            padding=True
        )

        # Create mask for non-padding tokens
        non_padding_mask = (input_tensor != tokenizer.pad_token_id)

        # Return only attributions for actual molecule tokens (removes padding entirely)
        filtered_attributions = attributions[non_padding_mask]
        return filtered_attributions

    @staticmethod
    def _create_ground_truth_mask(filtered_result: FilteredAttributions) -> np.ndarray:
        """Create boolean mask for ground truth atoms"""
        total_atoms = len(filtered_result.complete_attributions.atom_attributions)
        gt_mask = np.zeros(total_atoms, dtype=bool)

        for atom_idx in filtered_result.target_indices:
            gt_mask[atom_idx] = True

        return gt_mask

    @staticmethod
    def _calculate_absolute_target_fraction(filtered_result: FilteredAttributions) -> Dict[
        str, int | float | list[int]]:
        """
        Calculate the fraction of absolute attributions that belong to the target substructure.

        Args:
            filtered_result: Result from extractor.filter_attributions

        Returns:
            Dict with target fraction metrics
        """
        if not filtered_result.has_targets or filtered_result.target_attributions is None:
            return {}

        target_attrs = filtered_result.target_attributions.atom_attributions
        complete_attrs = filtered_result.complete_attributions.atom_attributions

        target_abs_sum = np.sum(np.abs(target_attrs))
        total_abs_sum = np.sum(np.abs(complete_attrs))

        target_fraction = (target_abs_sum / total_abs_sum) if total_abs_sum > 0 else 0.0

        return {
            'target_fraction': float(target_fraction),
            'target_atoms_count': filtered_result.n_target_atoms,
            'total_atoms_count': len(complete_attrs),
            'target_indices': list(filtered_result.target_indices)
        }

    def _calculate_precision_at_k_gt(self, filtered_result) -> Dict[str, Any]:
        """
        Calculate precision@K where K = number of ground truth atoms.
        Since K = n_relevant_items, precision = recall = f1.

        Args:
            filtered_result: Result from extractor.filter_attributions

        Returns:
            Dict with precision@K metrics where K = number of GT atoms
        """
        if not filtered_result.has_targets or filtered_result.target_attributions is None:
            return {}

        complete_attrs = filtered_result.complete_attributions.atom_attributions
        gt_mask = self._create_ground_truth_mask(filtered_result)
        n_atoms_gt = np.sum(gt_mask)

        # Get top-K attributed atoms where K = number of ground truth atoms
        top_k_indices = np.argsort(np.abs(complete_attrs))[-n_atoms_gt:]
        predicted_important = np.zeros_like(complete_attrs, dtype=bool)
        predicted_important[top_k_indices] = True

        TP = np.sum(predicted_important & gt_mask)
        FP = np.sum(predicted_important & ~gt_mask)
        FN = np.sum(~predicted_important & gt_mask)

        # Since K = n_gt: precision = recall = f1 = TP/K
        precision_at_k_gt = TP / n_atoms_gt if n_atoms_gt > 0 else 0

        return {
            'precision_at_k_gt': float(precision_at_k_gt),
            'k_value_gt': int(n_atoms_gt)
        }

    def _calculate_f_score_at_mass_thres(self, filtered_result: FilteredAttributions) -> Dict[str, float | int]:
        """
        Calculate precision and recall at multiple mass thresholds.

        Args:
            filtered_result: Result from extractor.filter_attributions

        Returns:
            Dict with precision/recall metrics at multiple mass thresholds
        """
        if not filtered_result.has_targets or filtered_result.target_attributions is None:
            return {}

        complete_attrs = filtered_result.complete_attributions.atom_attributions
        gt_mask = self._create_ground_truth_mask(filtered_result)
        n_atoms_gt = np.sum(gt_mask)

        # Calculate mass needed to cover all target atoms
        mass_to_cover_all_targets = self._calculate_mass_to_cover_all_targets(complete_attrs, gt_mask)

        mass_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = {}
        for mass_threshold in mass_thresholds:
            # Find K needed to capture this mass threshold
            k = self._find_k_for_mass_threshold(complete_attrs, mass_threshold)

            # Calculate precision and recall at this K
            if k > 0:
                top_k_indices = np.argsort(np.abs(complete_attrs))[-k:]
                predicted_important = np.zeros_like(complete_attrs, dtype=bool)
                predicted_important[top_k_indices] = True

                TP = np.sum(predicted_important & gt_mask)
                FP = np.sum(predicted_important & ~gt_mask)
                FN = np.sum(~predicted_important & gt_mask)

                precision = TP / k
                recall = TP / n_atoms_gt if n_atoms_gt > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                results[f'precision_at_mass_{int(mass_threshold * 100)}'] = float(precision)
                results[f'recall_at_mass_{int(mass_threshold * 100)}'] = float(recall)
                results[f'f1_at_mass_{int(mass_threshold * 100)}'] = float(f1_score)
                results[f'k_at_mass_{int(mass_threshold * 100)}'] = int(k)

        results['mass_to_cover_all_targets'] = float(mass_to_cover_all_targets)

        return results

    @staticmethod
    def _find_k_for_mass_threshold(attributions: np.ndarray, mass_threshold: float) -> int:
        """
        Find the minimum K needed to capture mass_threshold of total attribution mass.

        Args:
            attributions: Attribution scores
            mass_threshold: Fraction of total mass to capture (0.0 to 1.0)

        Returns:
            K value (number of top features needed)
        """
        attr_abs = np.abs(attributions)
        total_mass = np.sum(attr_abs)

        if total_mass == 0:
            return len(attributions)

        # Normalize to relative importance (sums to 1)
        relative_importance = attr_abs / total_mass
        sorted_indices = np.argsort(relative_importance)[::-1]  # Descending order

        cumulative_mass = 0.0
        for k, idx in enumerate(sorted_indices, 1):
            cumulative_mass += relative_importance[idx]
            if cumulative_mass >= mass_threshold:
                return k

        return len(attributions)  # Edge case: need all features

    @staticmethod
    def _calculate_mass_to_cover_all_targets(attributions: np.ndarray, gt_benzene_mask: np.ndarray) -> float:
        """
        Calculate what fraction of total attribution mass is needed to cover all target atoms.

        Args:
            attributions: Attribution scores
            gt_benzene_mask: Boolean mask for target atoms

        Returns:
            Mass fraction needed to cover all targets (0.0 to 1.0)
        """
        attr_abs = np.abs(attributions)
        total_mass = np.sum(attr_abs)

        # Sort by importance (descending) and track which are targets
        sorted_indices = np.argsort(attr_abs)[::-1]
        targets_found = 0
        total_targets = np.sum(gt_benzene_mask)
        cumulative_mass = 0.0

        for idx in sorted_indices:
            cumulative_mass += attr_abs[idx]
            if gt_benzene_mask[idx]:
                targets_found += 1

            # If we've found all targets, return the mass fraction
            if targets_found == total_targets:
                return cumulative_mass / total_mass

        return 1.0  # Edge case: need all mass to find all targets

    @staticmethod
    def _calculate_attribution_concentration(filtered_attributions: FilteredAttributions) \
            -> (dict[str, None] | dict[str, float]):
        """
        Calculate attribution concentration metrics.

        Args:
            filtered_attributions: Preprocessed atom-level attributions

        Returns:
            Dict with concentration metrics
        """
        attributions = filtered_attributions.complete_attributions.atom_attributions

        attr_abs = np.abs(attributions)
        attr_normalized = attr_abs / np.sum(attr_abs)

        entropy = -np.sum(attr_normalized * np.log(attr_normalized + 1e-10))
        max_entropy = np.log(len(attributions))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return {
            'entropy': float(entropy),
            'normalized_entropy': float(normalized_entropy)
        }

    @staticmethod
    def _preprocess_graph_attributions(attributions: np.ndarray) -> np.ndarray:
        # [n_atoms, n_features] -> [n_atoms]
        return np.sum(np.abs(attributions), axis=-1)

    @staticmethod
    def _preprocess_sequence_attributions(mol: Mol, attributions: np.ndarray, vocab_path: str) -> np.ndarray:
        """
        Filters out the token attributions and then maps atom token attributions to atom attributions.
        The attributions are on token level so we need to get from token to atoms. We do this by only considering
        the attributions that we get for the token representing chemical symbols.
        The result is that instead of token attributions (shape [sequence_length]) we get attributions for only atoms
        (shape [n_atoms]), which is the same shape as for graph inputs
        """
        atom_to_token_mapping = map_tokens_to_atoms(mol=mol, vocab_path=vocab_path)
        atom_level_attributions = aggregate_token_attributions_to_atoms(token_attributions=attributions,
                                                                        atom_to_token_mapping=atom_to_token_mapping)
        return atom_level_attributions

    def evaluate_single_molecule(self,
                                 explainer: Optional[Explainer],
                                 smiles: str,
                                 target: float,
                                 data_split_idx: Optional[int],
                                 config: DNNConfig) -> GTEvalResult:
        """
        Evaluate all metrics for a single molecule.

        Args:
            explainer: Explanation method instance (None for random baseline)
            smiles: SMILES string of the molecule
            target: Target value for the molecule
            data_split_idx: Index in data split (for tracking)
            config: Configuration object

        Returns:
            FractionResult with all metrics in metadata
        """
        mol = Chem.MolFromSmiles(smiles)

        # Should be not possible because of a preprocessing of the dataset but just in case
        if mol is None:
            return GTEvalResult(
                smiles=smiles,
                target=target,
                explanation_method=self.explanation_method.value,
                data_split_idx=data_split_idx,
                has_target_substructure=False,
                metadata={'error': 'Invalid SMILES'}
            )

        if self.explanation_method == ExplanationMethod.RANDOM_EXPLANATION:
            data = self._create_data(smiles, target, config)
            attributions = self._get_random_attributions(data)
        else:
            attributions = self._get_attributions(smiles, target, explainer)

        # Check if we have non-zero attributions
        if not np.any(attributions):
            return GTEvalResult(
                smiles=smiles,
                target=target,
                explanation_method=self.explanation_method.value,
                data_split_idx=data_split_idx,
                has_target_substructure=False,
                metadata={'error': '0 Attributions vector. No attributions found'}
            )

        is_graph_input = True if attributions.ndim == 2 else False

        if is_graph_input:
            processed_attributions = self._preprocess_graph_attributions(attributions)
        else:
            processed_attributions = self._preprocess_sequence_attributions(mol=mol,
                                                                            attributions=attributions,
                                                                            vocab_path=config.model.vocab_path)

        extractor = self._get_substructure_extractor(config)
        attributions_obj = Attributions(atom_attributions=processed_attributions)
        filtered_result = extractor.filter_attributions(molecule=mol, attributions=attributions_obj)

        fraction_metrics = self._calculate_absolute_target_fraction(filtered_result)
        f_score_metrics = self._calculate_f_score_at_mass_thres(filtered_result)
        precision_at_k_gt_metrics = self._calculate_precision_at_k_gt(filtered_result)
        concentration_metrics = self._calculate_attribution_concentration(filtered_result)

        metadata = {
            **fraction_metrics,
            **f_score_metrics,
            **precision_at_k_gt_metrics,
            **concentration_metrics
        }

        return GTEvalResult(
            smiles=smiles,
            target=target,
            explanation_method=self.explanation_method.value,
            data_split_idx=data_split_idx,
            has_target_substructure=filtered_result.has_targets,
            metadata=metadata
        )

    def evaluate_data_split(self, model_path: str, data_split_idx: Optional[int]) -> None:
        """
        Evaluate entire test dataset for a single data split
        """
        logger.info(f"Evaluating data split {data_split_idx} at {model_path} "
                    f"for xAI method {self.explanation_method.value} "
                    f"with substructure extractor {self.substructure_extractor_name}")

        # Check if results already exist
        if self.skip_existing and self._check_existing_results(data_split_idx):
            logger.info(f"Results already exist for data_split={data_split_idx}. Skipping.")
            return

        model, config = self._load_model_and_config(model_path)

        explainer = self._get_explainer(
            model_path) if self.explanation_method != ExplanationMethod.RANDOM_EXPLANATION else None

        test_data = self._load_test_dataset(config)

        results = []
        for idx, row in test_data.iterrows():
            smiles = row['SMILES']
            target = row['Target']

            result = self.evaluate_single_molecule(
                explainer=explainer,
                smiles=smiles,
                target=target,
                data_split_idx=data_split_idx,
                config=config
            )

            results.append(result)

            if idx % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(test_data)} molecules")

        logger.info(f"Saving {len(results)} results")
        self._save_results(results, data_split_idx)

    def _save_results(self, results: List[GTEvalResult], data_split_idx: Optional[int]) -> None:
        """Save target fraction results to CSV"""

        output_dir = os.path.join(
            self.output_folder,
            self.substructure_extractor_name,
            self.explanation_method.value,
        )

        if data_split_idx is not None:
            output_dir = os.path.join(output_dir, f'data_split_{data_split_idx}')

        os.makedirs(output_dir, exist_ok=True)

        data = []
        for result in results:
            row = {
                'smiles': result.smiles,
                'target': result.target,
                'explanation_method': result.explanation_method,
                'data_split_idx': result.data_split_idx,
                'has_target_substructure': result.has_target_substructure,
            }

            # Add metadata as separate columns
            if result.metadata:
                for key, value in result.metadata.items():
                    if isinstance(value, dict):
                        # Flatten nested dictionaries
                        for subkey, subvalue in value.items():
                            row[f'metadata_{key}_{subkey}'] = subvalue
                    else:
                        row[f'metadata_{key}'] = value

            data.append(row)

        df = pd.DataFrame(data)

        output_file = os.path.join(output_dir, 'target_fraction_results.csv')
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

        molecules_with_target = df['has_target_substructure'].sum()
        total_molecules = len(df)
        logger.info(f"Summary: {molecules_with_target}/{total_molecules} molecules contain target substructure")

    def evaluate_all(self) -> None:
        """Evaluate target fractions for all data splits"""
        data_split_paths = self._get_data_split_paths()

        logger.info(f"Starting target fraction evaluation")
        logger.info(f"Explanation method: {self.explanation_method.value}")
        logger.info(f"Substructure extractor: {self.substructure_extractor_name}")
        if self.skip_existing:
            logger.info("Skip existing results: ENABLED")
        else:
            logger.info("Skip existing results: DISABLED")

        for data_split_idx, model_path in data_split_paths:
            self.evaluate_data_split(model_path, data_split_idx)

    @staticmethod
    def _load_config(model_path: str) -> DNNConfig:
        """Load model configuration"""
        config_path = os.path.join(model_path, 'config.yaml')
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found: {config_path}")
        return get_config(config_file_path=config_path)


class GNNFractionEvaluator(BaseFractionEvaluator):
    """Target fraction evaluator for GNN models"""

    SUPPORTED_MODELS = [DNNModelNames.GCN.value, DNNModelNames.GAT.value, DNNModelNames.GIN.value]

    def __init__(self, model_path: str,
                 explanation_method: ExplanationMethod,
                 substructure_extractor_name: str,
                 output_folder: str,
                 explanation_methods_args: Optional[ExplanationMethodArgs] = None,
                 random_seed: int = 42,
                 skip_existing: bool = True):

        super().__init__(model_path, explanation_method, substructure_extractor_name,
                         output_folder, explanation_methods_args, random_seed, skip_existing)

        self.feature_provider = None

    def _load_model_and_config(self, model_path: str) -> Tuple[SupportedModels, DNNConfig]:
        """Load fine-tuned GNN model and config"""
        config = self._load_config(model_path)
        model = self._instantiate_model(config)

        # Load fine-tuned model weights using CheckpointHandler
        checkpoint_handler = CheckpointHandler(training_config=config.training)
        best_model_path = checkpoint_handler.get_best_model_path()

        if best_model_path is None:
            raise ValueError(f"No fine-tuned model found in {model_path}")

        model.load_params(path=best_model_path)
        model.eval()
        model.to(config.training.device)

        return model, config

    def _instantiate_model(self, config: DNNConfig) -> BasicGNNRegressor:
        """Instantiate GNN model based on config"""
        # Set input channels based on used features
        if self.feature_provider is None:
            self.feature_provider = self._get_feature_provider(config)

        config.model.c_in = self.feature_provider.num_node_features

        model_classes = {
            DNNModelNames.GCN.value: GCN,
            DNNModelNames.GAT.value: GAT,
            DNNModelNames.GIN.value: GIN
        }

        model_class = model_classes.get(config.training.model_name)
        if not model_class:
            raise ValueError(
                f"Unsupported GNN model: {config.training.model_name}. Supported models: {self.SUPPORTED_MODELS}")

        return model_class(config=config.model)

    @staticmethod
    def _get_feature_provider(config: DNNConfig) -> PytorchGeometricFeatures:
        """Get feature provider based on config"""
        if config.model.feature_provider == SupportedFeatureProviders.PYTORCH_GEOMETRIC.value:
            return PytorchGeometricFeatures()
        else:
            raise ValueError(f'Feature provider {config.model.feature_provider} is not supported!')

    def _get_explainer(self, model_path: str) -> Optional[Explainer]:
        """Get GNN explainer or None for random explanations"""
        if self.explanation_method == ExplanationMethod.RANDOM_EXPLANATION:
            return None

        return GNNExplainer(
            trained_model_path=model_path,
            explanation_method=self.explanation_method,
            explanation_methods_args=self.explanation_methods_args,
            use_randomized_weights=False,
            use_normalized_attributions=False
        )

    def _create_data(self, smiles: str, target: float, config: Optional[DNNConfig]) -> Data:
        """Create graph data for a single molecule"""
        data = self.feature_provider(smiles)

        # Make artificial batch
        data.batch = torch.zeros(data.x.size(0), dtype=torch.int64)

        data.y = torch.tensor(target, dtype=torch.float)

        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.edge_index = data.edge_index

        return data.to(config.training.device)

    def _get_random_attributions(self, data: Data) -> np.ndarray:
        """Generate random attributions for graph data"""
        if hasattr(data, 'x'):
            n_nodes, n_features = data.x.shape
            return np.random.uniform(0, 1, size=(n_nodes, n_features))
        else:
            raise ValueError(f"Cannot determine shape for random attributions from data type: {type(data)}")


class SequenceFractionEvaluator(BaseFractionEvaluator):
    """Target fraction evaluator for sequence models"""

    def __init__(self, model_path: str,
                 explanation_method: ExplanationMethod,
                 substructure_extractor_name: str,
                 output_folder: str,
                 explanation_methods_args: Optional[ExplanationMethodArgs] = None,
                 random_seed: int = 42,
                 skip_existing: bool = True):

        super().__init__(model_path, explanation_method, substructure_extractor_name,
                         output_folder, explanation_methods_args, random_seed, skip_existing)

    def _load_model_and_config(self, model_path: str):
        """Load fine-tuned sequence model and config"""
        config = self._load_config(model_path)
        tokenizer = SMILESTokenizer().load_vocab(config.model.vocab_path)

        model_name = config.training.model_name

        if model_name == 'CNN':
            model = BasicCNNRegressor(config=config.model, tokenizer=tokenizer)
        elif model_name == 'Transformer':
            model = BasicTransformerRegressor(config=config.model, tokenizer=tokenizer)
        else:
            raise ValueError(f'Unsupported model type: {model_name}')

        # Load fine-tuned model weights using CheckpointHandler
        checkpoint_handler = CheckpointHandler(training_config=config.training)
        best_model_path = checkpoint_handler.get_best_model_path()

        if best_model_path is None:
            raise ValueError(f"No fine-tuned model found in {model_path}")

        model.load_params(path=best_model_path)
        model.eval()
        model.to(config.training.device)

        return model, config

    def _get_explainer(self, model_path: str) -> Optional[Explainer]:
        """Get sequence explainer"""
        if self.explanation_method == ExplanationMethod.RANDOM_EXPLANATION:
            return None

        return SequenceExplainer(
            trained_model_path=model_path,
            explanation_method=self.explanation_method,
            explanation_methods_args=self.explanation_methods_args,
            use_randomized_weights=False,
            use_normalized_attributions=False
        )

    def _create_data(self, smiles: str, target: float, config: Optional[DNNConfig]):
        """Create sequence data"""
        tokenizer = SMILESTokenizer().load_vocab(config.model.vocab_path)

        input_tensor = tokenizer.encode(
            smiles=smiles,
            max_seq_len=config.model.max_seq_len,
            padding=True
        ).unsqueeze(0)

        input_tensor = input_tensor.to(config.training.device)

        return input_tensor

    def _get_random_attributions(self, data: torch.Tensor) -> np.ndarray:
        """Generate random attributions for sequence data"""
        if hasattr(data, 'shape'):
            seq_len = data.shape[0] if len(data.shape) == 1 else data.shape[1]
            return np.random.uniform(0, 1, size=(seq_len,))
        else:
            raise ValueError(f"Cannot determine shape for random attributions from data type: {type(data)}")


def get_fraction_evaluator(model_path: str,
                           explanation_method: ExplanationMethod,
                           substructure_extractor_name: str,
                           output_folder: str,
                           explanation_methods_args: Optional[ExplanationMethodArgs] = None,
                           random_seed: int = 42,
                           skip_existing: bool = True) -> BaseFractionEvaluator:
    """Factory function to get appropriate fraction evaluator based on model architecture"""

    if os.path.isdir(model_path):
        # Check if multi-split or single model
        data_split_folders = [f for f in os.listdir(model_path)
                              if f.startswith('data_split_') and os.path.isdir(os.path.join(model_path, f))]

        if data_split_folders:
            # Multi-split: check first split for architecture
            first_split = sorted(data_split_folders)[0]
            config_path = os.path.join(model_path, first_split, 'config.yaml')
        else:
            # Single model
            config_path = os.path.join(model_path, 'config.yaml')

        if os.path.exists(config_path):
            config = get_config(config_file_path=config_path)
            model_name = config.training.model_name

            if model_name in GNNFractionEvaluator.SUPPORTED_MODELS:
                return GNNFractionEvaluator(model_path, explanation_method, substructure_extractor_name,
                                            output_folder, explanation_methods_args, random_seed, skip_existing)
            else:
                return SequenceFractionEvaluator(model_path, explanation_method, substructure_extractor_name,
                                                 output_folder, explanation_methods_args, random_seed, skip_existing)
        else:
            raise ValueError(f"Config file not found: {config_path}")
    else:
        raise ValueError(f"Model path is not a directory: {model_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Ground Truth Target Fraction Evaluation for Molecular XAI')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model(s) - can be base path or direct model path')
    parser.add_argument('--explanation_method', type=str, required=True,
                        choices=[method.value for method in ExplanationMethod],
                        help='XAI method to evaluate')
    parser.add_argument('--substructure_extractor', type=str, required=True,
                        choices=list(AVAILABLE_EXTRACTORS.keys()),
                        help='Substructure extractor to use for target identification')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Output folder for target fraction results')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false', default=True,
                        help='Force re-evaluation even if results exist (default: skip existing results)')

    args = parser.parse_args()

    evaluator = get_fraction_evaluator(
        model_path=args.model_path,
        explanation_method=ExplanationMethod(args.explanation_method),
        substructure_extractor_name=args.substructure_extractor,
        output_folder=args.output_folder,
        random_seed=args.random_seed,
        skip_existing=args.skip_existing
    )

    evaluator.evaluate_all()


if __name__ == '__main__':
    main()