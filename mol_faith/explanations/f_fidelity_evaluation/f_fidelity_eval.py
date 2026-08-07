import logging
import math
import numpy as np
import os
import pandas as pd
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch_geometric.data import Data
from typing import List, Optional, Dict, Any, Union, Tuple

from mol_faith.data.data_managers.graph_data_manager import PytorchGeometricFeatures, SupportedFeatureProviders
from mol_faith.data.data_managers.smiles_data_manager import SMILESTokenizer
from mol_faith.data.types import DataCategoryType
from mol_faith.explanations.explainer.base_explainer import Explainer
from mol_faith.explanations.explainer.gnn_explainer import GNNExplainer
from mol_faith.explanations.explainer.sequence_explainer import SequenceExplainer
from mol_faith.explanations.types import ExplanationMethod, ExplanationMethodArgs
from mol_faith.masking.mask_generator import GraphFFidelityMasker, SequenceFFidelityMasker, EvaluationStrategy
from mol_faith.model.CNN.cnn_model import BasicCNNRegressor
from mol_faith.model.GNN.base_gnn import BasicGNNRegressor
from mol_faith.model.GNN.gat_model import GAT
from mol_faith.model.GNN.gcn_model import GCN
from mol_faith.model.GNN.gin_model import GIN
from mol_faith.model.configs.dnn_configs import get_config, DNNConfig, DNNModelNames
from mol_faith.model.transformer.transformer import BasicTransformerRegressor
from mol_faith.training.dnn_training.checkpoint_handler import CheckpointHandler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_THRESHOLD_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

type InputData = Union[torch.Tensor, Data]
type Masker = Union[GraphFFidelityMasker, SequenceFFidelityMasker]
type SupportedModels = Union[BasicGNNRegressor, BasicCNNRegressor, BasicTransformerRegressor]


@dataclass
class FFidelityResult:
    smiles: str
    target: float
    threshold_value: float  # Can be mass threshold or percentage threshold
    evaluation_strategy: str  # mass_threshold or top_percentage
    explanation_method: str
    data_split_idx: Optional[int]
    fid_plus: Optional[float]
    fid_minus: Optional[float]
    metadata: Dict[str, Any]


class BaseFidelityEvaluator(ABC):
    def __init__(self,
                 model_path: str,
                 explanation_method: ExplanationMethod,
                 output_folder: str,
                 threshold_values: Optional[List[float]] = None,
                 evaluation_strategies: Optional[List[EvaluationStrategy]] = None,
                 explanation_methods_args: Optional[ExplanationMethodArgs] = None,
                 random_seed: int = 42,
                 skip_existing: bool = True):
        if threshold_values is None:
            threshold_values = DEFAULT_THRESHOLD_VALUES
        if evaluation_strategies is None:
            evaluation_strategies = [EvaluationStrategy.MASS_THRESHOLD, EvaluationStrategy.TOP_PERCENTAGE]
        self.model_path = model_path
        self.explanation_method = explanation_method
        self.output_folder = output_folder
        self.threshold_values = sorted(threshold_values)
        self.evaluation_strategies = evaluation_strategies
        self.explanation_methods_args = explanation_methods_args
        self.random_seed = random_seed
        self.skip_existing = skip_existing
        self._set_seeds()
        self.is_multi_split = self._detect_multi_split_structure()

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
    def _get_masker(self, config: DNNConfig, n_samples: int = 50) -> Masker:
        """Get appropriate masker for this model type"""
        pass

    @abstractmethod
    def _create_data(self, smiles: str, target: float, config) -> InputData:
        """Create data appropriate for this model type"""
        pass

    @abstractmethod
    def _get_random_attributions(self, data: InputData) -> np.ndarray:
        """Generate random attributions matching input shape"""
        pass

    # Shared F-Fidelity evaluation methods
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

    def _check_existing_results(self, threshold: float, evaluation_strategy: EvaluationStrategy,
                                data_split_idx: Optional[int]) -> bool:
        """Check if results already exist for given threshold, evaluation type and data split"""
        output_dir = os.path.join(
            self.output_folder,
            self.explanation_method.value,
            evaluation_strategy.value,
            f'threshold_{threshold}'
        )

        if data_split_idx is not None:
            output_dir = os.path.join(output_dir, f'data_split_{data_split_idx}')

        output_file = os.path.join(output_dir, 'fidelity_results.csv')

        exists = os.path.exists(output_file)
        if exists:
            logger.info(f"Found existing results for {evaluation_strategy.value}={threshold}, "
                        f"data_split={data_split_idx} at {output_file}")

        return exists

    def _filter_thresholds_for_split(self, evaluation_type: EvaluationStrategy,
                                     data_split_idx: Optional[int]) -> List[float]:
        """Filter thresholds to only include those without existing results"""
        if not self.skip_existing:
            return self.threshold_values

        filtered_thresholds = []
        skipped_count = 0

        for threshold in self.threshold_values:
            if self._check_existing_results(threshold, evaluation_type, data_split_idx):
                skipped_count += 1
                logger.info(
                    f"Skipping {evaluation_type.value}={threshold} for data_split={data_split_idx} (already exists)")
            else:
                filtered_thresholds.append(threshold)

        if skipped_count > 0:
            logger.info(
                f"Skipped {skipped_count}/{len(self.threshold_values)} {evaluation_type.value} thresholds "
                f"for data_split={data_split_idx}")

        return filtered_thresholds

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
    def _calculate_tanh_fidelity_scores(model: SupportedModels, data: InputData,
                                        masks: torch.Tensor, target: float) -> List[float]:
        """
        Calculate F-Fidelity scores using tanh transformation

        Returns values in [-1, 1] where:
        - Positive values: masking hurt performance (good explanation)
        - Zero: no performance change (weak explanation)
        - Negative values: masking helped performance (suspicious explanation)
        """
        scores = []

        # Get original prediction and MAE
        with torch.no_grad():
            original_pred = model(data).item()
            original_mae = abs(original_pred - target)

        # Evaluate each mask
        for mask in masks:
            with torch.no_grad():
                masked_pred = model(data, mask).item()
                masked_mae = abs(masked_pred - target)

                # Calculate relative change in error
                relative_change = (masked_mae - original_mae) / (original_mae + 1e-8)

                # Apply tanh transformation for bounded, interpretable scores
                tanh_score = math.tanh(relative_change)
                scores.append(tanh_score)

        return scores

    @staticmethod
    def _calculate_relative_fidelity_scores(model: SupportedModels, data: InputData,
                                            masks: torch.Tensor, target: float) -> List[float]:
        """
        Calculate F-Fidelity scores using tanh transformation

        Returns values in [-1, 1] where:
        - Positive values: masking hurt performance (good explanation)
        - Zero: no performance change (weak explanation)
        - Negative values: masking helped performance (suspicious explanation)
        """
        scores = []

        # Get original prediction and MAE
        with torch.no_grad():
            original_pred = model(data).item()
            original_mae = abs(original_pred - target)

        # Evaluate each mask
        for mask in masks:
            with torch.no_grad():
                masked_pred = model(data, mask).item()
                masked_mae = abs(masked_pred - target)

                # Calculate relative change in error
                relative_change = (masked_mae - original_mae) / (original_mae + 1e-8)

                scores.append(relative_change)

        return scores

    @staticmethod
    def _calculate_squared_error_fidelity_scores(model: SupportedModels, data: InputData,
                                                 masks: torch.Tensor, target: float) -> List[float]:
        """
        Calculate F-Fidelity scores using tanh transformation

        Returns values in [-1, 1] where:
        - Positive values: masking hurt performance (good explanation)
        - Zero: no performance change (weak explanation)
        - Negative values: masking helped performance (suspicious explanation)
        """
        scores = []

        # Get original prediction and MAE
        with torch.no_grad():
            original_pred = model(data).item()
            original_mse = (original_pred - target) ** 2

        # Evaluate each mask
        for mask in masks:
            with torch.no_grad():
                masked_pred = model(data, mask).item()
                masked_mse = (masked_pred - target) ** 2

                mse_change = masked_mse - original_mse

                scores.append(mse_change)

        return scores

    @staticmethod
    def _calculate_absolute_error_fidelity_scores(model: SupportedModels, data: InputData,
                                                  masks: torch.Tensor, target: float) -> List[float]:
        """
        Calculate F-Fidelity scores using tanh transformation

        Returns values in [-1, 1] where:
        - Positive values: masking hurt performance (good explanation)
        - Zero: no performance change (weak explanation)
        - Negative values: masking helped performance (suspicious explanation)
        """
        scores = []

        # Get original prediction and MAE
        with torch.no_grad():
            original_pred = model(data).item()
            original_abs_error = abs(original_pred - target)

        # Evaluate each mask
        for mask in masks:
            with torch.no_grad():
                masked_pred = model(data, mask).item()
                masked_abs_error = abs(masked_pred - target)

                abs_error_change = masked_abs_error - original_abs_error

                scores.append(abs_error_change)

        return scores

    def evaluate_single_threshold(self,
                                  model: SupportedModels,
                                  masker: Masker,
                                  data: InputData,
                                  attributions: np.ndarray,
                                  smiles: str,
                                  target: float,
                                  threshold_value: float,
                                  evaluation_strategy: EvaluationStrategy,
                                  data_split_idx: Optional[int],
                                  config: DNNConfig) -> FFidelityResult:
        """
        Evaluate F-Fidelity for a single threshold using pre-computed attributions
        """
        attribution_tensor = torch.tensor(attributions) if not isinstance(attributions, torch.Tensor) else attributions
        attribution_tensor = attribution_tensor.to(config.training.device)

        # If the xAI method produces only 0 attributions, save the result with corresponding metadata
        if torch.all(attribution_tensor == 0):
            return FFidelityResult(
                smiles=smiles,
                target=target,
                threshold_value=threshold_value,
                explanation_method=self.explanation_method.value,
                data_split_idx=data_split_idx,
                evaluation_strategy=evaluation_strategy.value,
                fid_plus=None,
                fid_minus=None,
                metadata={'extreme_case': True, 'reason': 'all_zero_attributions'})

        # Use masker to create evaluation masks for specific threshold and strategy
        mask_results = masker.create_evaluation_masks(data=data,
                                                      attribution_scores=attribution_tensor,
                                                      threshold_value=threshold_value,
                                                      strategy=evaluation_strategy)

        if mask_results is None:
            # Extreme case: s=0 or s=td
            return FFidelityResult(
                smiles=smiles,
                target=target,
                threshold_value=threshold_value,
                explanation_method=self.explanation_method.value,
                data_split_idx=data_split_idx,
                evaluation_strategy=evaluation_strategy.value,
                fid_plus=None,
                fid_minus=None,
                metadata={'extreme_case': True, 'reason': 'Cannot evaluate: s=0 or s=td'})

        fid_plus_masks = mask_results['fid_plus_masks']
        fid_minus_masks = mask_results['fid_minus_masks']
        metadata = mask_results['metadata']

        # Calculate F-Fidelity scores
        fid_plus_scores = self._calculate_absolute_error_fidelity_scores(model, data, fid_plus_masks, target)
        fid_minus_scores = self._calculate_absolute_error_fidelity_scores(model, data, fid_minus_masks, target)

        fid_plus = np.mean(fid_plus_scores, dtype=np.float64)
        fid_minus = np.mean(fid_minus_scores, dtype=np.float64)

        metadata.update({
            'fid_plus_samples': len(fid_plus_scores),
            'fid_minus_samples': len(fid_minus_scores),
            'fid_plus_std': np.std(fid_plus_scores),
            'fid_minus_std': np.std(fid_minus_scores),
        })

        return FFidelityResult(
            smiles=smiles,
            target=target,
            threshold_value=threshold_value,
            explanation_method=self.explanation_method.value,
            data_split_idx=data_split_idx,
            evaluation_strategy=evaluation_strategy.value,
            fid_plus=fid_plus,
            fid_minus=fid_minus,
            metadata=metadata)

    def evaluate_data_split(self, model_path: str, data_split_idx: Optional[int]) -> None:
        """
        Evaluate entire test dataset for a single data split

        Flow:
        1. Filter mass thresholds to skip existing results
        2. For each molecule:
           a. Compute attributions
           b. For each remaining mass threshold: use attributions, create masks, evaluate
        3. Collect results by threshold and save separately
        """
        logger.info(f"Evaluating data split {data_split_idx} at {model_path} "
                    f"for xAI method {self.explanation_method.value}")
        logger.info(f"Evaluation strategies: "
                    f"{[evaluation_strat.value for evaluation_strat in self.evaluation_strategies]}")

        # Collect thresholds to evaluate for each evaluation type. Skip if the threshold exists and if skipping is enabled
        thresholds_to_evaluate = {}
        for evaluation_strategy in self.evaluation_strategies:
            filtered_thresholds = self._filter_thresholds_for_split(evaluation_strategy, data_split_idx)
            if filtered_thresholds:
                thresholds_to_evaluate[evaluation_strategy] = filtered_thresholds
                logger.info(f"For strategy {evaluation_strategy.value}: "
                            f"Evaluate {len(thresholds_to_evaluate[evaluation_strategy])}/{len(self.threshold_values)} "
                            f"thresholds: {thresholds_to_evaluate[evaluation_strategy]}")
            else:
                logger.info(f"All {evaluation_strategy.value} thresholds already evaluated for "
                            f"data_split={data_split_idx}. Skipping.")

        if not thresholds_to_evaluate:
            logger.info(f"No thresholds to evaluate for data_split={data_split_idx}. Skipping.")
            return

        model, config = self._load_model_and_config(model_path)
        explainer = self._get_explainer(model_path)
        masker = self._get_masker(config)
        test_data = self._load_test_dataset(config)

        # Prepare storing of results for the different strategies
        results_by_evaluation_strategy = {}
        for evaluation_strategy in thresholds_to_evaluate.keys():
            results_by_evaluation_strategy[evaluation_strategy] = {
                threshold: [] for threshold in thresholds_to_evaluate[evaluation_strategy]
            }

        for idx, row in test_data.iterrows():
            smiles = row['SMILES']
            target = row['Target']

            data = self._create_data(smiles, target, config)

            if self.explanation_method == ExplanationMethod.RANDOM_EXPLANATION:
                attributions = self._get_random_attributions(data)
            else:
                attributions = self._get_attributions(smiles, target, explainer)

            # Evaluate all thresholds for all evaluation types
            for evaluation_strategy, thresholds in thresholds_to_evaluate.items():
                for threshold in thresholds:
                    result = self.evaluate_single_threshold(
                        model=model,
                        masker=masker,
                        data=data,
                        attributions=attributions,
                        smiles=smiles,
                        target=target,
                        threshold_value=threshold,
                        evaluation_strategy=evaluation_strategy,
                        data_split_idx=data_split_idx,
                        config=config)
                    results_by_evaluation_strategy[evaluation_strategy][threshold].append(result)

            if idx % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(test_data)} molecules")

        # Save results for each evaluation type and threshold
        for evaluation_type, results_by_threshold in results_by_evaluation_strategy.items():
            for threshold, results in results_by_threshold.items():
                if results:  # Only save if we have results
                    logger.info(f"Saving {len(results)} results for {evaluation_type.value} threshold {threshold}")
                    self._save_results(results, threshold, evaluation_type, data_split_idx)

    def _save_results(self, results: List[FFidelityResult], threshold: float,
                      evaluation_strategy: EvaluationStrategy, data_split_idx: Optional[int]) -> None:
        """Save F-Fidelity results to CSV with evaluation type organization"""

        output_dir = os.path.join(
            self.output_folder,
            evaluation_strategy.value,
            self.explanation_method.value,
            f'threshold_{threshold}'
        )

        if data_split_idx is not None:
            output_dir = os.path.join(output_dir, f'data_split_{data_split_idx}')

        os.makedirs(output_dir, exist_ok=True)

        data = []
        for result in results:
            row = {
                'smiles': result.smiles,
                'target': result.target,
                'threshold_value': result.threshold_value,
                'evaluation_strategy': result.evaluation_strategy,
                'explanation_method': result.explanation_method,
                'data_split_idx': result.data_split_idx,
                'fid_plus': result.fid_plus,
                'fid_minus': result.fid_minus,
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

        output_file = os.path.join(output_dir, 'fidelity_results.csv')
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

    def evaluate_all(self) -> None:
        """Evaluate F-Fidelity for all data splits"""
        data_split_paths = self._get_data_split_paths()

        logger.info(f"Starting F-Fidelity evaluation")
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


class GNNFidelityEvaluator(BaseFidelityEvaluator):
    """F-Fidelity evaluator for GNN models"""

    SUPPORTED_MODELS = [DNNModelNames.GCN.value, DNNModelNames.GAT.value, DNNModelNames.GIN.value]

    def __init__(self, model_path: str,
                 explanation_method: ExplanationMethod,
                 output_folder: str,
                 thresholds: Optional[List[float]] = None,
                 evaluation_strategies: Optional[List[EvaluationStrategy]] = None,
                 explanation_methods_args: Optional[ExplanationMethodArgs] = None,
                 random_seed: int = 42,
                 skip_existing: bool = True):

        super().__init__(model_path, explanation_method, output_folder, thresholds,
                         evaluation_strategies, explanation_methods_args, random_seed, skip_existing)
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

    def _get_masker(self, config: DNNConfig, n_samples: int = 50) -> Masker:
        """Get graph masker"""
        return GraphFFidelityMasker(beta=config.training.finetune_beta, n_samples=n_samples,
                                    device=config.training.device)

    def _create_data(self, smiles: str, target: float, config: DNNConfig) -> Data:
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


class SequenceFidelityEvaluator(BaseFidelityEvaluator):
    """F-Fidelity evaluator for CNN models - Placeholder implementation"""

    def __init__(self, model_path: str,
                 explanation_method: ExplanationMethod,
                 output_folder: str,
                 thresholds: Optional[List[float]] = None,
                 evaluation_strategies: Optional[List[EvaluationStrategy]] = None,
                 explanation_methods_args: Optional[ExplanationMethodArgs] = None,
                 random_seed: int = 42,
                 skip_existing: bool = True):

        super().__init__(model_path, explanation_method, output_folder, thresholds,
                         evaluation_strategies, explanation_methods_args, random_seed, skip_existing)

    def _load_model_and_config(self, model_path: str):
        """Load fine-tuned CNN model and config"""
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
        """Get CNN explainer"""
        if self.explanation_method == ExplanationMethod.RANDOM_EXPLANATION:
            return None

        return SequenceExplainer(
            trained_model_path=model_path,
            explanation_method=self.explanation_method,
            explanation_methods_args=self.explanation_methods_args,
            use_randomized_weights=False,
            use_normalized_attributions=False
        )

    def _get_masker(self, config, n_samples: int = 50):
        """Get sequence masker"""
        tokenizer = SMILESTokenizer().load_vocab(config.model.vocab_path)
        return SequenceFFidelityMasker(tokenizer=tokenizer, beta=config.training.finetune_beta, n_samples=n_samples,
                                       device=config.training.device)

    def _create_data(self, smiles: str, target: float, config: DNNConfig):
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


def get_fidelity_evaluator(model_path: str,
                           explanation_method: ExplanationMethod,
                           output_folder: str,
                           threshold_values: Optional[List[float]] = None,
                           evaluation_strategies: Optional[List[EvaluationStrategy]] = None,
                           explanation_methods_args: Optional[ExplanationMethodArgs] = None,
                           random_seed: int = 42,
                           skip_existing: bool = True) -> BaseFidelityEvaluator:
    """Factory function to get appropriate F-Fidelity evaluator based on model architecture"""

    if threshold_values is None:
        threshold_values = DEFAULT_THRESHOLD_VALUES

    if evaluation_strategies is None:
        evaluation_strategies = [EvaluationStrategy.MASS_THRESHOLD, EvaluationStrategy.TOP_PERCENTAGE]

    if os.path.isdir(model_path):
        # Check if multi-split or single model
        data_split_folders = [f for f in os.listdir(model_path) if f.startswith('data_split_') and
                              os.path.isdir(os.path.join(model_path, f))]

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

            if model_name in GNNFidelityEvaluator.SUPPORTED_MODELS:
                return GNNFidelityEvaluator(model_path, explanation_method, output_folder,
                                            threshold_values, evaluation_strategies, explanation_methods_args,
                                            random_seed, skip_existing)
            else:
                return SequenceFidelityEvaluator(model_path, explanation_method, output_folder,
                                                 threshold_values, evaluation_strategies, explanation_methods_args,
                                                 random_seed, skip_existing)
        else:
            raise ValueError(f"Config file not found: {config_path}")
    else:
        raise ValueError(f"Model path is not a directory: {model_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Optimized F-Fidelity Evaluation for Molecular XAI')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model(s) - can be base path or direct model path')
    parser.add_argument('--explanation_method', type=str, required=True,
                        choices=[method.value for method in ExplanationMethod],
                        help='XAI method to evaluate')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Output folder for F-Fidelity results')
    parser.add_argument('--threshold_values', nargs='+', type=float,
                        default=DEFAULT_THRESHOLD_VALUES,
                        help='Thresholds for explanation selection (used for the different evaluation strategies')
    parser.add_argument('--evaluation_strategies', nargs='+',
                        choices=[et.value for et in EvaluationStrategy],
                        default=[et.value for et in EvaluationStrategy],
                        help='Types of evaluation to run (default: both)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false', default=True,
                        help='Force re-evaluation of all mass thresholds, even if results exist (default: skip existing results)')

    args = parser.parse_args()

    evaluation_strategies = [EvaluationStrategy(eval_strat) for eval_strat in args.evaluation_strategies]

    evaluator = get_fidelity_evaluator(
        model_path=args.model_path,
        explanation_method=ExplanationMethod(args.explanation_method),
        evaluation_strategies=evaluation_strategies,
        output_folder=args.output_folder,
        threshold_values=args.threshold_values,
        random_seed=args.random_seed,
        skip_existing=args.skip_existing
    )

    evaluator.evaluate_all()


if __name__ == '__main__':
    main()