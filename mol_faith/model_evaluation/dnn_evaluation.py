from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import torch
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch import nn
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from mol_faith.data.data_managers.graph_data_manager import GraphDataManager, PytorchGeometricFeatures, \
    SupportedFeatureProviders
from mol_faith.data.data_managers.smiles_data_manager import SMILESTokenizer, SMILESSequenceDataManager
from mol_faith.data.types import DataCategoryType
from mol_faith.masking.mask_generator import GraphFFidelityMasker, SequenceFFidelityMasker
from mol_faith.model.CNN.cnn_model import BasicCNNRegressor
from mol_faith.model.GNN.gat_model import GAT
from mol_faith.model.GNN.gcn_model import GCN
from mol_faith.model.GNN.gin_model import GIN
from mol_faith.model.configs.dnn_configs import get_config, DNNModelNames, DNNConfig
from mol_faith.model.transformer.transformer import BasicTransformerRegressor
from mol_faith.model_evaluation.model_evaluation import ModelEvaluator, EvaluationMetrics, calc_f3_metric
from mol_faith.training.dnn_training.checkpoint_handler import CheckpointHandler


@dataclass
class ModelMaskingDegradationResults:
    """Results for masking degradation analysis of a single model"""
    model_name: str
    model_path: str
    beta_range: List[float]  # 0 to max_beta
    model_variants: Dict[
        str, Dict[str, List[float]]]  # variant_name -> {metrics: {metric: [values]}, std: {metric: [values]}}
    num_splits: int


@dataclass
class MultiModelPerformanceResults:
    """Results for multiple models without beta analysis"""
    model_results: Dict[str, EvaluationMetrics]  # model_name -> metrics
    model_results_std: Dict[str, EvaluationMetrics]  # model_name -> std metrics
    num_splits: int


@dataclass
class SingleModelPerformanceResults:
    """Results for a single model performance evaluation"""
    model_name: str
    model_path: str
    metrics: EvaluationMetrics
    metrics_std: EvaluationMetrics
    num_splits: int


class DNNEvaluator(ModelEvaluator):
    def evaluate_model(self, base_result_path: str, data_category: str) -> EvaluationMetrics:
        pass

    def __init__(self, model_type: str = 'DNN', figsize=(16, 12)):
        super().__init__(model_type=model_type)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metric_names = ['mse', 'rmse', 'mae', 'r2', 'pearson_r', 'f3']
        self.figsize = figsize
        sns.set_palette("husl")

    @staticmethod
    def _get_tokenizer(config: DNNConfig):
        return SMILESTokenizer.load_vocab(config.model.vocab_path)

    @staticmethod
    def detect_model_architecture(config: DNNConfig) -> str:
        """Automatically detect if this is a graph or sequence model"""
        model_name = config.training.model_name.upper()
        if model_name in ['GCN', 'GAT', 'GIN']:
            return "graph"
        elif model_name in ['CNN', 'TRANSFORMER']:
            return "sequence"
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    def _get_model(self, config: DNNConfig, model_architecture: str = None):
        """Get model based on architecture type"""
        if model_architecture is None:
            model_architecture = self.detect_model_architecture(config)

        model_name = config.training.model_name

        if model_architecture == "graph":
            if model_name == DNNModelNames.GCN.value:
                model = GCN(config=config.model)
            elif model_name == DNNModelNames.GAT.value:
                model = GAT(config=config.model)
            elif model_name == DNNModelNames.GIN.value:
                model = GIN(config=config.model)
            else:
                raise ValueError(f"Unknown graph model: {model_name}")
        elif model_architecture == "sequence":
            if model_name == DNNModelNames.CNN.value:
                model = BasicCNNRegressor(config=config.model, tokenizer=self._get_tokenizer(config))
            elif model_name == DNNModelNames.Transformer.value:
                model = BasicTransformerRegressor(config=config.model, tokenizer=self._get_tokenizer(config))
            else:
                raise ValueError(f"Unknown sequence model: {model_name}")
        else:
            raise ValueError(f"Unknown architecture: {model_architecture}")

        return model

    def _get_data_manager(self, config: DNNConfig, model_architecture: str = None):
        """Get appropriate data manager based on architecture"""
        if model_architecture is None:
            model_architecture = self.detect_model_architecture(config)

        if model_architecture == "graph":
            feature_provider = self._get_feature_provider(config)
            data_manager = GraphDataManager(
                data_path=config.training.dataset_folder_path,
                batch_size=1,
                feature_provider=feature_provider
            )
        elif model_architecture == "sequence":
            data_manager = SMILESSequenceDataManager(
                data_path=config.training.dataset_folder_path,
                batch_size=1,
                max_seq_len=config.model.max_seq_len,
                tokenizer=self._get_tokenizer(config))
        else:
            raise ValueError(f"Unknown architecture: {model_architecture}")

        return data_manager

    @staticmethod
    def _get_feature_provider(config: DNNConfig):
        if config.model.feature_provider == SupportedFeatureProviders.PYTORCH_GEOMETRIC.value:
            feature_provider = PytorchGeometricFeatures()
        else:
            raise ValueError(f'Feature provider {config.model.feature_provider} not supported!')
        return feature_provider

    def _get_masker(self, config: DNNConfig, model_architecture: str, beta: float):
        """Get appropriate masker based on architecture"""
        if model_architecture == "graph":
            masker = GraphFFidelityMasker(beta=beta)
        elif model_architecture == "sequence":
            masker = SequenceFFidelityMasker(tokenizer=self._get_tokenizer(config), beta=beta)
        else:
            raise ValueError(f"Unknown architecture: {model_architecture}")
        return masker

    def _find_data_splits(self, base_path: str) -> List[str]:
        """Find all data_split_x folders in the base directory"""
        split_folders = []
        if not os.path.exists(base_path):
            raise ValueError(f"Base path does not exist: {base_path}")

        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and item.startswith('data_split_'):
                split_folders.append(item_path)

        split_folders.sort()
        return split_folders

    def _find_beta_folders(self, finetuning_base_path: str, beta_values: List[float]) -> List[Tuple[float, str]]:
        """Find beta folders matching specified beta values and return sorted list of (beta_value, path) tuples"""
        beta_folders = []
        if not os.path.exists(finetuning_base_path):
            raise ValueError(f"Finetuning base path does not exist: {finetuning_base_path}")

        for item in os.listdir(finetuning_base_path):
            item_path = os.path.join(finetuning_base_path, item)
            if os.path.isdir(item_path) and item.startswith('beta_'):
                # Extract beta value from folder name
                beta_match = re.match(r'beta_(\d+\.?\d*)', item)
                if beta_match:
                    beta_value = float(beta_match.group(1))
                    # Only include if in specified beta_values
                    if beta_value in beta_values:
                        beta_folders.append((beta_value, item_path))

        # Sort by beta value
        beta_folders.sort(key=lambda x: x[0])
        return beta_folders

    def _load_model_from_path(self, model_result_path: str) -> Tuple[nn.Module, str, DNNConfig]:
        """Load a model from its result path"""
        config = get_config(config_file_path=os.path.join(model_result_path, 'config.yaml'))
        model_architecture = self.detect_model_architecture(config)

        # Set up model
        if model_architecture == "graph":
            feature_provider = self._get_feature_provider(config=config)
            config.model.c_in = feature_provider.num_node_features

        # Use CheckpointHandler to find the best model
        checkpoint_handler = CheckpointHandler(training_config=config.training)
        model_path = checkpoint_handler.get_best_model_path()
        if model_path is None:
            raise ValueError(f"No best model found in {model_result_path}")

        model = self._get_model(config=config, model_architecture=model_architecture)
        model.load_params(model_path)
        model.eval()
        model.to(self.device)

        return model, model_architecture, config

    def _evaluate_single_model_split(self, model: nn.Module, data_loader, model_architecture: str,
                                     config: DNNConfig, use_masking: bool = False,
                                     beta: float = 0.1) -> EvaluationMetrics:
        """Evaluate a single model on a single data split"""

        # Get targets for F3 calculation
        if model_architecture == "graph":
            targets = [data.y for data in data_loader.dataset]
        else:
            targets = [target for data, target in data_loader.dataset]

        # Get train targets for F3 metric
        data_manager = self._get_data_manager(config=config, model_architecture=model_architecture)
        train_data_loader = data_manager.load_data(
            data_category_type=DataCategoryType.TRAIN.value,
            target_column_name=config.training.target_column_name
        )
        if model_architecture == "graph":
            train_targets = [data.y for data in train_data_loader.dataset]
        else:
            train_targets = [target for data, target in train_data_loader.dataset]

        # Get masker if needed
        masker = self._get_masker(config, model_architecture, beta) if use_masking else None

        preds = []
        mse_values = []
        mae_values = []

        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            with torch.no_grad():
                if model_architecture == "graph":
                    batch = batch.to(self.device)
                    if use_masking:
                        mask = masker.create_finetuning_masks(batch)
                        prediction = model(batch, mask)
                    else:
                        prediction = model(batch)
                    target = batch.y
                else:  # sequence
                    inputs, target = batch
                    inputs = inputs.to(self.device)
                    if use_masking:
                        mask = masker.create_finetuning_masks(inputs)
                        prediction = model(inputs, mask)
                    else:
                        prediction = model(inputs)

            target = target.cpu()
            prediction = prediction.cpu()

            mse = mean_squared_error(y_true=target, y_pred=prediction)
            mae = mean_absolute_error(y_true=target, y_pred=prediction)

            mse_values.append(mse)
            mae_values.append(mae)
            preds.append(prediction)

        preds = np.array(preds)

        if model_architecture == "graph":
            targets = np.array(targets)
            train_true = np.array(train_targets)
        else:
            # Adjust shape to (num_data, 1)
            targets = np.array(targets).reshape((-1, 1))
            train_true = np.array(train_targets).reshape((-1, 1))

        metrics = EvaluationMetrics()
        metrics.mse = np.mean(mse_values)
        metrics.rmse = np.sqrt(metrics.mse)
        metrics.mae = np.mean(mae_values)
        metrics.pearson_r = stats.pearsonr(x=preds, y=targets)[0]
        metrics.r2 = r2_score(y_pred=preds, y_true=targets)
        metrics.f3 = calc_f3_metric(train_true=train_true, val_pred=preds, val_true=targets)

        return metrics

    def evaluate_single_model_performance(self,
                                          model_name: str,
                                          model_base_path: str,
                                          data_category: str = "test") -> SingleModelPerformanceResults:
        """
        Evaluate performance of a single model across multiple data splits

        Args:
            model_name: Name of the model for identification
            model_base_path: Base path containing data_split_x folders
            data_category: Data split to evaluate on (train/val/test)

        Returns:
            SingleModelPerformanceResults containing metrics and statistics
        """
        print(f"Evaluating single model performance: {model_name}")

        # Find data splits
        split_folders = self._find_data_splits(model_base_path)
        if not split_folders:
            raise ValueError(f"No data splits found in {model_base_path}")

        print(f"Found {len(split_folders)} data splits")

        # Evaluate all splits
        split_metrics = []

        for split_folder in split_folders:
            print(f"  Evaluating split: {os.path.basename(split_folder)}")
            model, model_arch, config = self._load_model_from_path(split_folder)

            # Load data
            data_manager = self._get_data_manager(config, model_arch)
            data_loader = data_manager.load_data(
                data_category_type=data_category,
                target_column_name=config.training.target_column_name
            )

            # Evaluate model (no masking for basic performance evaluation)
            metrics = self._evaluate_single_model_split(
                model, data_loader, model_arch, config, use_masking=False
            )
            split_metrics.append(metrics)

        # Calculate mean and std across splits
        mean_metrics = EvaluationMetrics()
        std_metrics = EvaluationMetrics()

        for metric in self.metric_names:
            values = [getattr(m, metric) for m in split_metrics]
            setattr(mean_metrics, metric, np.mean(values))
            setattr(std_metrics, metric, np.std(values))

        print(f"  {model_name} - RMSE: {mean_metrics.rmse:.4f} ± {std_metrics.rmse:.4f}")
        print(f"  {model_name} - R²: {mean_metrics.r2:.4f} ± {std_metrics.r2:.4f}")
        print(f"  {model_name} - F3: {mean_metrics.f3:.4f} ± {std_metrics.f3:.4f}")
        print(f"  {model_name} - MAE: {mean_metrics.mae:.4f} ± {std_metrics.mae:.4f}")


        return SingleModelPerformanceResults(
            model_name=model_name,
            model_path=model_base_path,
            metrics=mean_metrics,
            metrics_std=std_metrics,
            num_splits=len(split_folders)
        )

    def evaluate_masking_degradation_analysis(self,
                                              model_configs: List[Tuple[str, str, str]],
                                              beta_values: List[float],
                                              data_category: str = "test") -> List[ModelMaskingDegradationResults]:
        """
        Function 1: Evaluate how performance degrades with increasing masking levels for multiple models

        For each model, evaluates:
        - Pretrained model under masking levels 0 to max_beta
        - Each finetuned variant (beta_0.1, beta_0.2, etc.) under masking levels 0 to max_beta

        Args:
            model_configs: List of (model_name, pretrained_path, finetuning_base_path) tuples
            beta_values: List of beta values to use for evaluation (e.g., [0.1, 0.2, 0.3])
            data_category: Data split to evaluate on

        Returns:
            List of ModelMaskingDegradationResults for each model
        """
        print("Evaluating masking degradation analysis")

        all_results = []

        for model_name, pretrained_path, finetuning_base_path in model_configs:
            print(f"\nProcessing model: {model_name}")

            # Find beta folders for specified beta values only
            beta_folders = self._find_beta_folders(finetuning_base_path, beta_values)
            if not beta_folders:
                raise ValueError(
                    f"No matching beta folders found in {finetuning_base_path} for beta values {beta_values}")

            finetuning_beta_values = [beta for beta, _ in beta_folders]
            print(f"Using finetuning beta values: {finetuning_beta_values}")

            # Create full beta range: 0 (no masking) to specified betas
            full_beta_range = [0.0] + finetuning_beta_values
            print(f"Full beta evaluation range: {full_beta_range}")

            # Load pretrained model splits
            pretrained_splits = self._find_data_splits(pretrained_path)
            num_splits = len(pretrained_splits)

            # Initialize result containers for all model variants
            model_variants = {}

            # Add pretrained model
            model_variants['Pretrained'] = {
                'metrics': {metric: [] for metric in self.metric_names},
                'std': {metric: [] for metric in self.metric_names}
            }

            # Add each finetuned variant
            for beta, _ in beta_folders:
                variant_name = f'Finetuned_β{beta}'
                model_variants[variant_name] = {
                    'metrics': {metric: [] for metric in self.metric_names},
                    'std': {metric: [] for metric in self.metric_names}
                }

            # Evaluate each model variant across all beta levels
            for eval_beta in full_beta_range:
                print(f"  Evaluating at masking level β={eval_beta}")

                # Evaluate pretrained model at this masking level
                pretrained_split_results = []
                for pretrained_split in pretrained_splits:
                    pretrained_model, model_arch, config = self._load_model_from_path(pretrained_split)

                    data_manager = self._get_data_manager(config, model_arch)
                    data_loader = data_manager.load_data(
                        data_category_type=data_category,
                        target_column_name=config.training.target_column_name
                    )

                    use_masking = eval_beta > 0
                    result = self._evaluate_single_model_split(
                        pretrained_model, data_loader, model_arch, config,
                        use_masking=use_masking, beta=eval_beta
                    )
                    pretrained_split_results.append(result)

                # Store pretrained results
                for metric in self.metric_names:
                    values = [getattr(r, metric) for r in pretrained_split_results]
                    model_variants['Pretrained']['metrics'][metric].append(np.mean(values))
                    model_variants['Pretrained']['std'][metric].append(np.std(values))

                # Evaluate each finetuned variant at this masking level
                for beta_ft, beta_folder_path in beta_folders:
                    variant_name = f'Finetuned_β{beta_ft}'

                    finetuned_splits = self._find_data_splits(beta_folder_path)
                    if len(finetuned_splits) != num_splits:
                        raise ValueError(f"Mismatch in splits for {variant_name}")

                    finetuned_split_results = []
                    for finetuned_split in finetuned_splits:
                        finetuned_model, model_arch, config = self._load_model_from_path(finetuned_split)

                        data_manager = self._get_data_manager(config, model_arch)
                        data_loader = data_manager.load_data(
                            data_category_type=data_category,
                            target_column_name=config.training.target_column_name
                        )

                        use_masking = eval_beta > 0
                        result = self._evaluate_single_model_split(
                            finetuned_model, data_loader, model_arch, config,
                            use_masking=use_masking, beta=eval_beta
                        )
                        finetuned_split_results.append(result)

                    # Store finetuned results
                    for metric in self.metric_names:
                        values = [getattr(r, metric) for r in finetuned_split_results]
                        model_variants[variant_name]['metrics'][metric].append(np.mean(values))
                        model_variants[variant_name]['std'][metric].append(np.std(values))

            # Create result object
            result = ModelMaskingDegradationResults(
                model_name=model_name,
                model_path=finetuning_base_path,
                beta_range=full_beta_range,
                model_variants=model_variants,
                num_splits=num_splits
            )

            all_results.append(result)

        return all_results

    def evaluate_multi_model_performance(self,
                                         model_configs: List[Tuple[str, str]],
                                         data_category: str = "test") -> MultiModelPerformanceResults:
        """
        Function 2: Compare performance across different model architectures (no beta analysis)

        Args:
            model_configs: List of (model_name, model_base_path) tuples
            data_category: Data split to evaluate on

        Returns:
            MultiModelPerformanceResults containing comparison across models
        """
        print("Evaluating multi-model performance comparison")

        model_results = {}
        model_results_std = {}
        num_splits = None

        for model_name, model_base_path in model_configs:
            print(f"\nProcessing model: {model_name}")

            # Find data splits
            split_folders = self._find_data_splits(model_base_path)
            if not split_folders:
                raise ValueError(f"No data splits found in {model_base_path}")

            if num_splits is None:
                num_splits = len(split_folders)
            elif num_splits != len(split_folders):
                raise ValueError(f"Model {model_name} has {len(split_folders)} splits, expected {num_splits}")

            # Evaluate all splits
            split_metrics = []

            for split_folder in split_folders:
                model, model_arch, config = self._load_model_from_path(split_folder)

                # Load data
                data_manager = self._get_data_manager(config, model_arch)
                data_loader = data_manager.load_data(
                    data_category_type=data_category,
                    target_column_name=config.training.target_column_name
                )

                # Evaluate model
                metrics = self._evaluate_single_model_split(
                    model, data_loader, model_arch, config, use_masking=False
                )
                split_metrics.append(metrics)

            # Calculate mean and std across splits
            mean_metrics = EvaluationMetrics()
            std_metrics = EvaluationMetrics()

            for metric in self.metric_names:
                values = [getattr(m, metric) for m in split_metrics]
                setattr(mean_metrics, metric, np.mean(values))
                setattr(std_metrics, metric, np.std(values))

            model_results[model_name] = mean_metrics
            model_results_std[model_name] = std_metrics

            print(f"  {model_name} - RMSE: {mean_metrics.rmse:.4f} ± {std_metrics.rmse:.4f}")

        return MultiModelPerformanceResults(
            model_results=model_results,
            model_results_std=model_results_std,
            num_splits=num_splits
        )

    def save_multi_model_results_to_csv(self,
                                        results: MultiModelPerformanceResults,
                                        csv_path: str) -> None:
        """
        Save multi-model evaluation results to CSV with mean and std for each metric.

        The CSV will contain columns:
        model, <metric>_mean, <metric>_std for all metrics evaluated.
        """
        rows = []
        for model_name, mean_metrics in results.model_results.items():
            std_metrics = results.model_results_std.get(model_name)
            row = {"model": model_name}
            for metric in self.metric_names:
                row[f"{metric}_mean"] = float(getattr(mean_metrics, metric))
                row[f"{metric}_std"] = float(getattr(std_metrics, metric))
            rows.append(row)

        df = pd.DataFrame(rows)
        save_dir = os.path.dirname(csv_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved multi-model results to {csv_path}")

    # PLOTTING FUNCTIONS (moved from ModelComparisonVisualizer)

    def plot_single_model_performance(self, results: SingleModelPerformanceResults,
                                      save_path: Optional[str] = None):
        """
        Plot performance metrics for a single model as a bar chart with mean values displayed

        Args:
            results: SingleModelPerformanceResults object
            save_path: Optional path to save the plot
        """
        metric_labels = {
            'mse': 'MSE',
            'rmse': 'RMSE',
            'mae': 'MAE',
            'r2': 'R²',
            'pearson_r': 'Pearson R',
            'f3': 'F3'
        }

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        metrics = ['mse', 'rmse', 'mae', 'r2', 'pearson_r', 'f3']
        metric_display_names = [metric_labels[m] for m in metrics]

        means = [getattr(results.metrics, metric) for metric in metrics]
        stds = [getattr(results.metrics_std, metric) for metric in metrics]

        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))

        bars = ax.bar(metric_display_names, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        ax.set_ylabel('Metric Value')
        ax.set_title(f'{results.model_name} - Performance Metrics\n(n={results.num_splits} splits)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            # Position text above error bar
            text_y = height + std + max(means) * 0.02
            ax.text(bar.get_x() + bar.get_width() / 2., text_y,
                    f'{mean:.3f}±{std:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {save_path}")

        plt.show()

    def plot_masking_degradation_analysis(self, results: List[ModelMaskingDegradationResults],
                                          save_dir: Optional[str] = None):
        """
        Plot individual masking degradation analysis for each model
        Creates one plot per model, showing how performance degrades with increasing masking
        """
        metric_labels = {
            'mse': 'MSE',
            'rmse': 'RMSE',
            'mae': 'MAE',
            'r2': 'R²',
            'pearson_r': 'Pearson R',
            'f3': 'F3'
        }

        for result in results:
            fig, axes = plt.subplots(2, 3, figsize=self.figsize)
            axes = axes.flatten()

            # Get all variant names and colors
            variant_names = list(result.model_variants.keys())
            colors = plt.cm.tab10(np.linspace(0, 1, len(variant_names)))

            for i, metric in enumerate(['mse', 'rmse', 'mae', 'r2', 'pearson_r', 'f3']):
                ax = axes[i]

                for j, variant_name in enumerate(variant_names):
                    color = colors[j]

                    # Get metric values and std for this variant
                    metric_values = result.model_variants[variant_name]['metrics'][metric]
                    metric_std = result.model_variants[variant_name]['std'][metric]

                    # Determine line style
                    if variant_name == 'Pretrained':
                        linestyle = '--'
                        marker = 'o'
                        alpha = 0.8
                        linewidth = 2
                    else:
                        linestyle = '-'
                        marker = 's'
                        alpha = 1.0
                        linewidth = 2

                    # Plot line with error bars
                    ax.errorbar(result.beta_range, metric_values,
                                yerr=metric_std,
                                fmt=f'{marker}{linestyle}',
                                label=variant_name,
                                color=color, alpha=alpha,
                                capsize=3, markersize=6, linewidth=linewidth)

                ax.set_xlabel('Beta (Masking Level)')
                ax.set_ylabel(metric_labels[metric])
                ax.set_title(f'{metric_labels[metric]} vs Masking Level\n(n={result.num_splits} splits)')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

                # Add vertical line at beta=0 to show clean performance
                ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

            fig.suptitle(f'{result.model_name} - Masking Degradation Analysis', fontsize=16)
            plt.tight_layout()

            # Save individual plot
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{result.model_name}_masking_degradation.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved plot: {save_path}")

            plt.show()

    def plot_multi_model_performance(self, results: MultiModelPerformanceResults,
                                     save_path: Optional[str] = None):
        """
        Plot Function 2: Multi-model architecture comparison
        """
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.flatten()

        metric_labels = {
            'mse': 'MSE',
            'rmse': 'RMSE',
            'mae': 'MAE',
            'r2': 'R²',
            'pearson_r': 'Pearson R',
            'f3': 'F3'
        }

        model_names = list(results.model_results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

        for i, metric in enumerate(['mse', 'rmse', 'mae', 'r2', 'pearson_r', 'f3']):
            ax = axes[i]

            means = [getattr(results.model_results[model], metric) for model in model_names]
            stds = [getattr(results.model_results_std[model], metric) for model in model_names]

            bars = ax.bar(model_names, means, yerr=stds, capsize=5,
                          color=colors, alpha=0.7)

            ax.set_ylabel(metric_labels[metric])
            ax.set_title(f'{metric_labels[metric]} Comparison\n(n={results.num_splits} splits)')
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + std + max(means) * 0.02,
                        f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)

        fig.suptitle('Multi-Model Architecture Comparison', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    evaluator = DNNEvaluator()
    # model_configs_multi = [
    #     ("Transformer",
    #      "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/Transformer/pretraining/hp_tuned_parameters"),
    #     ("GCN",
    #      "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/GCN/pretraining/hp_tuned_parameters"),
    #     ("CNN",
    #      "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/CNN/pretraining/hp_tuned_parameters"),
    #     ("GAT",
    #      "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/GAT/pretraining/hp_tuned_parameters"),
    #     ("GIN",
    #      "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/GIN/pretraining/hp_tuned_parameters")
    # ]
    
    # multi_results = evaluator.evaluate_multi_model_performance(
    #     model_configs=model_configs_multi,
    #     data_category="test"
    # )
    
    # evaluator.save_multi_model_results_to_csv(results=multi_results, csv_path="/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/model_comparison/multi_model_comparison.csv")

    # evaluator.plot_multi_model_performance(
    #     results=multi_results,
    #     save_path="/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/model_comparison/multi_model_comparison.png"
    # )

    # model_configs = [
    #     ("Transformer",
    #      "results/hbond_donor/Transformer/pretraining/hp_tuned_parameters",
    #      "results/hbond_donor/Transformer/finetuning/hp_tuned_parameters"),
    #     ("CNN",
    #      "results/hbond_donor/CNN/pretraining/hp_tuned_parameters",
    #      "results/hbond_donor/CNN/finetuning/hp_tuned_parameters"),
    #     ("GCN",
    #      "results/hbond_donor/GCN/pretraining/hp_tuned_parameters",
    #      "results/hbond_donor/GCN/finetuning/hp_tuned_parameters"),
    #     ("GAT",
    #      "results/hbond_donor/GAT/pretraining/hp_tuned_parameters",
    #      "results/hbond_donor/GAT/finetuning/hp_tuned_parameters"),
    #     ("GIN",
    #      "results/hbond_donor/GIN/pretraining/hp_tuned_parameters",
    #      "results/hbond_donor/GIN/finetuning/hp_tuned_parameters"),
    # ]

    # degradation_results = evaluator.evaluate_masking_degradation_analysis(
    #     model_configs=model_configs,
    #     data_category="test",
    #     beta_values=[0.1]
    # )

    # evaluator.plot_masking_degradation_analysis(
    #     results=degradation_results,
    #     save_dir="results/hbond_donor/model_comparison"
    # )

    model_configs = [
        (
            "CNN",
            "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/CNN/pretraining/hp_tuned_parameters",
            "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/CNN/finetuning/hp_tuned_parameters",
        ),
        (
            "GAT",
            "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/GAT/pretraining/hp_tuned_parameters",
            "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/GAT/finetuning/hp_tuned_parameters",
        ),
        (
            "GCN",
            "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/GCN/pretraining/hp_tuned_parameters",
            "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/GCN/finetuning/hp_tuned_parameters",
        ),
        (
            "GIN",
            "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/GIN/pretraining/hp_tuned_parameters",
            "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/GIN/finetuning/hp_tuned_parameters",
        ),
        (
            "Transformer",
            "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/Transformer/pretraining/hp_tuned_parameters",
            "/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/Transformer/finetuning/hp_tuned_parameters",
        ),
    ]

    degradation_results = evaluator.evaluate_masking_degradation_analysis(
        model_configs=model_configs,
        data_category="test",
        beta_values=[0.1]
    )

    evaluator.plot_masking_degradation_analysis(
        results=degradation_results,
        save_dir="/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/LogP/model_training/logp_target/model_comparison"
    )