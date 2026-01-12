import logging
import math
import numpy as np
import os
import pandas as pd
import seaborn as sns
import yaml
from abc import abstractmethod
from dataclasses import dataclass, field, fields, asdict
from glob import glob
from matplotlib import pyplot as plt
from pandas import DataFrame
from typing import List

from mol_faith.data.types import DataCategoryType

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class EvaluationMetrics:
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    f3: float = 0.0
    pearson_r: float = 0.0


@dataclass
class MetricsStorage:
    output_path: str
    data_category: str
    mse: List[float] = field(default_factory=list, metadata={'name': 'mse'})
    rmse: List[float] = field(default_factory=list, metadata={'name': 'rmse'})
    mae: List[float] = field(default_factory=list, metadata={'name': 'mae'})
    r2: List[float] = field(default_factory=list, metadata={'name': 'r2'})
    f3: List[float] = field(default_factory=list, metadata={'name': 'f3'})
    pearson_r: List[float] = field(default_factory=list, metadata={'name': 'pearson_r'})

    def add_metric_values(self, metrics: EvaluationMetrics):
        for field_info in fields(metrics):
            field_name = field_info.name
            if hasattr(self, field_name):
                getattr(self, field_name).append(getattr(metrics, field_name))
            else:
                raise AttributeError(f'Field "{field_name}" does not exist in MetricsStorage')

    def compute_statistics(self) -> DataFrame:
        stats = []

        for field_info in fields(self):
            field_name = field_info.name
            if field_name in ['output_path', 'data_category']:
                continue

            values = getattr(self, field_name)
            if values:
                stats.append({
                    'Metric': field_name,
                    'Mean': float(np.mean(values)),
                    'Std Dev': float(np.std(values, ddof=1)),
                    'Min': float(np.min(values)),
                    'Max': float(np.max(values))
                })

        return pd.DataFrame(stats)

    def save_raw_values(self):
        os.makedirs(self.output_path, exist_ok=True)

        full_dict = asdict(self)

        metrics_dict = {key: full_dict[key] for key in full_dict if key not in ['output_path', 'data_category']}
        metrics_dict = {key: [convert_numpy(val) for val in values] for key, values in metrics_dict.items()}

        path_to_file = os.path.join(self.output_path, f'raw_{self.data_category}_evaluation_metrics.yaml')
        with open(path_to_file, 'w') as yaml_file:
            yaml.dump(metrics_dict, yaml_file, default_flow_style=False)

    def save_metric_statistics(self):
        statistics = self.compute_statistics()

        os.makedirs(self.output_path, exist_ok=True)

        path_to_file = os.path.join(self.output_path, f'{self.data_category}_evaluation_metrics_stats.yaml')

        statistics = statistics.transpose().rename(columns=statistics.iloc[:, 0]).drop('Metric')
        with open(path_to_file, 'w') as yaml_file:
            yaml.dump(statistics.to_dict(), yaml_file, default_flow_style=False)

    def plot_metrics_mean_and_std(self):
        stats_df = self.compute_statistics()

        fig, axes = plt.subplots(1, 1, figsize=(10, 5))

        sns.barplot(data=stats_df, x='Metric', y='Mean', ax=axes, capsize=0.2)
        axes.errorbar(stats_df['Metric'], stats_df['Mean'], yerr=stats_df['Std Dev'], fmt='none', capsize=5,
                      color='black')
        axes.set_title('Mean & Standard Deviation of Metrics')
        axes.set_xlabel('Metric')
        axes.set_ylabel('Mean Value')
        axes.set_ylim(0, 1)
        plt.tight_layout()

        file_path = os.path.join(self.output_path, f'{self.data_category}_metric_stats')
        plt.savefig(file_path + '.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(file_path + '.png', format='png', bbox_inches='tight')


def convert_numpy(value):
    if isinstance(value, np.ndarray):
        value = value.squeeze()
        return value.tolist()
    elif isinstance(value, np.generic):
        return value.item()
    return value


def calc_f3_metric(train_true: np.array, val_true: np.array, val_pred: np.array):
    n_tr = len(train_true)
    n_out = len(val_true)

    train_avg_true = np.mean(train_true, axis=0)

    squared_residual_tr = np.square(train_true - train_avg_true)
    squared_residual_out = np.square(val_true - val_pred)

    sum_squared_residual_tr = np.sum(squared_residual_tr)
    sum_squared_residual_out = np.sum(squared_residual_out)

    f3 = 1 - (sum_squared_residual_out / n_out) / (sum_squared_residual_tr / n_tr)

    return f3


class ModelEvaluator:
    def __init__(self, model_type: str):
        self.model_type = model_type

    @abstractmethod
    def evaluate_model(self, base_result_path: str, data_category: str) -> EvaluationMetrics:
        raise NotImplementedError

    def evaluate_hp_tuning(self, hp_tuning_result_folder_path: str, output_path: str, data_type: str) -> None:
        logger.info(f'Evaluate model {self.model_type} on {data_type} of every data split!')

        os.makedirs(output_path, exist_ok=True)

        metrics_storage = MetricsStorage(output_path=output_path, data_category=data_type)

        data_split_result_paths = sorted(glob(os.path.join(hp_tuning_result_folder_path, 'data_split_*')))
        for result_path in data_split_result_paths:
            metrics = self.evaluate_model(base_result_path=result_path, data_category=data_type)
            metrics_storage.add_metric_values(metrics=metrics)

        metrics_storage.save_raw_values()
        metrics_storage.save_metric_statistics()
        metrics_storage.plot_metrics_mean_and_std()

    def evaluate_output_perturbation_training(self, perturbation_result_folder_path: str) -> None:
        logger.info(f'Evaluate model {self.model_type} on {DataCategoryType.TEST.value} data for every alpha value '
                    f'and data split.')

        alpha_values_result_paths = sorted(glob(os.path.join(perturbation_result_folder_path, 'alpha_value_*')))
        for alpha_val_result_path in alpha_values_result_paths:

            output_path = os.path.join(alpha_val_result_path, 'evaluation')
            metrics_storage = MetricsStorage(output_path=output_path, data_category=DataCategoryType.TEST.value)
            os.makedirs(output_path, exist_ok=True)

            data_splits_result_paths = sorted(glob(os.path.join(alpha_val_result_path, 'data_split_*')))
            for data_split_result_path in data_splits_result_paths:
                metrics = self.evaluate_model(base_result_path=data_split_result_path,
                                              data_category=DataCategoryType.TEST.value)
                metrics_storage.add_metric_values(metrics=metrics)

            metrics_storage.save_raw_values()
            metrics_storage.save_metric_statistics()
            metrics_storage.plot_metrics_mean_and_std()

        self.evaluate_model_performance_across_alpha_values(
            perturbation_result_folder_path=perturbation_result_folder_path)

    @staticmethod
    def evaluate_model_performance_across_alpha_values(perturbation_result_folder_path: str) -> None:
        alpha_values_result_paths = sorted(glob(os.path.join(perturbation_result_folder_path, 'alpha_value_*')))

        metrics_by_alpha = {}
        for alpha_val_result_path in alpha_values_result_paths:
            raw_eval_metrics_path = os.path.join(alpha_val_result_path, 'evaluation',
                                                 'raw_test_evaluation_metrics.yaml')

            if not os.path.isfile(raw_eval_metrics_path):
                raise FileNotFoundError(f'The file for the raw evaluation metrics could not be found under '
                                        f'{raw_eval_metrics_path}. Please run evaluate_output_perturbation_training() '
                                        f'first.')

            with open(raw_eval_metrics_path, 'r') as f:
                metrics = yaml.safe_load(f)

            alpha = alpha_val_result_path.split('_')[-1]
            metrics_by_alpha[alpha] = metrics

        sample_alpha = next(iter(metrics_by_alpha))
        metric_names = list(metrics_by_alpha[sample_alpha].keys())
        n_metrics = len(metric_names)

        n_cols = 4
        n_rows = math.ceil(n_metrics / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        sorted_alphas = sorted(metrics_by_alpha.keys())

        for i, metric in enumerate(metric_names):
            means = []
            stds = []
            all_values = []
            for alpha in sorted_alphas:
                values = metrics_by_alpha[alpha].get(metric, [])
                assert len(values) > 0
                values = np.array(values)
                means.append(np.mean(values))
                stds.append(np.std(values))
                all_values.extend(values)
            ax = axes[i]
            ax.errorbar(sorted_alphas, means, yerr=stds, fmt='-o', capsize=5, color='blue', ecolor='red', elinewidth=2)
            ax.set_xlabel("Alpha")
            ax.set_ylabel(metric)
            ax.set_xticks(sorted_alphas)
            ax.set_xticklabels([str(alpha) for alpha in sorted_alphas], rotation=45)
            ax.grid(True)
            upper_bound = max(max(all_values), max(m + s for m, s in zip(means, stds)))

            if metric in ['f3', 'pearson_r', 'r2']:
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, upper_bound + 0.1)

        # Turn off any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

        file_path = os.path.join(perturbation_result_folder_path, 'evaluation')
        os.makedirs(file_path, exist_ok=True)
        plt.savefig(os.path.join(file_path, 'model_performance_across_alpha_values.pdf'), format='pdf',
                    bbox_inches='tight')
        plt.savefig(os.path.join(file_path, 'model_performance_across_alpha_values.png'), format='png',
                    bbox_inches='tight')
