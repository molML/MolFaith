import itertools
import os
from dataclasses import dataclass, field, fields, asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
import wandb
from pandas import DataFrame


# noinspection PyDataclass
@dataclass
class MetricsStorage:
    mse: list = field(default_factory=list, metadata={"name": 'mse'})
    rmse: list = field(default_factory=list, metadata={"name": 'rmse'})
    mae: list = field(default_factory=list, metadata={"name": 'mae'})
    epoch_index: list = field(default_factory=list, metadata={"name": 'epoch_index'})

    def add_metrics(self, mse: float, rmse: float, mae: float, epoch_index: int) -> None:
        self.mse.append(mse)
        self.rmse.append(rmse)
        self.mae.append(mae)
        self.epoch_index.append(epoch_index)

    def log_last_entry_to_wandb(self, train: bool, epoch_index: Optional[int] = None) -> None:
        pre_fix = 'train_' if train else 'val_'

        for field_data in fields(self):
            # We do not want to log the epoch index
            if field_data.name == 'epoch_index':
                continue
            wandb.log({pre_fix + field_data.name: getattr(self, field_data.name)[-1]}, step=epoch_index)


class Metrics:
    def __init__(self, output_folder_path: str):
        self.output_folder_path = output_folder_path
        self.training_metrics = MetricsStorage()
        self.val_metrics = MetricsStorage()
        self.mse_func = torch.nn.MSELoss()
        self.mae_func = torch.nn.L1Loss()

    def calculate_metrics(self, train: bool, y_pred: torch.Tensor, y_true: torch.Tensor, epoch_index: int) -> None:
        mse_score = self.mse_func(y_pred, y_true)
        mae_score = self.mae_func(y_pred, y_true)
        rmse_score = torch.sqrt(mse_score)

        if train:
            self.training_metrics.add_metrics(mse=mse_score.item(), rmse=rmse_score.item(), mae=mae_score.item(),
                                              epoch_index=epoch_index)
        else:
            self.val_metrics.add_metrics(mse=mse_score.item(), rmse=rmse_score.item(), mae=mae_score.item(),
                                         epoch_index=epoch_index)

    def save_train_val_metrics_to_csv(self) -> None:
        os.makedirs(self.output_folder_path, exist_ok=True)

        train_base_file_name = 'train_metrics'
        path_for_train_metrics = os.path.join(self.output_folder_path, f'{train_base_file_name}.csv')
        df = pd.DataFrame(asdict(self.training_metrics))
        df.to_csv(path_for_train_metrics, index=False)

        val_base_file_name = 'val_metrics'
        path_for_val_metrics = os.path.join(self.output_folder_path, f'{val_base_file_name}.csv')
        df = pd.DataFrame(asdict(self.val_metrics))
        df.to_csv(path_for_val_metrics, index=False)

    def get_val_score_avg(self, epoch_index: int, metric_name: str) -> np.float64:
        df = pd.DataFrame(asdict(self.val_metrics))

        if metric_name not in [field_data.name for field_data in fields(self.val_metrics)]:
            raise ValueError(
                f'The given metric {metric_name} is not available in the list of available validation metrics.')

        metric_avg_per_epoch = df.groupby('epoch_index')[metric_name].mean().reset_index()
        return metric_avg_per_epoch[metric_avg_per_epoch['epoch_index'] == epoch_index].iloc[0][metric_name]

    def log_avg_metric(self, epoch_index: int, train: bool):
        pre_fix = 'train_' if train else 'val_'
        metrics_collection = self.training_metrics if train else self.val_metrics

        for field_data in fields(metrics_collection):
            # We do not want to log the epoch index
            if field_data.name == 'epoch_index':
                continue

            # Calculates the avg value for the current epoch
            target_epoch_selector = [indices == epoch_index for indices in
                                     getattr(metrics_collection, 'epoch_index')]
            target_epoch_metric = list(
                itertools.compress(getattr(metrics_collection, field_data.name), target_epoch_selector))
            metric_avg_per_epoch = np.mean(target_epoch_metric)

            wandb.log({pre_fix + field_data.name: metric_avg_per_epoch.item()}, step=epoch_index)
