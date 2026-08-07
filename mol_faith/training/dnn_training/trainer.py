import logging

import numpy as np
import os
import random
import torch
import wandb
from abc import abstractmethod
from dataclasses import asdict
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from mol_faith.data.types import DataCategoryType
from mol_faith.model.configs.dnn_configs import get_config
from mol_faith.training.dnn_training.checkpoint_handler import CheckpointHandler
from mol_faith.training.dnn_training.metrics import Metrics

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DNNTrainer:
    def __init__(self, config_file_path: str):
        self.config = get_config(config_file_path=config_file_path)
        self.device = self.config.training.device
        self._set_seeds()
        self._setup_components()

    def _setup_components(self):
        """Override this method in subclasses for custom setup order (e.g. CNN)"""
        self.data_manager = self._get_data_manager()
        self.model = self._get_model()
        self.checkpoint_handler = self._get_checkpoint_handler()
        self.mask_generator = self._get_mask_generator()

    def _set_seeds(self):
        seed = self.config.training.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    @abstractmethod
    def _get_mask_generator(self):
        raise NotImplementedError

    @abstractmethod
    def _get_data_manager(self):
        raise NotImplementedError

    @abstractmethod
    def _get_model(self):
        raise NotImplementedError

    def _get_checkpoint_handler(self):
        return CheckpointHandler(training_config=self.config.training)

    def load_pretrained_model(self, pretrained_checkpoint_path: str):
        """Load pretrained model for fine-tuning"""
        assert self.model is not None, 'Model needs to be loaded before loading weights'

        if os.path.exists(pretrained_checkpoint_path):
            self.model.load_params(path=pretrained_checkpoint_path)
            logger.info(f"Loaded pretrained model from: {pretrained_checkpoint_path}")
        else:
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_checkpoint_path}")

    def set_wandb_meta_data(self):
        # If not set, specify a random run name
        if len(self.config.training.wandb_run_name) == 0:
            self.config.training.wandb_run_name = f'{wandb.util.generate_id()}'

        # If we specified a tag in the config (len > 0 as we always set it to an empty list as a default)
        # use specified tags or else set None so that wandb will ignore it
        if len(self.config.training.wandb_tags) == 0:
            self.config.training.wandb_tags = None

        if len(self.config.training.wandb_group_prefix) > 0:
            self.config.training.wandb_group_name = (f'{self.config.training.wandb_group_prefix} - '
                                                     f'{self.config.training.wandb_group_name}')

    def train(self):
        self.set_wandb_meta_data()

        wandb.login()

        training_config = self.config.training
        model_config = self.config.model

        wandb_run = wandb.init(
            project=training_config.wandb_project_name,
            dir=training_config.output_dir_path,
            tags=training_config.wandb_tags,
            name=training_config.wandb_run_name,
            group=training_config.wandb_group_name,
            config={
                'training': asdict(training_config),
                'model': asdict(model_config),
            },
        )

        metrics_collection = Metrics(output_folder_path=training_config.output_dir_path)

        self.model.to(self.device)

        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=training_config.lr,
                                     weight_decay=training_config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                         T_0=training_config.initial_warm_up_epochs,
                                                                         T_mult=training_config.t_mult)

        train_data_loader = self.data_manager.load_data(target_column_name=self.config.training.target_column_name,
                                                        data_category_type=DataCategoryType.TRAIN.value)
        val_data_loader = self.data_manager.load_data(target_column_name=self.config.training.target_column_name,
                                                      data_category_type=DataCategoryType.VAL.value)

        for epoch_index in tqdm(range(1, training_config.num_epochs + 1)):
            self.train_epoch(optimizer=optimizer, loss_func=loss_func, metrics_collection=metrics_collection,
                             epoch_index=epoch_index, train_data_loader=train_data_loader,
                             scheduler=scheduler)

            if epoch_index % training_config.val_frequency == 0:
                self.validate_model(metrics_collection=metrics_collection, epoch_index=epoch_index,
                                    val_data_loader=val_data_loader)

                should_stop = self.checkpoint_handler(model=self.model, epoch=epoch_index,
                                                      metric_val=metrics_collection.get_val_score_avg(
                                                          epoch_index=epoch_index,
                                                          metric_name='mse'))
                if should_stop:
                    break

            wandb.log({'lr': scheduler.get_last_lr()[0]})
            # wandb.log({'lr': optimizer.param_groups[0]['lr']})

        metrics_collection.save_train_val_metrics_to_csv()

        wandb.finish()

    def _process_batch(self, batch):
        raise NotImplementedError("Subclasses must implement _process_batch")

    def train_epoch(self, optimizer: Optimizer, loss_func: nn.Module, metrics_collection: Metrics, epoch_index: int,
                    scheduler: LRScheduler, train_data_loader: DataLoader):
        self.model.train()
        for batch in train_data_loader:
            optimizer.zero_grad()
            inputs, targets = self._process_batch(batch)
            if self.config.training.finetune_beta:
                masks = self.mask_generator.create_finetuning_masks(batch_data=inputs)
                output = self.model(inputs, masks)
            else:
                output = self.model(inputs)
            loss = loss_func(output, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            metrics_collection.calculate_metrics(train=True, y_pred=output, y_true=targets, epoch_index=epoch_index)
        metrics_collection.log_avg_metric(epoch_index=epoch_index, train=True)

    def validate_model(self, metrics_collection: Metrics, epoch_index: int, val_data_loader: DataLoader):
        self.model.eval()
        for batch in val_data_loader:
            inputs, targets = self._process_batch(batch)
            with torch.set_grad_enabled(False):
                if self.config.training.finetune_beta:
                    masks = self.mask_generator.create_finetuning_masks(batch_data=inputs)
                    output = self.model(inputs, masks)
                else:
                    output = self.model(inputs)
                metrics_collection.calculate_metrics(train=False, y_pred=output, y_true=targets,
                                                     epoch_index=epoch_index)
        metrics_collection.log_avg_metric(epoch_index=epoch_index, train=False)
