import copy
import logging
import numpy as np
import optuna
import os
import yaml
from abc import abstractmethod
from optuna import Study, Trial
from optuna.pruners import MedianPruner
from optuna.samplers import GPSampler, TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from typing import Optional

from mol_faith.model.types import Config
from mol_faith.training.hyperparameter_tuning.utils import adjust_configs_hp, adjust_wandb_tag_config, \
    create_trial_output_folder, get_num_data_splits, create_data_split_output_folder, \
    save_config_in_dir, SamplingAlgorithms
from mol_faith.training.types import Trainer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Objective:
    def __init__(self, base_config: Config, enable_pruning: bool = True):
        self.base_config = base_config
        self.enable_pruning = enable_pruning

    def __call__(self, trial: Trial):
        logger.info(f'Trial {trial.number}')

        # Copy the config so that we can adjust it for each trial
        base_config = copy.deepcopy(self.base_config)

        base_config = adjust_configs_hp(trial=trial, config=base_config)

        adjust_wandb_tag_config(config=base_config, trial_idx=trial.number)

        trial_output_dir_path = create_trial_output_folder(base_output_dir=base_config.training.output_dir_path,
                                                           trial_idx=trial.number)

        num_data_splits = get_num_data_splits(data_folder_path=base_config.training.dataset_folder_path)

        val_scores = []
        for data_split_index in range(num_data_splits):
            config = copy.deepcopy(base_config)

            config.training.dataset_folder_path = os.path.join(self.base_config.training.dataset_folder_path,
                                                               f'data_split_{data_split_index}')

            if len(config.training.wandb_tags) >= 0:
                config.training.wandb_tags.append(f'data_split_{data_split_index}')
            else:
                config.training.wandb_tags = [f'data_split_{data_split_index}']

            data_split_folder_path = create_data_split_output_folder(base_path=trial_output_dir_path,
                                                                     data_split_index=data_split_index)
            config.training.output_dir_path = data_split_folder_path

            trial_config_file_path = save_config_in_dir(output_path=data_split_folder_path, config=config)

            logger.info(f'Start training model on data split {data_split_index}')

            trainer = self.get_trainer(trial_config_file_path=trial_config_file_path)
            trainer.train()

            logger.info(f'Start validate model on data split {data_split_index}')
            val_score = self.validate_model(trainer=trainer)
            logger.info(f'Validation score for split index {data_split_index}: {val_score}')
            val_scores.append(val_score)

            # Report intermediate value after each data split
            if self.enable_pruning:
                intermediate_score = np.mean(val_scores[:data_split_index + 1])
                trial.report(intermediate_score, step=data_split_index)

                # Check if trial should be pruned
                if trial.should_prune():
                    logger.info(f'Trial {trial.number} pruned at data split {data_split_index}')
                    raise optuna.TrialPruned()

        avg_val_score = np.mean(val_scores)
        logger.info(f'Average validation score for trial {trial.number}: {avg_val_score}')
        return avg_val_score

    @abstractmethod
    def get_trainer(self, trial_config_file_path: str):
        raise NotImplementedError

    @abstractmethod
    def validate_model(self, trainer: Trainer):
        raise NotImplementedError


class HPTuner:
    def __init__(self, config_path: str, num_trials: int, sampling_algo: str,
                 seed: int = 1234, enable_pruning: bool = True,
                 pruning_params: Optional[dict] = None):
        self.num_trials: int = num_trials
        self.seed: int = seed
        self.sampling_algo: str = sampling_algo
        self.enable_pruning: bool = enable_pruning
        self.pruning_params: dict = pruning_params or self._get_default_pruning_params()
        self.base_config: Config = self.get_config(config_path)
        self.study: Study = self.create_study()
        self.objective: Objective = self.get_objective()

    @staticmethod
    def _get_default_pruning_params() -> dict:
        return {
            'n_startup_trials': 20,  # Wait for 20 trials before pruning (10 startup trials in the TPE sampler)
            'n_warmup_steps': 2,  # Wait for 2 data splits before pruning
            'interval_steps': 1  # Check for pruning at every step (every report step)
        }

    @abstractmethod
    def get_config(self, config_path) -> Config:
        raise NotImplementedError

    @abstractmethod
    def get_objective(self):
        raise NotImplementedError

    def create_optuna_storage(self) -> JournalStorage:
        path_to_log_file = os.path.join(self.base_config.training.output_dir_path, 'hp_results.log')
        storage = JournalStorage(JournalFileBackend(path_to_log_file))
        return storage

    def get_sampling_algorithm(self):
        if self.sampling_algo == SamplingAlgorithms.TPE.value:
            return TPESampler(seed=self.seed)
        elif self.sampling_algo == SamplingAlgorithms.GP.value:
            return GPSampler(seed=self.seed)
        else:
            raise ValueError(f'Sampling algorithm {self.sampling_algo} is not supported!')

    def get_pruning_algorithm(self):
        if not self.enable_pruning:
            return None

        return MedianPruner(
            n_startup_trials=self.pruning_params.get('n_startup_trials'),
            n_warmup_steps=self.pruning_params.get('n_warmup_steps'),
            interval_steps=self.pruning_params.get('interval_steps')
        )

    @staticmethod
    def check_if_study_exists(storage: JournalStorage) -> bool:
        studies = optuna.get_all_study_summaries(storage)

        if studies:
            return True
        else:
            return False

    def create_study(self) -> Study:
        storage = self.create_optuna_storage()
        sampler = self.get_sampling_algorithm()
        pruner = self.get_pruning_algorithm()

        if self.check_if_study_exists(storage=storage):
            logger.info('Found a study and will continue the optimization.')
            study = optuna.load_study(study_name=None, storage=storage)
        else:
            logger.info(f'Created a new study with pruning: {self.enable_pruning}')
            study = optuna.create_study(
                sampler=sampler,
                pruner=pruner,
                direction='minimize',
                storage=storage
            )

        return study

    def save_best_params(self):
        best_params = self.study.best_params

        with open(os.path.join(self.base_config.training.output_dir_path,
                               f'best_params_trial_{self.study.best_trial.number}.yaml'), 'w') as f:
            yaml.dump(best_params, f)

    def save_pruning_statistics(self):
        """Save statistics about pruned trials."""
        pruned_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        stats = {
            'total_trials': len(self.study.trials),
            'completed_trials': len(completed_trials),
            'pruned_trials': len(pruned_trials),
            'pruning_rate': len(pruned_trials) / len(self.study.trials) if self.study.trials else 0,
            'pruning_algorithm': 'median',
            'pruning_enabled': self.enable_pruning,
            'pruning_params': self.pruning_params
        }

        with open(os.path.join(self.base_config.training.output_dir_path, 'pruning_stats.yaml'), 'w') as f:
            yaml.dump(stats, f)

        logger.info(f"Pruning statistics: {stats}")

    def tune_hyperparameters(self) -> None:
        logger.info('Start hyperparameter tuning!')

        if self.enable_pruning:
            logger.info('Pruning enabled with MedianPruner')
            logger.info(f'Pruning parameters: {self.pruning_params}')

        if not os.path.isdir(self.base_config.training.dataset_folder_path):
            raise NotADirectoryError(f'Path {self.base_config.training.dataset_folder_path} is not a directory!')

        self.study.optimize(self.objective, n_trials=self.num_trials)
        self.save_best_params()

        if self.enable_pruning:
            self.save_pruning_statistics()
