from typing import Optional

from mol_faith.model.configs.dnn_configs import get_config, DNNModelNames
from mol_faith.model.types import Config
from mol_faith.training.dnn_training.gnn_trainer import GNNTrainer
from mol_faith.training.dnn_training.sequence_model_trainer import SequenceModelTrainer
from mol_faith.training.dnn_training.trainer import DNNTrainer
from mol_faith.training.hyperparameter_tuning.hp_tuner import HPTuner, Objective


class DNNObjective(Objective):
    def __init__(self, base_config: Config, enable_pruning: bool = True):
        super().__init__(base_config, enable_pruning)

    def get_trainer(self, trial_config_file_path: str):
        model_name = self.base_config.training.model_name

        if model_name in [DNNModelNames.GAT.value, DNNModelNames.GCN.value, DNNModelNames.GIN.value]:
            trainer = GNNTrainer(config_file_path=trial_config_file_path)
        elif model_name in [DNNModelNames.CNN.value, DNNModelNames.Transformer.value]:
            trainer = SequenceModelTrainer(config_file_path=trial_config_file_path)
        else:
            raise ValueError(f'Model name {model_name} not supported!')

        return trainer

    def validate_model(self, trainer: DNNTrainer):
        return trainer.checkpoint_handler.best_metric_val


class DNNHPTuner(HPTuner):
    def __init__(self, config_path: str, num_trials: int, sampling_algo: str, seed: int = 1234,
                 enable_pruning: bool = True, pruning_params: Optional[dict] = None):
        super().__init__(config_path=config_path,
                         num_trials=num_trials,
                         seed=seed,
                         sampling_algo=sampling_algo,
                         enable_pruning=enable_pruning,
                         pruning_params=pruning_params
                         )

    def get_config(self, config_path) -> Config:
        return get_config(config_file_path=config_path)

    def get_objective(self):
        return DNNObjective(base_config=self.base_config, enable_pruning=self.enable_pruning)
