import dataclasses
import enum
import os
import yaml
from dataclasses import fields
from optuna import Trial

from mol_faith.model.configs.utils import TuningOptions, HPTypes
from mol_faith.model.types import Config, TrainingConfig, ModelConfig


class SamplingAlgorithms(str, enum.Enum):
    GP = 'GP'
    TPE = 'TPE'

def adjust_configs_hp(trial: Trial, config: Config) -> Config:
    config.training = set_hyperparameters(trial=trial, config=config.training)
    config.model = set_hyperparameters(trial=trial, config=config.model)
    return config

def handle_transformer_dim_feedforward_parameter(config: TrainingConfig | ModelConfig) -> int:
    d_model = getattr(config, "d_model", None)

    if type(d_model) != int or d_model is None:
        raise ValueError(f"dim_feedforward requires d_model to be set but found {d_model}")

    return 4 * d_model

def set_hyperparameters(trial: Trial, config: TrainingConfig | ModelConfig) -> TrainingConfig | ModelConfig:
    for field in fields(config):
        field_val = getattr(config, field.name)

        # If it already a fixed value, we do not get a suggestion
        if type(field_val) != TuningOptions:
            continue

        # Need to handle transformer parameter dim_feedforward as it depends on the parameter d_model!
        if field.name == "dim_feedforward":
            suggested_value = handle_transformer_dim_feedforward_parameter(config)
        else:
            suggested_value = suggest_hyperparameter(trial=trial, parameter=field_val)

        config = dataclasses.replace(config, **{field.name: suggested_value})

    return config


def suggest_hyperparameter(trial: Trial, parameter: TuningOptions) -> int | float | str:
    if parameter.type == HPTypes.INTEGER.value:
        if parameter.step:
            suggested_value = trial.suggest_int(name=parameter.name, low=parameter.low, high=parameter.high,
                                                step=parameter.step)
        else:
            suggested_value = trial.suggest_int(name=parameter.name, low=parameter.low, high=parameter.high,
                                                log=parameter.log)
    elif parameter.type == HPTypes.FLOAT.value:
        if parameter.step:
            suggested_value = trial.suggest_float(name=parameter.name, low=parameter.low, high=parameter.high,
                                                  step=parameter.step)
        else:
            suggested_value = trial.suggest_float(name=parameter.name, low=parameter.low, high=parameter.high,
                                                  log=parameter.log)
    elif parameter.type == HPTypes.CATEGORICAL.value:
        suggested_value = trial.suggest_categorical(name=parameter.name, choices=parameter.choices)
    else:
        raise ValueError(
            f'Could not get hp parameter for {parameter.name}! Parameter type {parameter.type} is not known!')
    return suggested_value


def adjust_wandb_tag_config(config: Config, trial_idx: int):
    wandb_tag_name = f'hp_tuning_trial_{trial_idx}'
    config.training.wandb_tags.append(wandb_tag_name)


def create_trial_output_folder(base_output_dir: str, trial_idx: int) -> str:
    trial_output_dir_path = os.path.join(base_output_dir, 'trials', f'trial_{trial_idx}')
    os.makedirs(trial_output_dir_path, exist_ok=True)
    return trial_output_dir_path


def get_num_data_splits(data_folder_path: str) -> int:
    return len(next(os.walk(data_folder_path))[1])


def create_data_split_output_folder(base_path: str, data_split_index: int) -> str:
    data_split_folder_path = os.path.join(base_path, f'data_split_{data_split_index}')
    os.makedirs(data_split_folder_path, exist_ok=True)
    return data_split_folder_path


def save_config_in_dir(output_path: str, config: Config) -> str:
    file_path = os.path.join(output_path, 'config.yaml')
    with open(file_path, 'w') as outfile:
        yaml.dump(dataclasses.asdict(config), outfile, default_flow_style=False)
    return file_path
