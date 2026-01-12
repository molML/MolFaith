import argparse
import logging

from mol_faith.model.types import SUPPORTED_MODEL_TYPES, SUPPORTED_DNN_MODEL_TYPES
from mol_faith.training.hyperparameter_tuning.dnn_hp_tuner import DNNHPTuner
from mol_faith.training.hyperparameter_tuning.utils import SamplingAlgorithms

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def tune_hyperparams(config_path: str, model_type: str, num_trials: int, sampling_algo: str):
    if model_type not in SUPPORTED_DNN_MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {SUPPORTED_DNN_MODEL_TYPES}")
    
    hp_tuner = DNNHPTuner(config_path=config_path, num_trials=num_trials, sampling_algo=sampling_algo)
    hp_tuner.tune_hyperparameters()


def main():
    parser = argparse.ArgumentParser(description='Tune the hyperparameters of a given model')
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to the configuration file.'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        choices=SUPPORTED_MODEL_TYPES,
        required=True,
        help=f'Specify which model to use ({[model_type for model_type in SUPPORTED_MODEL_TYPES]}).'
    )
    parser.add_argument(
        '--num_trials',
        type=int,
        required=True,
        help=f'Specify the number of trials for the Sampler.'
    )
    parser.add_argument(
        '--sampling_algo',
        type=str,
        required=True,
        choices=[sampling_algo.value for sampling_algo in SamplingAlgorithms],
        help=f'Specify the sampling algorithm: {[sampling_algo.value for sampling_algo in SamplingAlgorithms]}'
    )

    args = parser.parse_args()

    tune_hyperparams(config_path=args.config_path, model_type=args.model_type, num_trials=args.num_trials,
                     sampling_algo=args.sampling_algo)


if __name__ == '__main__':
    main()
