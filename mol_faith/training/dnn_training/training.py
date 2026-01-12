import argparse
import copy
import logging
import os
from pathlib import Path

from mol_faith.model.configs.dnn_configs import get_config, DNNConfig
from mol_faith.model.configs.utils import save_config
from mol_faith.model.types import ModelTypes, SUPPORTED_DNN_MODEL_TYPES
from mol_faith.training.dnn_training.gnn_trainer import GNNTrainer
from mol_faith.training.dnn_training.sequence_model_trainer import SequenceModelTrainer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_trainer(config_file_path: str, model_type: str):
    """Create trainer instance based on model type and config path."""
    if model_type == ModelTypes.GNN.value:
        return GNNTrainer(config_file_path=config_file_path)
    elif model_type == ModelTypes.CNN.value or model_type == ModelTypes.TRANSFORMER.value:
        return SequenceModelTrainer(config_file_path=config_file_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def validate_data_directory(data_dir: str, required_files: list = None) -> bool:
    """
    Validate that a data directory contains required files.

    Args:
        data_dir: Path to data directory
        required_files: List of required files (default: ['train.csv', 'val.csv'])

    Returns:
        True if all required files exist, False otherwise
    """
    if required_files is None:
        required_files = ['train.csv', 'val.csv']

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return False

    missing_files = []
    for file_name in required_files:
        file_path = data_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        logger.error(f"Missing required files in {data_dir}: {missing_files}")
        return False

    logger.info(f"Data directory validated: {data_dir}")
    return True


def find_data_splits(base_data_dir: str) -> list:
    """
    Find all data_split_x directories in the base data directory.

    Args:
        base_data_dir: Base directory containing data_split_x subdirectories

    Returns:
        List of tuples (split_name, split_path)
    """
    splits = []
    base_path = Path(base_data_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Base data directory not found: {base_data_dir}")

    # Look for data_split_x directories
    for split_dir in sorted(base_path.glob("data_split_*")):
        if split_dir.is_dir():
            split_name = split_dir.name
            split_path = str(split_dir)

            # Validate that this split has required data files
            if validate_data_directory(split_path):
                splits.append((split_name, split_path))
                logger.info(f"Found valid data split: {split_name} at {split_path}")
            else:
                raise FileNotFoundError(f"Invalid data split {split_name}: missing required files")

    if not splits:
        raise FileNotFoundError(f"No valid data splits found in {base_data_dir}")

    logger.info(f"Found {len(splits)} valid data splits")
    return splits


def train_single_model(config_file_path: str, model_type: str):
    """
    Train a single model with the given configuration.

    Args:
        config_file_path: Path to the configuration file
        model_type: Type of model ('GNN' or 'CNN')
    """
    logger.info(f"Starting single model training")
    logger.info(f"Config file: {config_file_path}")
    logger.info(f"Model type: {model_type}")

    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")

    # Validate data directory
    config = get_config(config_file_path=config_file_path)
    data_dir = config.training.dataset_folder_path

    if not data_dir:
        raise ValueError("dataset_folder_path not found in configuration")

    if not validate_data_directory(data_dir):
        raise ValueError(f"Invalid data directory: {data_dir}")

    # Create and run trainer
    trainer = create_trainer(config_file_path, model_type)
    trainer.train()

    logger.info("Training completed")


def set_wandb_metadata(config: DNNConfig, split_name: str) -> None:
    config.training.wandb_run_name = f'{split_name}'
    config.training.wandb_tags.append(f'{split_name}')


def train_multiple_splits(base_config_path: str, model_type: str):
    """
    Train multiple models on different data splits.

    Args:
        base_config_path: Path to the base configuration file
        model_type: Type of model ('GNN' or 'CNN')
    """
    logger.info(f"Starting training for multiple splits")
    logger.info(f"Base config: {base_config_path}")
    logger.info(f"Model type: {model_type}")

    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")

    # Load base configuration
    base_config = get_config(base_config_path)
    base_dataset_path = base_config.training.dataset_folder_path
    base_output_dir = base_config.training.output_dir_path

    if not base_dataset_path:
        raise ValueError("dataset_folder_path not found in base configuration")
    if not base_output_dir:
        raise ValueError("output_dir not found in base configuration")

    logger.info(f"Base dataset path: {base_dataset_path}")
    logger.info(f"Base output dir: {base_output_dir}")

    # Find all data splits
    data_splits = find_data_splits(base_dataset_path)

    successful_splits = []
    failed_splits = []

    # Train each split
    for split_name, split_data_path in data_splits:
        try:
            logger.info(f"Processing {split_name}")

            # Create output directory for this split
            split_output_dir = os.path.join(base_output_dir, split_name)
            os.makedirs(split_output_dir, exist_ok=True)
            logger.info(f"Created output directory: {split_output_dir}")

            # Create split-specific config
            split_config = copy.deepcopy(base_config)
            split_config.training.dataset_folder_path = split_data_path
            split_config.training.output_dir_path = split_output_dir
            set_wandb_metadata(config=split_config, split_name=split_name)

            # Save split config
            split_config_path = os.path.join(split_output_dir, f"config.yaml")
            save_config(split_config, split_config_path)
            logger.info(f"Saved split config: {split_config_path}")

            # Train this split
            train_single_model(config_file_path=split_config_path,
                               model_type=model_type)

            successful_splits.append(split_name)
            logger.info(f"Successfully completed training for {split_name}")

        except Exception as e:
            logger.error(f"Failed to train {split_name}: {e}")
            failed_splits.append((split_name, str(e)))

    # Summary
    logger.info(f"Training summary:")
    logger.info(f"Successfully processed: {len(successful_splits)} splits")
    logger.info(f"Failed: {len(failed_splits)} splits")

    if successful_splits:
        logger.info(f"Successful splits: {', '.join(successful_splits)}")

    if failed_splits:
        logger.warning(f"Failed splits:")
        for split_name, error in failed_splits:
            logger.warning(f"  {split_name}: {error}")

    if failed_splits:
        raise RuntimeError(f"Some splits failed to train. Check logs for details.")


def main():
    parser = argparse.ArgumentParser(
        description='Train neural network(s) (GNN or CNN) with given configuration(s)'
    )

    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to configuration file (base config for multiple splits, or single config for single split)'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        choices=SUPPORTED_DNN_MODEL_TYPES,
        required=True,
        help='Type of model to train.'
    )

    parser.add_argument(
        '--multiple_splits',
        action='store_true',
        help='Train multiple models from different data splits (expects data_split_x subdirectories in dataset_folder_path)'
    )

    args = parser.parse_args()

    # Validate config path
    if not os.path.exists(args.config_path):
        parser.error(f"Configuration file not found: {args.config_path}")

    try:
        # Run training
        if args.multiple_splits:
            train_multiple_splits(
                base_config_path=args.config_path,
                model_type=args.model_type
            )
        else:
            train_single_model(
                config_file_path=args.config_path,
                model_type=args.model_type
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
