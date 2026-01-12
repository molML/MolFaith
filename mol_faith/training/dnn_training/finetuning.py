import argparse
import copy
import logging
import os
from pathlib import Path

from mol_faith.model.configs.dnn_configs import get_config
from mol_faith.model.configs.utils import save_config
from mol_faith.model.types import SUPPORTED_DNN_MODEL_TYPES, ModelTypes
from mol_faith.training.dnn_training.checkpoint_handler import CheckpointHandler
from mol_faith.training.dnn_training.gnn_trainer import GNNTrainer
from mol_faith.training.dnn_training.sequence_model_trainer import SequenceModelTrainer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_finetuning_config(original_config, new_output_dir: str, beta: float):
    """Create a new configuration for fine-tuning with F-Fidelity parameters."""

    # Deep copy the original config to avoid modifying it
    finetune_config = copy.deepcopy(original_config)

    # 1. Add F-Fidelity fine-tuning beta parameter
    finetune_config.training.finetune_beta = beta

    # 2. Adjust output directory for fine-tuning results
    finetune_config.training.output_dir_path = new_output_dir

    # 3. Update wandb settings for fine-tuning
    if hasattr(finetune_config.training, 'wandb_group_prefix') and finetune_config.training.wandb_group_prefix:
        finetune_config.training.wandb_group_prefix += '_finetuning'
    else:
        finetune_config.training.wandb_group_prefix = 'finetuning'
    finetune_config.training.wandb_tags.append(f'beta_{beta}')
    return finetune_config


def validate_model_checkpoint(original_config):
    """Validate that the model checkpoint exists and can be loaded."""
    try:
        checkpoint_handler = CheckpointHandler(training_config=original_config.training)
        best_model_path = checkpoint_handler.get_best_model_path()

        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {best_model_path}")

        logger.info(f"Found model checkpoint: {best_model_path}")
        return best_model_path

    except Exception as e:
        raise RuntimeError(f"Failed to locate model checkpoint: {e}")


def create_trainer(finetune_config_path: str, model_type: str, pretrained_checkpoint_path: str):
    """Create trainer instance based on model type and config path."""
    if model_type == ModelTypes.GNN.value:
        trainer = GNNTrainer(config_file_path=finetune_config_path)
    elif model_type == ModelTypes.CNN.value or model_type == ModelTypes.TRANSFORMER.value:
        trainer = SequenceModelTrainer(config_file_path=finetune_config_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load pretrained model
    trainer.load_pretrained_model(pretrained_checkpoint_path)

    return trainer


def finetune_single_model(original_config_path: str, new_output_dir: str, beta: float, model_type: str):
    """
    Fine-tune a single pre-trained model using F-Fidelity methodology.

    Args:
        original_config_path: Path to the original model's configuration file
        new_output_dir: Directory where fine-tuned model and config will be saved
        beta: F-Fidelity beta parameter (fraction of input that can be masked)
        model_type: Type of model ('GNN' or 'CNN')
    """

    logger.info(f"Starting F-Fidelity fine-tuning process")
    logger.info(f"Original config: {original_config_path}")
    logger.info(f"Output directory: {new_output_dir}")
    logger.info(f"Beta parameter: {beta}")
    logger.info(f"Model type: {model_type}")

    # 1. Load original configuration
    logger.info("Loading original configuration")
    original_config = get_config(config_file_path=original_config_path)

    # 2. Validate that the pre-trained model exists and get its path
    logger.info("Validating pre-trained model checkpoint")
    pretrained_checkpoint_path = validate_model_checkpoint(original_config)

    # 3. Create fine-tuning configuration (with new output directory)
    logger.info(f"Creating fine-tuning configuration with beta={beta}")
    finetune_config = create_finetuning_config(original_config, new_output_dir, beta)

    # 4. Create output directory and save new configuration
    logger.info("Saving fine-tuning configuration")
    os.makedirs(new_output_dir, exist_ok=True)

    finetune_config_path = os.path.join(new_output_dir, "config.yaml")
    save_config(finetune_config, finetune_config_path)
    logger.info(f"Saved configuration to: {finetune_config_path}")

    # 5. Create trainer and load pretrained model
    logger.info(f"Creating trainer and loading pretrained checkpoint: {pretrained_checkpoint_path}")
    trainer = create_trainer(finetune_config_path=finetune_config_path, model_type=model_type,
                             pretrained_checkpoint_path=pretrained_checkpoint_path)

    # 6. Start fine-tuning
    logger.info(f"Starting F-Fidelity fine-tuning with random masking (beta={beta})")
    trainer.train()

    logger.info("F-Fidelity fine-tuning completed")
    logger.info(f"Fine-tuned model saved in: {new_output_dir}")
    logger.info(f"Configuration saved as: {finetune_config_path}")


def find_split_configs(base_output_path: str):
    """
    Find configuration files for all data splits.

    Args:
        base_output_path: Base directory containing data_split_x subdirectories

    Returns:
        List of tuples (split_name, config_path)
    """
    split_configs = []
    base_path = Path(base_output_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Base output path not found: {base_output_path}")

    # Look for data_split_x directories
    for split_dir in sorted(base_path.glob("data_split_*")):
        if split_dir.is_dir():
            # Look for config files in the split directory
            config_files = list(split_dir.glob("*.yaml")) + list(split_dir.glob("*.yml"))

            if len(config_files) == 1:
                split_name = split_dir.name
                config_path = str(config_files[0])
                split_configs.append((split_name, config_path))
                logger.info(f"Found config for {split_name}: {config_path}")
            elif len(config_files) == 0:
                logger.warning(f"No config file found in {split_dir}")
            else:
                logger.warning(f"Multiple config files found in {split_dir}, skipping")

    if not split_configs:
        raise FileNotFoundError(f"No valid split configurations found in {base_output_path}")

    logger.info(f"Found {len(split_configs)} split configurations")
    return split_configs


def finetune_multiple_splits(base_output_path: str, new_output_base_dir: str, beta: float, model_type: str):
    """
    Fine-tune multiple models trained on different data splits.

    Args:
        base_output_path: Base directory containing data_split_x subdirectories with trained models
        new_output_base_dir: Base directory where fine-tuned models will be saved
        beta: F-Fidelity beta parameter (fraction of input that can be masked)
        model_type: Type of model ('GNN' or 'CNN')
    """

    logger.info(f"Starting F-Fidelity fine-tuning for multiple splits")
    logger.info(f"Base output path: {base_output_path}")
    logger.info(f"New output base directory: {new_output_base_dir}")
    logger.info(f"Beta parameter: {beta}")
    logger.info(f"Model type: {model_type}")

    # Find all split configurations
    split_configs = find_split_configs(base_output_path)

    successful_splits = []
    failed_splits = []

    # Fine-tune each split
    for split_name, config_path in split_configs:
        try:
            logger.info(f"Processing {split_name}")

            # Create output directory for this split
            split_output_dir = os.path.join(new_output_base_dir, split_name)

            # Fine-tune this split
            finetune_single_model(
                original_config_path=config_path,
                new_output_dir=split_output_dir,
                beta=beta,
                model_type=model_type
            )

            successful_splits.append(split_name)
            logger.info(f"Successfully completed fine-tuning for {split_name}")

        except Exception as e:
            failed_splits.append((split_name, str(e)))

    # Summary
    logger.info(f"Fine-tuning summary:")
    logger.info(f"Successfully processed: {len(successful_splits)} splits")
    logger.info(f"Failed: {len(failed_splits)} splits")

    if successful_splits:
        logger.info(f"Successful splits: {', '.join(successful_splits)}")

    if failed_splits:
        logger.warning(f"Failed splits:")
        for split_name, error in failed_splits:
            logger.warning(f"  {split_name}: {error}")

    if failed_splits:
        raise RuntimeError(f"Some splits failed to fine-tune. Check logs for details.")


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune a pre-trained neural network using F-Fidelity methodology'
    )

    parser.add_argument(
        '--config_path_or_base',
        type=str,
        required=True,
        help='Path to single model config file OR base directory containing data_split_x subdirectories'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where fine-tuned model and configuration will be saved'
    )

    parser.add_argument(
        '--beta',
        type=float,
        required=True,
        help='F-Fidelity beta parameter: maximum fraction of input elements that can be masked.'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        choices=SUPPORTED_DNN_MODEL_TYPES,
        required=True,
        help='Type of model to fine-tune.'
    )

    parser.add_argument(
        '--multiple_splits',
        action='store_true',
        help='Fine-tune multiple models from different data splits (expect data_split_x subdirectories)'
    )

    args = parser.parse_args()

    # Validate beta parameter
    if not 0.0 < args.beta <= 1.0:
        parser.error("Beta parameter must be between 0 and 1 (exclusive of 0)")

    # Validate paths
    if not os.path.exists(args.config_path_or_base):
        parser.error(f"Config path or base directory not found: {args.config_path_or_base}")

    try:
        # Run fine-tuning
        if args.multiple_splits:
            finetune_multiple_splits(
                base_output_path=args.config_path_or_base,
                new_output_base_dir=args.output_dir,
                beta=args.beta,
                model_type=args.model_type
            )
        else:
            finetune_single_model(
                original_config_path=args.config_path_or_base,
                new_output_dir=args.output_dir,
                beta=args.beta,
                model_type=args.model_type
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
