import json
import os
from pathlib import Path

from mol_faith.model.configs.dnn_configs import get_config
from mol_faith.model.configs.utils import save_config


def update_config_file(config_path: str, new_output_base: str, is_root_config: bool = False) -> bool:
    """
    Update the output_dir_path in a config file using mol_faith config utilities.
    For data_split configs: replaces everything before data_split_x with the new base path.
    For root config: replaces the entire path with the new base path.

    Args:
        config_path: Path to the config.yaml file
        new_output_base: New base path (everything before data_split_x)
        is_root_config: True if this is the root config file

    Returns:
        True if file was modified, False otherwise
    """
    try:
        # Load config using the mol_faith function
        config = get_config(config_path)

        if not hasattr(config, 'training') or not hasattr(config.training, 'output_dir_path'):
            print(f"Warning: No training.output_dir_path found in {config_path}")
            return False

        old_path = config.training.output_dir_path

        if is_root_config:
            # For root config, just use the new base path directly
            new_path = new_output_base
        else:
            # For data_split configs, find data_split_x pattern in the path
            import re
            match = re.search(r'(data_split_\d+.*)', old_path)

            if match:
                # Extract everything from data_split_x onwards
                data_split_suffix = match.group(1)
                new_path = os.path.join(new_output_base, data_split_suffix)
            else:
                print(f"No data_split_x pattern found in path: {old_path}")
                return False

        config.training.output_dir_path = new_path

        # Save config using the mol_faith function
        save_config(config, config_path)

        print(f"Updated {config_path}")
        print(f"  Old: {old_path}")
        print(f"  New: {new_path}")
        return True

    except Exception as e:
        print(f"Error updating {config_path}: {e}")
        return False


def update_checkpoint_metadata(metadata_path: str, new_output_base: str) -> bool:
    """
    Update the checkpoint paths in checkpoint_metadata.json.
    Replaces everything before data_split_x with the new base path.

    Args:
        metadata_path: Path to the checkpoint_metadata.json file
        new_output_base: New base path (everything before data_split_x)

    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if 'top_model_paths' not in metadata:
            print(f"Warning: No top_model_paths found in {metadata_path}")
            return False

        modified = False
        import re

        for model_info in metadata['top_model_paths']:
            if 'path' in model_info:
                old_path = model_info['path']

                # Find data_split_x pattern in the path
                match = re.search(r'(data_split_\d+.*)', old_path)

                if match:
                    # Extract everything from data_split_x onwards
                    data_split_suffix = match.group(1)
                    new_path = os.path.join(new_output_base, data_split_suffix)

                    model_info['path'] = new_path
                    modified = True
                    print(f"Updated checkpoint path in {metadata_path}")
                    print(f"  Old: {old_path}")
                    print(f"  New: {new_path}")
                else:
                    print(f"No data_split_x pattern found in path: {old_path}")

        if modified:
            # Write back to file with proper formatting
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        else:
            print(f"No paths updated in {metadata_path}")
            return False

    except Exception as e:
        print(f"Error updating {metadata_path}: {e}")
        return False


def update_dataset_folder_path(config_path: str, new_dataset_base: str) -> bool:
    """
    Update the dataset_folder_path in a config file.
    Replaces everything before data_split_x with the new base path.

    Args:
        config_path: Path to the config.yaml file
        new_dataset_base: New base path (everything before data_split_x)

    Returns:
        True if file was modified, False otherwise
    """
    try:
        config = get_config(config_path)

        if not hasattr(config, 'training') or not hasattr(config.training, 'dataset_folder_path'):
            print(f"Warning: No training.dataset_folder_path found in {config_path}")
            return False

        old_path = config.training.dataset_folder_path

        import re
        match = re.search(r'(data_split_\d+.*)', old_path)
        if match:
            data_split_suffix = match.group(1)
            new_path = os.path.join(new_dataset_base, data_split_suffix)
        else:
            print(f"No data_split_x pattern found in path: {old_path}")
            return False

        config.training.dataset_folder_path = new_path
        save_config(config, config_path)

        print(f"Updated {config_path}")
        print(f"  Old dataset_folder_path: {old_path}")
        print(f"  New dataset_folder_path: {new_path}")
        return True

    except Exception as e:
        print(f"Error updating {config_path}: {e}")
        return False


def find_and_update_configs(root_dir: str, new_output_base: str):
    """
    Find and update all config files in the directory structure.

    Args:
        root_dir: Root directory to search in
        new_output_base: New base path (everything before data_split_x)
    """
    root_path = Path(root_dir)

    # Update root config.yaml
    root_config = root_path / "config.yaml"
    if root_config.exists():
        print(f"\n=== Updating root config ===")
        update_config_file(str(root_config), new_output_base, is_root_config=True)

    # Find and update data_split_x configs
    print(f"\n=== Updating data split configs ===")
    for data_split_dir in root_path.glob("data_split_*"):
        if data_split_dir.is_dir():
            config_file = data_split_dir / "config.yaml"
            if config_file.exists():
                update_config_file(str(config_file), new_output_base, is_root_config=False)


def find_and_update_checkpoint_metadata(root_dir: str, new_output_base: str):
    """
    Find and update all checkpoint metadata files in the directory structure.

    Args:
        root_dir: Root directory to search in
        new_output_base: New base path (everything before data_split_x)
    """
    root_path = Path(root_dir)

    print(f"\n=== Updating checkpoint metadata ===")
    for data_split_dir in root_path.glob("data_split_*"):
        if data_split_dir.is_dir():
            metadata_file = data_split_dir / "checkpoints" / "checkpoint_metadata.json"
            if metadata_file.exists():
                update_checkpoint_metadata(str(metadata_file), new_output_base)


def find_and_update_dataset_paths(root_dir: str, new_dataset_base: str):
    """
    Find and update dataset_folder_path in all data_split_* config files.

    Args:
        root_dir: Root directory to search in
        new_dataset_base: New base path (everything before data_split_x)
    """
    root_path = Path(root_dir)
    print(f"\n=== Updating dataset_folder_path in data split configs ===")
    for data_split_dir in root_path.glob("data_split_*"):
        if data_split_dir.is_dir():
            config_file = data_split_dir / "config.yaml"
            if config_file.exists():
                update_dataset_folder_path(str(config_file), new_dataset_base)


def find_and_update_vocab_configs(root_dir: str, new_vocab_path: str):
    """
    Find and update vocab_path in all config files in the directory structure.

    Args:
        root_dir: Root directory to search in
        new_vocab_path: New vocab path
    """
    root_path = Path(root_dir)

    # Update root config.yaml
    root_config = root_path / "config.yaml"
    if root_config.exists():
        print(f"\n=== Updating root config vocab ===")
        update_vocab_config_file(str(root_config), new_vocab_path)

    # Find and update data_split_x configs
    print(f"\n=== Updating data split configs vocab ===")
    for data_split_dir in root_path.glob("data_split_*"):
        if data_split_dir.is_dir():
            config_file = data_split_dir / "config.yaml"
            if config_file.exists():
                update_vocab_config_file(str(config_file), new_vocab_path)


def update_vocab_config_file(config_path: str, new_vocab_path: str) -> bool:
    """
    Update the vocab_path in a config file using mol_faith config utilities.

    Args:
        config_path: Path to the config.yaml file
        new_vocab_path: New vocab path

    Returns:
        True if file was modified, False otherwise
    """
    try:
        # Load config using the mol_faith function
        config = get_config(config_path)

        if not hasattr(config, 'model') or not hasattr(config.model, 'vocab_path'):
            print(f"Warning: No model.vocab_path found in {config_path}")
            return False

        old_path = config.model.vocab_path
        config.model.vocab_path = new_vocab_path

        # Save config using the mol_faith function
        save_config(config, config_path)

        print(f"Updated vocab in {config_path}")
        print(f"  Old: {old_path}")
        print(f"  New: {new_vocab_path}")
        return True

    except Exception as e:
        print(f"Error updating vocab in {config_path}: {e}")
        return False


if __name__ == '__main__':
    root_dir = 'results/real_world/ABL1_pki/CNN/finetuning/hp_tuned_parameters/beta_0.1'
    new_output_base = 'results/real_world/ABL1_pki/CNN/finetuning/hp_tuned_parameters/beta_0.1'
    new_dataset_base = 'data/real_world/ABL1_pki/data_splits'
    new_vocab_path = 'data/real_world/ABL1_pki/cleaned_and_filtered/tokenizer_vocab.txt'

    find_and_update_configs(
        root_dir=root_dir,
        new_output_base=new_output_base)

    find_and_update_checkpoint_metadata(
        root_dir=root_dir,
        new_output_base=new_output_base)

    find_and_update_dataset_paths(
        root_dir=root_dir,
        new_dataset_base=new_dataset_base)

    find_and_update_vocab_configs(
        root_dir=root_dir,
        new_vocab_path=new_vocab_path)

    # for beta_value in [0, 1, 2, 3, 4]:
    #     root_dir_beta = f"{root_dir}{beta_value}"
    #     new_output_base_beta = f"{new_output_base}{beta_value}"
    #     find_and_update_configs(
    #         root_dir=root_dir_beta,
    #         new_output_base=new_output_base_beta)

    #     find_and_update_checkpoint_metadata(
    #         root_dir=root_dir_beta,
    #         new_output_base=new_output_base_beta)

    #     find_and_update_dataset_paths(
    #         root_dir=root_dir_beta,
    #         new_dataset_base=new_dataset_base)

    # find_and_update_vocab_configs(
    #     root_dir=root_dir_beta,
    #     new_vocab_path='/home/mhiltscher/data/logP/cleaned_and_filtered/tokenizer_vocab.txt')
