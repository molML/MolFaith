import logging
import os

import numpy as np
import wandb
import json

from mol_faith.model.configs.dnn_configs import TrainingConfig

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CheckpointHandler:
    def __init__(self, training_config: TrainingConfig, decreasing: bool = True) -> None:
        self.training_config = training_config
        self.checkpoint_dir_path = os.path.join(self.training_config.output_dir_path, 'checkpoints')
        os.makedirs(self.checkpoint_dir_path, exist_ok=True)
        self.model_file_name = 'model_ckpt'
        self.top_n = self.training_config.keep_n_checkpoints
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.inf if decreasing else -np.inf

        # Early stopping parameters
        self.early_stopping_patience = self.training_config.early_stopping_patience
        self.early_stopping_delta = self.training_config.early_stopping_delta
        self.moving_avg_window = self.training_config.moving_avg_window
        self.early_stopping_enabled = self.early_stopping_patience is not None
        self.early_stopping_counter = 0
        self.early_stop = False
        self.last_metric_val = np.inf if decreasing else -np.inf
        self.metric_history = []
        self.min_epochs_since_best = 25

        # Path for the metadata file that will store information about checkpoints
        self.metadata_path = os.path.join(self.checkpoint_dir_path, 'checkpoint_metadata.json')

        # Load existing metadata if it exists
        self._load_metadata()

    def __call__(self, model, epoch, metric_val):
        """
        Process a new metric value for checkpointing and early stopping.

        Args:
            model: The model to potentially save
            epoch: Current epoch number
            metric_val: Current metric value to evaluate

        Returns:
            bool: True if early stopping criteria is met, False otherwise
        """
        if np.isnan(metric_val):
            logger.warning(f"Metric value is NaN at epoch {epoch}. Ignoring this epoch for early stopping.")
            return self.early_stop if self.early_stopping_enabled else False

        if self.early_stopping_enabled:
            self._update_early_stopping(metric_val, epoch)

        model_path = os.path.join(self.checkpoint_dir_path, model.__class__.__name__ + f'_epoch_{epoch}.pt')
        save = metric_val < self.best_metric_val if self.decreasing else metric_val > self.best_metric_val

        if not save:
            return self.early_stop if self.early_stopping_enabled else False

        logger.info(f"Current metric value {metric_val} better than best {self.best_metric_val}, "
                     f"saving model at {model_path}")

        self.last_best_epoch = epoch

        self.best_metric_val = metric_val
        model.save_params(path=model_path)

        # Add new model to top_model_paths
        self.top_model_paths.append({
            'path': model_path,
            'score': float(metric_val),
            'epoch': epoch,
            'is_best': True
        })

        # Update previous best model
        for entry in self.top_model_paths[:-1]:
            entry['is_best'] = False

        # Sort models by score
        self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)

        # Save metadata after each update
        self._save_metadata()

        if len(self.top_model_paths) > self.top_n:
            self.cleanup()

        return self.early_stop if self.early_stopping_enabled else False

    def _update_early_stopping(self, metric_val, epoch):
        """Update early stopping state based on current metric value using moving average."""
        # Add current metric to history
        self.metric_history.append(metric_val)

        # Keep only the window size for moving average
        if len(self.metric_history) > self.moving_avg_window:
            self.metric_history.pop(0)

        # Need at least 2 windows worth of data to start comparing moving averages
        if len(self.metric_history) < self.moving_avg_window:
            logger.info(f"Early stopping: collecting data "
                         f"({len(self.metric_history)}/{self.moving_avg_window}), current: {metric_val:.6f}")
            return

        # Calculate current moving average
        current_avg = np.mean(self.metric_history)

        # Only start counting if we haven't improved best model in a while
        epochs_since_best = epoch - getattr(self, 'last_best_epoch', 0)

        if epochs_since_best < self.min_epochs_since_best:
            # Too early to consider stopping - reset counter
            self.early_stopping_counter = 0
            logger.info(
                f"Still within grace period ({epochs_since_best}/{self.min_epochs_since_best} epochs since best)")
            self.last_metric_val = current_avg
            return

        # For the first comparison, initialize with current average
        if self.last_metric_val == (np.inf if self.decreasing else -np.inf):
            self.last_metric_val = current_avg
            logger.info(f"Early stopping initialized with moving average: {current_avg:.6f}")
            return

        # Check for improvement in moving average compared to previous moving average
        improvement_condition = (
                (self.decreasing and current_avg < self.last_metric_val - self.early_stopping_delta) or
                (not self.decreasing and current_avg > self.last_metric_val + self.early_stopping_delta)
        )

        if improvement_condition:
            # Moving average improved significantly
            self.early_stopping_counter = 0
            direction = "decreased" if self.decreasing else "increased"
            logger.info(f"Moving average {direction} from {self.last_metric_val:.6f} to {current_avg:.6f}. "
                         f"Resetting early stopping counter.")
        else:
            # No significant improvement in moving average
            self.early_stopping_counter += 1
            logger.info(f'Early stopping counter: {self.early_stopping_counter} out of {self.early_stopping_patience} '
                         f'(current avg: {current_avg:.6f}, previous avg: {self.last_metric_val:.6f}, '
                         f'current val: {metric_val:.6f})')

            if self.early_stopping_counter >= self.early_stopping_patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}. "
                             f"Moving average stagnated for {self.early_stopping_patience} consecutive epochs.")

        # Update last moving average for next comparison
        self.last_metric_val = current_avg

    def should_stop_early(self):
        """
        Check if early stopping criteria has been met.

        Returns:
            bool: True if training should be stopped early, False otherwise
        """
        return self.early_stop if self.early_stopping_enabled else False

    def get_early_stopping_info(self):
        """
        Get current early stopping state information.

        Returns:
            dict: Dictionary containing early stopping state info
        """
        if not self.early_stopping_enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "patience": self.early_stopping_patience,
            "counter": self.early_stopping_counter,
            "early_stop": self.early_stop,
            "best_metric": self.best_metric_val,
            "current_moving_avg": np.mean(self.metric_history) if self.metric_history else None,
            "previous_moving_avg": self.last_metric_val,
            "delta": self.early_stopping_delta,
            "window_size": self.moving_avg_window
        }

    def reset_early_stopping(self):
        """Reset early stopping state (useful for resuming training)."""
        if self.early_stopping_enabled:
            self.early_stopping_counter = 0
            self.early_stop = False
            self.last_metric_val = np.inf if self.decreasing else -np.inf
            self.metric_history = []
            logger.info("Early stopping state has been reset.")

    @staticmethod
    def log_artifact(filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact).wait()

    def _remove_local_checkpoints(self):
        to_remove = self.top_model_paths[self.top_n:]
        logger.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            try:
                os.remove(o['path'])
            except FileNotFoundError:
                logger.warning(f"Couldn't find file to remove: {o['path']}")
        self.top_model_paths = self.top_model_paths[:self.top_n]
        self._save_metadata()

    def _remove_wandb_checkpoints(self):
        api = wandb.Api(overrides={"project": self.training_config.wandb_project_name})
        artifacts = api.artifacts(name=f'{self.model_file_name}', type_name='model')
        artifacts_sorted = sorted(artifacts, key=lambda artifact: int(artifact.version[1:]), reverse=True)
        for artifact in artifacts_sorted[self.top_n:]:
            logger.info(f"Deleting wandb artifact: {artifact.name} (Version: {artifact.version})")
            artifact.delete()

    def cleanup(self):
        self._remove_local_checkpoints()

    def _save_metadata(self):
        metadata = {
            'top_model_paths': self.top_model_paths,
            'best_metric_val': float(self.best_metric_val),
            'decreasing': self.decreasing
        }

        # Add early stopping metadata if enabled
        if self.early_stopping_enabled:
            metadata['early_stopping'] = {
                'patience': self.early_stopping_patience,
                'delta': self.early_stopping_delta,
                'counter': self.early_stopping_counter,
                'early_stop': self.early_stop
            }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Checkpoint metadata saved to {self.metadata_path}")

    def _load_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                self.top_model_paths = metadata['top_model_paths']
                self.best_metric_val = metadata['best_metric_val']

                # Only load decreasing if it matches initialization value
                if metadata.get('decreasing') == self.decreasing:
                    logger.info(f"Loaded checkpoint metadata from {self.metadata_path}")
                else:
                    logger.warning("Loaded metadata has different 'decreasing' value than current initialization")

                # Load early stopping metadata if available and enabled
                if self.early_stopping_enabled and 'early_stopping' in metadata:
                    es_data = metadata['early_stopping']
                    if es_data.get('patience') == self.early_stopping_patience:
                        self.early_stopping_counter = es_data.get('counter', 0)
                        self.early_stop = es_data.get('early_stop', False)
                        self.last_metric_val = es_data.get('last_moving_avg', np.inf if self.decreasing else -np.inf)
                        self.metric_history = es_data.get('metric_history', [])
                        self.moving_avg_window = es_data.get('window_size', self.moving_avg_window)
                        logger.info("Loaded early stopping state from metadata")
                    else:
                        logger.warning("Early stopping patience in metadata differs from current setting")

    def get_best_model_path(self):
        for entry in self.top_model_paths:
            if entry.get('is_best', False):
                return entry['path']

        # If no best model is explicitly marked, return the first one (best by score)
        if self.top_model_paths:
            return self.top_model_paths[0]['path']

        return None