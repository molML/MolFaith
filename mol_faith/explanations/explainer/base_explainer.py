import time
from abc import abstractmethod
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Iterator

from mol_faith.explanations.explainer.processing_modes import ProcessingCapabilties
from mol_faith.explanations.types import ExplanationMethod
from mol_faith.explanations.explainer.utils import Attributions


class Explainer:
    def __init__(self, explanation_method: ExplanationMethod, trained_model_path: str,
                 use_normalized_attributions: bool = False,
                 explanation_methods_args: Optional[Dict] = None):
        self.trained_model_path = trained_model_path
        self.explanation_method = explanation_method
        self.explanation_methods_args = explanation_methods_args or {}
        self.use_normalized_attributions = use_normalized_attributions
        self.need_initial_training = False
        self.batch_size = 64

    @property
    def trained_model_path(self) -> str:
        return self._trained_model_path

    @trained_model_path.setter
    def trained_model_path(self, new_path: str) -> None:
        self._trained_model_path = new_path

    @property
    def explanation_method(self) -> ExplanationMethod:
        return self._explanation_method

    @explanation_method.setter
    def explanation_method(self, new_explanation_method: ExplanationMethod) -> None:
        self._explanation_method = new_explanation_method

    @property
    def need_initial_training(self) -> bool:
        return self._need_initial_training

    @need_initial_training.setter
    def need_initial_training(self, need_initial_training: bool):
        self._need_initial_training = need_initial_training

    @property
    @abstractmethod
    def explanation_method_capabilities(self) -> ProcessingCapabilties:
        """Get capabilities of current explanation method."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_capabilities(self) -> ProcessingCapabilties:
        """Get capabilities of the model architecture."""
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        """Get the model."""
        raise NotImplementedError

    @abstractmethod
    def _get_attribution_provider(self):
        """Get the attribution wrapper. It will be used to get attributions."""
        raise NotImplementedError

    def get_attributions(self, smiles_data: Union[str, List[str]], target_values: Union[float, List[float]]) -> Union[
        Attributions, List[Attributions]]:
        """
        Get attributions for one or multiple SMILES strings using the most efficient available processing mode.
        
        Args:
            smiles_data: Single SMILES string or list of SMILES strings
            target_values: Single target value or list of target values
            
        Returns:
            Single Attribution or list of Attributions depending on input
        """
        # Convert single inputs to lists for uniform processing
        single_input = isinstance(smiles_data, str)
        smiles_list = [smiles_data] if single_input else smiles_data
        target_list = [target_values] if single_input else target_values

        # Choose best available processing mode
        if self.explanation_method_capabilities.supports_batch:
            results = self._get_attributions_batch(smiles_list, target_list)
        elif self.explanation_method_capabilities.supports_mp:
            results = self._get_attributions_mp(smiles_list, target_list)
        else:
            results = self._get_attributions_sequential(smiles_list, target_list)
        return results[0] if single_input else results

    @abstractmethod
    def get_single_attributions_scores(self, smiles: str, target_value: float) -> Attributions:
        """Process a single SMILES string to get attributions. All explainers must implement this."""
        raise NotImplementedError

    def _get_attributions_sequential(self, smiles_list: List[str], target_values: List[float]) -> List[Attributions]:
        """Sequential processing using process_single_smiles."""
        return [self.get_single_attributions_scores(smiles, target)
                for smiles, target in tqdm(zip(smiles_list, target_values), total=len(smiles_list))]

    def _get_attributions_batch(self, smiles_list: List[str], target_values: List[float]) -> List[Attributions]:
        """Batch processing. Override if explainer supports batch processing."""
        return self._get_attributions_sequential(smiles_list, target_values)

    def _get_attributions_mp(self, smiles_list: List[str], target_values: List[float]) -> List[Attributions]:
        """Multiprocessing using process_single_smiles. Override if different MP implementation needed."""
        from multiprocess import Pool, cpu_count

        num_workers = min(cpu_count(), 5)

        # Create argument tuples for the worker function
        work_args = [(self, smiles, target)
                     for smiles, target in zip(smiles_list, target_values)]

        with Pool(num_workers) as pool:
            return pool.map(_process_single_smiles_mp, work_args)

    def get_model_prediction_error(self, smiles_data: Union[str, List[str]],
                                   target_values: Union[float, List[float]]) -> Union[float, List[float]]:
        """
        Get model prediction error for one or multiple SMILES strings using the most efficient available processing mode.
        
        Args:
            smiles_data: Single SMILES string or list of SMILES strings
            target_values: Single target value or list of target values
            
        Returns:
            Single error value or list of error values depending on input
        """
        # Convert single inputs to lists for uniform processing
        single_input = isinstance(smiles_data, str)
        smiles_list = [smiles_data] if single_input else smiles_data
        target_list = [target_values] if single_input else target_values

        # Choose best available processing mode
        if self.model_capabilities.supports_batch:
            results = self._get_prediction_error_batch(smiles_list, target_list)
        elif self.model_capabilities.supports_mp:
            results = self._get_prediction_error_mp(smiles_list, target_list)
        else:
            results = self._get_prediction_error_sequential(smiles_list, target_list)

        return results[0] if single_input else results

    @abstractmethod
    def get_single_prediction_error(self, smiles: str, target_value: float) -> float:
        """Process a single SMILES string to get prediction error. All explainers must implement this."""
        raise NotImplementedError

    def _get_prediction_error_sequential(self, smiles_list: List[str], target_values: List[float]) -> List[float]:
        """Sequential processing using process_single_prediction_error."""
        return [self.get_single_prediction_error(smiles, target)
                for smiles, target in zip(smiles_list, target_values)]

    def _get_prediction_error_batch(self, smiles_list: List[str], target_values: List[float]) -> List[float]:
        """Batch processing. Override if explainer supports batch processing."""
        return self._get_prediction_error_sequential(smiles_list, target_values)

    def _get_prediction_error_mp(self, smiles_list: List[str], target_values: List[float]) -> List[float]:
        """Multiprocessing using process_single_prediction_error."""
        from multiprocessing import Pool, cpu_count
        num_workers = min(cpu_count(), 7)
        with Pool(num_workers) as pool:
            return pool.starmap(self.get_single_prediction_error,
                                [(smiles, target) for smiles, target in zip(smiles_list, target_values)])

    @abstractmethod
    def train_explainer(self) -> None:
        raise NotImplementedError

    def _batch_iterator(self, smiles_list: List[str], target_values: List[float]) -> Iterator[
        tuple[List[str], List[float]]]:
        """Create batches of data for processing."""
        for i in range(0, len(smiles_list), self.batch_size):
            batch_smiles = smiles_list[i:i + self.batch_size]
            batch_targets = target_values[i:i + self.batch_size]
            yield batch_smiles, batch_targets


def _process_single_smiles_mp(args):
    """External worker function for multiprocessing.
    
    Args:
        args: Tuple containing (explainer_instance, smiles, target)
    """
    explainer, smiles, target = args
    return explainer.get_single_attributions_scores(smiles, target)
