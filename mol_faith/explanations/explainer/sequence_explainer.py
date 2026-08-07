import os
import torch
from typing import List, Optional, Union

from mol_faith.data.data_managers.smiles_data_manager import SMILESTokenizer
from mol_faith.explanations.attribution_providers.captum_methods_provider import CaptumAttributionProvider
from mol_faith.explanations.explainer.base_explainer import Explainer
from mol_faith.explanations.explainer.processing_modes import ProcessingCapabilties
from mol_faith.explanations.explainer.utils import normalize_attributions, Attributions
from mol_faith.explanations.types import ExplanationMethod, ExplanationMethodArgs
from mol_faith.explanations.wrappers.cnn_wrappers import \
    CaptumModelEmbeddingWrapper as CNNCaptumEmbeddingWrapper
from mol_faith.explanations.wrappers.transformer_wrapper import \
    CaptumModelEmbeddingWrapper as TransformerCaptumEmbeddingWrapper
from mol_faith.model.CNN.cnn_model import BasicCNNRegressor
from mol_faith.model.configs.dnn_configs import get_config, DNNConfig
from mol_faith.model.transformer.transformer import BasicTransformerRegressor
from mol_faith.training.dnn_training.checkpoint_handler import CheckpointHandler


class SequenceExplainer(Explainer):
    def __init__(self, use_normalized_attributions: bool, trained_model_path: str,
                 explanation_method: ExplanationMethod, use_randomized_weights: bool,
                 explanation_methods_args: Optional[ExplanationMethodArgs] = None):
        super().__init__(use_normalized_attributions=use_normalized_attributions,
                         explanation_method=explanation_method,
                         trained_model_path=trained_model_path,
                         explanation_methods_args=explanation_methods_args)
        self.use_randomized_weights: bool = use_randomized_weights
        self.config: DNNConfig = get_config(config_file_path=os.path.join(self.trained_model_path, 'config.yaml'))
        self.device = self.config.training.device
        self.tokenizer: SMILESTokenizer = self._load_tokenizer()
        self.model = self.get_model()
        self.attribution_provider = self._get_attribution_provider()

    def _load_tokenizer(self) -> SMILESTokenizer:
        """Load the SMILES tokenizer from the vocab path in config."""
        vocab_path = self.config.model.vocab_path
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at: {vocab_path}")
        return SMILESTokenizer.load_vocab(vocab_path)

    def _get_model_wrapper(self):
        """Get appropriate model wrapper based on model type."""
        model_name = self.config.training.model_name

        if model_name == 'CNN':
            return CNNCaptumEmbeddingWrapper(self.model)
        elif model_name == 'Transformer':
            return TransformerCaptumEmbeddingWrapper(self.model)
        else:
            raise ValueError(f'Unsupported model type: {model_name}')

    def _get_attribution_provider(self):
        """Get appropriate attribution provider based on explanation method."""
        model_wrapper = self._get_model_wrapper()

        if self.explanation_method in [ExplanationMethod.KERNEL_SHAP,
                                       ExplanationMethod.LIME,
                                       ExplanationMethod.INTEGRATED_GRADIENTS,
                                       ExplanationMethod.SHAP_SAMPLING]:
            return CaptumAttributionProvider(
                model=model_wrapper,
                method=self.explanation_method,
                method_args=self.explanation_methods_args
            )
        elif self.explanation_method in [ExplanationMethod.GRAD_CAM, ExplanationMethod.GUIDED_GRAD_CAM]:
            model_name = self.config.training.model_name

            # GradCAM based methods are not applicable for Transformers
            if model_name == 'Transformer':
                raise ValueError('Grad-CAM based methods may not be suitable for Transformer models.')

            if 'explainer_args' not in self.explanation_methods_args:
                self.explanation_methods_args['explainer_args'] = {}

            self.explanation_methods_args['explainer_args']['target_layers'] = self._get_cnn_layers_for_gradcam()

            return CaptumAttributionProvider(
                model=model_wrapper,
                method=self.explanation_method,
                method_args=self.explanation_methods_args
            )
        else:
            raise ValueError(f'Unsupported explanation method: {self.explanation_method}')

    def _get_cnn_layers_for_gradcam(self) -> List:
        """Get all CNN conv layers for multi-layer GradCAM."""
        layers = []
        for branch in self.model.encoder:
            # Get the Conv1d layer from each branch
            conv_layer = branch.conv_block[0]  # The Conv1d layer
            layers.append(conv_layer)
        return layers

    @property
    def explanation_method_capabilities(self) -> ProcessingCapabilties:
        """Get capabilities of the CNN data processing for the explanation method."""
        # CNNs can potentially support batch processing for explanations
        # but multiprocessing depends on the specific explanation method
        return ProcessingCapabilties(
            supports_batch=True,
            supports_mp=False
        )

    @property
    def model_capabilities(self) -> ProcessingCapabilties:
        """Get capabilities of the CNN model."""
        return ProcessingCapabilties(
            supports_batch=True,
            supports_mp=False
        )

    def create_data(self, smiles: str, target_value: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Create input tensor and target tensor from SMILES string."""
        # Tokenize and encode SMILES
        input_tensor = self.tokenizer.encode(
            smiles=smiles,
            max_seq_len=self.config.model.max_seq_len,
            padding=True
        ).unsqueeze(0)  # Add batch dimension

        # Create target tensor
        target_tensor = torch.tensor(target_value, dtype=torch.float)

        # Move to device
        input_tensor = input_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)

        return input_tensor, target_tensor

    def train_explainer(self) -> None:
        """Current XAI methods do not need training or cannot be trained once before."""
        pass

    def instantiate_model(self) -> Union[BasicCNNRegressor, BasicTransformerRegressor]:
        """Instantiate the appropriate model based on config."""
        model_name = self.config.training.model_name

        if model_name == 'CNN':
            return BasicCNNRegressor(config=self.config.model, tokenizer=self.tokenizer)
        elif model_name == 'Transformer':
            return BasicTransformerRegressor(config=self.config.model, tokenizer=self.tokenizer)
        else:
            raise ValueError(f'Unsupported model type: {model_name}')

    def get_model(self) -> BasicCNNRegressor:
        model = self.instantiate_model()

        if self.use_randomized_weights:
            # Reset to random weights if needed
            model.reset_parameters()
        else:
            # Load trained weights
            checkpoint_handler = CheckpointHandler(training_config=self.config.training)
            best_model_path = checkpoint_handler.get_best_model_path()
            model.load_params(path=best_model_path)

        model.eval()
        model.to(self.device)

        return model

    def get_single_attributions_scores(self, smiles: str, target_value: float) -> Attributions:
        """Process a single SMILES string to get attributions."""
        input_tensor, _ = self.create_data(smiles=smiles, target_value=target_value)

        attributions = self.attribution_provider.explain(input_tensor)

        if self.use_normalized_attributions:
            attributions = normalize_attributions(attributions=attributions)

        return Attributions(atom_attributions=attributions)

    def get_single_prediction_error(self, smiles: str, target_value: float) -> float:
        """Process a single SMILES string to get prediction error."""
        input_tensor, target_tensor = self.create_data(smiles=smiles, target_value=target_value)

        with torch.no_grad():
            prediction = self.model(input_tensor).item()

        return (target_value - prediction) ** 2

    def _get_prediction_error_batch(self, smiles_list: List[str], target_values: List[float]) -> List[float]:
        """Process prediction errors in batches."""
        all_errors = []

        for batch_smiles, batch_targets in self._batch_iterator(smiles_list, target_values):
            # Create batch tensors
            batch_inputs = []
            for smiles in batch_smiles:
                input_tensor = self.tokenizer.encode(
                    smiles=smiles,
                    max_seq_len=self.config.model.max_seq_len,
                    padding=True
                )
                batch_inputs.append(input_tensor)

            # Stack into batch
            batch_tensor = torch.stack(batch_inputs).to(self.device)

            # Get predictions for batch
            with torch.no_grad():
                predictions = self.model(batch_tensor).squeeze().detach().cpu().numpy()

            # Handle single prediction case
            if predictions.ndim == 0:
                predictions = [predictions.item()]

            # Calculate errors for this batch
            batch_errors = [(target - pred) ** 2 for target, pred in zip(batch_targets, predictions)]
            all_errors.extend(batch_errors)

        return all_errors


if __name__ == '__main__':
    # CNN: results/benzene_target/CNN/finetuning/c/beta_0.1/data_split_0
    # Transformer: results/benzene_target/Transformer/finetuning/hp_tuned_parameters/beta_0.1/data_split_0
    explainer = SequenceExplainer(use_normalized_attributions=False, use_randomized_weights=False,
                                  trained_model_path='results/benzene_target/CNN/finetuning/hp_tuned_parameters/beta_0.1/data_split_0',
                                  explanation_method=ExplanationMethod.GRAPH_MASK)
    smiles = 'CC(=O)[C]1C=Nc2ccccc21'
    target = 0.25
    explanation = explainer.get_single_attributions_scores(smiles=smiles, target_value=target)
    print(explanation.atom_attributions)
