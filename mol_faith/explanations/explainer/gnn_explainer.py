import numpy as np
import os
import torch
from torch_geometric.data import Batch
from torch_geometric.explain import Explanation
from typing import List, Optional

from mol_faith.data.data_managers.graph_data_manager import SupportedFeatureProviders, PytorchGeometricFeatures, \
    FeatureProviders
from mol_faith.explanations.attribution_providers.captum_methods_provider import CaptumAttributionProvider
from mol_faith.explanations.attribution_providers.pyg_methods_provider import PyTorchGeometricProvider
from mol_faith.explanations.explainer.base_explainer import Explainer
from mol_faith.explanations.explainer.processing_modes import ProcessingCapabilties
from mol_faith.explanations.explainer.utils import normalize_attributions, Attributions
from mol_faith.explanations.types import ExplanationMethod, ExplanationMethodArgs
from mol_faith.explanations.wrappers.gnn_wrappers import CaptumModelEmbeddingWrapper, PyGEmbeddingWrapper
from mol_faith.model.GNN.base_gnn import BasicGNNRegressor
from mol_faith.model.GNN.gat_model import GAT
from mol_faith.model.GNN.gcn_model import GCN
from mol_faith.model.GNN.gin_model import GIN
from mol_faith.model.configs.dnn_configs import get_config, DNNConfig, DNNModelNames
from mol_faith.training.dnn_training.checkpoint_handler import CheckpointHandler

SUPPORTED_MODELS = [DNNModelNames.GCN.value, DNNModelNames.GAT.value, DNNModelNames.GIN.value]


class GNNExplainer(Explainer):
    def __init__(self, use_normalized_attributions: bool, trained_model_path: str,
                 explanation_method: ExplanationMethod, use_randomized_weights: bool,
                 explanation_methods_args: Optional[ExplanationMethodArgs] = None):
        super().__init__(use_normalized_attributions=use_normalized_attributions, explanation_method=explanation_method,
                         trained_model_path=trained_model_path, explanation_methods_args=explanation_methods_args)
        self.use_randomized_weights: bool = use_randomized_weights
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config: DNNConfig = get_config(config_file_path=os.path.join(self.trained_model_path, 'config.yaml'))
        self.feature_provider: FeatureProviders = self.get_feature_provider()
        self.model = self.get_model()
        self.attribution_provider = self._get_attribution_provider()

    def _get_last_conv_layer(self):
        """Get the last (final) convolutional layer for GradCAM and GuidedGradCAM."""
        if hasattr(self.model, 'encoder') and len(self.model.encoder) > 0:
            # Get the last GNN block (final layer in the sequence)
            last_block = self.model.encoder[-1]

            if hasattr(last_block, 'gnn_layer'):
                return last_block.gnn_layer
            else:
                raise ValueError("Last encoder block doesn't have a gnn_layer attribute")
        else:
            raise ValueError("Model doesn't have encoder layers or encoder is empty")

    def _get_attribution_provider(self):
        """Get appropriate attribution provider based on explanation method."""
        if self.explanation_method in [ExplanationMethod.GNN_EXPLAINER, ExplanationMethod.GRAPH_MASK]:
            return PyTorchGeometricProvider(
                model=PyGEmbeddingWrapper(self.model),
                method=self.explanation_method,
                method_args=self.explanation_methods_args
            )
        elif self.explanation_method in [ExplanationMethod.GRAD_CAM, ExplanationMethod.GUIDED_GRAD_CAM]:
            if self.model.__class__.__name__ == 'GAT' or self.model.__class__.__name__ == 'GIN':
                raise ValueError('Grad-CAM based xAI methods are not supported for GAT or GIN models.')

            if 'explainer_args' not in self.explanation_methods_args:
                self.explanation_methods_args['explainer_args'] = {}

            self.explanation_methods_args['explainer_args']['target_layer'] = self._get_last_conv_layer()

            return CaptumAttributionProvider(
                model=CaptumModelEmbeddingWrapper(self.model),
                method=self.explanation_method,
                method_args=self.explanation_methods_args
            )
        elif self.explanation_method in [ExplanationMethod.KERNEL_SHAP,
                                         ExplanationMethod.LIME,
                                         ExplanationMethod.INTEGRATED_GRADIENTS,
                                         ExplanationMethod.SHAP_SAMPLING]:
            return CaptumAttributionProvider(
                model=CaptumModelEmbeddingWrapper(self.model),
                method=self.explanation_method,
                method_args=self.explanation_methods_args
            )
        else:
            raise ValueError(f'Unsupported explanation method: {self.explanation_method}')

    @property
    def explanation_method_capabilities(self) -> ProcessingCapabilties:
        """Get capabilities of the GNNs data processing for the explanation method."""
        # Independent of the xAI method, we can't support batch processing because of the data structure of
        # pytorch geometric and the input of e.g. captum
        # We currently do not support multiprocessing for the GNN
        return ProcessingCapabilties(
            supports_batch=False,
            supports_mp=False
        )

    @property
    def model_capabilities(self) -> ProcessingCapabilties:
        """Get capabilities of the GNN model."""
        return ProcessingCapabilties(
            supports_batch=True,
            supports_mp=False
        )

    @staticmethod
    def extract_positive_node_attributions(attributions: np.ndarray) -> np.ndarray:
        """Extract and sum only positive attribution scores for each node."""
        return np.sum(np.maximum(attributions, 0), axis=1)

    def extract_attributions(self, explanation: Explanation, batch_idx: Optional[int] = None) -> Attributions:
        """Extract and process attributions for a single molecule from an explanation object.
        
        Args:
            explanation: The explanation object from the explainer
            batch_idx: Index in the batch to extract attributions for (None for single molecule)
            
        Returns:
            Processed attributions for a single molecule
        """
        if batch_idx is None:
            # Single molecule case
            atom_attributions = explanation.node_stores[0]['node_mask'].detach().cpu().numpy()
        else:
            # Part of a batch - use batch_index to get the right nodes
            batch_mask = explanation.node_stores[0]['batch_index'] == batch_idx
            atom_attributions = explanation.node_stores[0]['node_mask'][batch_mask].detach().cpu().numpy()

        # Get positive attributions
        atom_attributions = self.extract_positive_node_attributions(atom_attributions)

        if self.use_normalized_attributions:
            atom_attributions = normalize_attributions(attributions=atom_attributions)

        return Attributions(atom_attributions=atom_attributions)

    def get_feature_provider(self) -> PytorchGeometricFeatures:
        if self.config.model.feature_provider == SupportedFeatureProviders.PYTORCH_GEOMETRIC.value:
            feature_provider = PytorchGeometricFeatures()
        else:
            raise ValueError(f'Feature provider {self.config.model.feature_provider} is not supported!')

        return feature_provider

    def create_data(self, smiles: str, target_value: float):
        data = self.feature_provider(smiles)

        # Make artificial batch
        data.batch = torch.zeros(data.x.size(0), dtype=torch.int64)

        data.y = torch.tensor(target_value, dtype=torch.float)

        data.x = data.x.float()
        data.edge_attr = data.edge_attr.float()
        data.edge_index = data.edge_index

        return data.to(self.device)

    def train_explainer(self) -> None:
        # Current xAI methods do not need a training or cannot be trained once before
        pass

    def instantiate_model(self) -> BasicGNNRegressor:
        # ToDo: Extract these functionalities to a utils file and use it here as well as in the gnn_trainer.py
        # Set input channels based on used features
        self.config.model.c_in = self.feature_provider.num_node_features

        model_classes = {DNNModelNames.GCN.value: GCN, DNNModelNames.GAT.value: GAT, DNNModelNames.GIN.value: GIN}
        model_class = model_classes.get(self.config.training.model_name)

        if not model_class:
            raise ValueError(
                f"Unsupported model: {self.config.training.model_name}. Supported models: {SUPPORTED_MODELS}.")

        return model_class(config=self.config.model)

    def get_model(self) -> BasicGNNRegressor:
        model = self.instantiate_model()

        if self.use_randomized_weights:
            model.reset_parameters()
        else:
            checkpoint_handler = CheckpointHandler(training_config=self.config.training)
            best_model_path = checkpoint_handler.get_best_model_path()
            model.load_params(path=best_model_path)

        model.eval()
        model.to(self.device)

        return model

    def get_single_attributions_scores(self, smiles: str, target_value: float) -> Attributions:
        """Process a single SMILES string to get attributions."""
        data = self.create_data(smiles=smiles, target_value=target_value)

        node_mask = self.attribution_provider.explain(data)

        if self.use_normalized_attributions:
            node_mask = normalize_attributions(attributions=node_mask)

        return Attributions(atom_attributions=node_mask)

    def get_single_prediction_error(self, smiles: str, target_value: float) -> float:
        """Process a single SMILES string to get prediction error."""
        data = self.create_data(smiles=smiles, target_value=target_value)
        prediction = self.model(data).item()
        return (target_value - prediction) ** 2

    def _get_prediction_error_batch(self, smiles_list: List[str], target_values: List[float]) -> List[float]:
        """Process prediction errors in batches."""
        all_errors = []

        for batch_smiles, batch_targets in self._batch_iterator(smiles_list, target_values):
            # Convert batch to graph data
            batch_data = Batch.from_data_list([
                self.create_data(smiles=smiles, target_value=target)
                for smiles, target in zip(batch_smiles, batch_targets)
            ])

            with torch.no_grad():
                predictions = self.model(batch_data).detach().cpu().numpy()

            # Calculate errors for this batch
            batch_errors = [(target - pred) ** 2 for target, pred in zip(batch_targets, predictions)]
            all_errors.extend(batch_errors)

        return all_errors


if __name__ == '__main__':
    explainer = GNNExplainer(use_normalized_attributions=False, use_randomized_weights=False,
                             trained_model_path='results/benzene_target/GIN/pretraining/hp_tuned_parameters/data_split_0',
                             explanation_method=ExplanationMethod.GNN_EXPLAINER)
    smiles = 'CNC(=O)N(C)C(=O)Nc1cc(OC(C)C)ccc1F'
    target = 0.25
    explanation = explainer.get_single_attributions_scores(smiles=smiles, target_value=target)
    # mol = Chem.MolFromSmiles(smiles)
    # if mol is None:
    #     print(f"Invalid SMILES: {smiles}")
    #
    # rdDepictor.Compute2DCoords(mol)
    #
    # drawer = rdMolDraw2D.MolDraw2DSVG(600, 400)
    #
    # # Add atom indices
    # for atom in mol.GetAtoms():
    #     atom.SetProp('atomLabel', str(atom.GetIdx()))
    #
    # drawer.DrawMolecule(mol)
    # drawer.FinishDrawing()
    #
    # with open('molecule.svg', 'w') as f:
    #     f.write(drawer.GetDrawingText())

    print(explanation.atom_attributions)
    print(explanation.atom_attributions.shape)
