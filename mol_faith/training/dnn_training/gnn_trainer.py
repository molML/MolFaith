from torch_geometric.data import Data

from mol_faith.masking.mask_generator import GraphFFidelityMasker
from mol_faith.model.configs.dnn_configs import DNNModelNames
from mol_faith.data.data_managers.graph_data_manager import GraphDataManager, PytorchGeometricFeatures, SupportedFeatureProviders
from mol_faith.model.GNN.gat_model import GAT
from mol_faith.model.GNN.gcn_model import GCN
from mol_faith.model.GNN.gin_model import GIN
from mol_faith.training.dnn_training.trainer import DNNTrainer


class GNNTrainer(DNNTrainer):
    def __init__(self, config_file_path: str):
        super().__init__(config_file_path)

    def set_missing_configs(self, num_node_features: int):
        # Set input channels based on used features
        self.config.model.c_in = num_node_features

    def _get_data_manager(self):
        if self.config.model.feature_provider == SupportedFeatureProviders.PYTORCH_GEOMETRIC.value:
            feature_provider = PytorchGeometricFeatures()

            # Depending on the features used for training, we need to set it in the model config!
            self.set_missing_configs(num_node_features=feature_provider.num_node_features)

            return GraphDataManager(data_path=self.config.training.dataset_folder_path,
                                    batch_size=self.config.training.batch_size,
                                    scale_features=self.config.model.scale_features,
                                    feature_provider=feature_provider)
        else:
            raise ValueError(f'Feature provider {self.config.model.feature_provider} is not supported!')

    def _get_mask_generator(self):
        if self.config.training.finetune_beta:
            return GraphFFidelityMasker(beta=self.config.training.finetune_beta, device=self.config.training.device)
        else:
            return None

    def _get_model(self):
        model_config = self.config.model
        model_name = self.config.training.model_name

        if model_name == DNNModelNames.GCN.value:
            model = GCN(config=model_config)
        elif model_name == DNNModelNames.GAT.value:
            model = GAT(config=model_config)
        elif model_name == DNNModelNames.GIN.value:
            model = GIN(config=model_config)
        else:
            raise ValueError(f'Model {model_name} not supported. Choose one of following: [GCN, GAT, GIN].')

        return model

    def _process_batch(self, batch: Data):
        batch = batch.to(self.device)
        return batch, batch.y
