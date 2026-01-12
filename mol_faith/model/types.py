import enum

from mol_faith.model.configs.dnn_configs import DNNConfig, TrainingConfig as DNNTrainingConfig, \
    DNNModelConfig

class ModelTypes(str, enum.Enum):
    CNN = 'cnn'
    GNN = 'gnn'
    TRANSFORMER = 'transformer'

SUPPORTED_MODEL_TYPES = [model_type.value for model_type in ModelTypes]

SUPPORTED_DNN_MODEL_TYPES = [model_type.value for model_type in ModelTypes]

type TrainingConfig = DNNTrainingConfig

type ModelConfig = DNNModelConfig

type Config = DNNConfig
