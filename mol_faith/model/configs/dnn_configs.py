import enum
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

from mol_faith.data.data_managers.graph_data_manager import SupportedFeatureProviders
from mol_faith.model.configs.utils import (
    TuningOptions,
    check_if_configs_are_set,
    read_config_file,
    wrapped_hyperparams,
)

type GCNConfig = BaseGNNConfig

type DNNModelConfig = GCNConfig | GATConfig | GINConfig | CNNConfig | TransformerConfig


class DNNModelNames(enum.Enum):
    GCN: str = "GCN"
    GAT: str = "GAT"
    GIN: str = "GIN"
    CNN: str = "CNN"
    Transformer: str = "Transformer"


@wrapped_hyperparams
class TrainingConfig:
    dataset_folder_path: str
    output_dir_path: str
    wandb_group_name: str
    wandb_project_name: str
    target_column_name: str
    num_epochs: int
    val_frequency: int
    keep_n_checkpoints: int
    model_name: str
    batch_size: int | TuningOptions
    lr: float | TuningOptions
    weight_decay: float | TuningOptions
    initial_warm_up_epochs: int | TuningOptions
    finetune_beta: Optional[float] = None  # The beta value of the finetuning with masking
    early_stopping_patience: Optional[int] = None  # Set this value to enable early stopping
    early_stopping_delta: Optional[float] = None  # Set the minimum increase of the validation metric
    moving_avg_window: Optional[float] = None  # Window size for moving average
    t_mult: int = 2  # The multiplier for the cosine annealing
    seed: int = 45  # The seed for all relevant random processes during training
    wandb_run_name: str = ''  # Will be automatically set or it can be set manually
    wandb_tags: List[str] = field(default_factory=list)  # Can be set manually. HP tuning will set this as well
    wandb_group_prefix: str = ''


@wrapped_hyperparams(kw_only=True)
class CNNConfig:
    embedding_dim: int | TuningOptions  # The embedding dimension for the features
    kernel_sizes: List[int]  # List of kernel sizes
    c_hidden: int | TuningOptions  # Hidden channel size inside the CNN layer block
    dp_rate: int | TuningOptions  # Dropout Rate inside the CNN layer block
    c_hidden_head: int | TuningOptions  # The hidden dimension in the prediction head
    dp_rate_head: float | TuningOptions  # The dropout rate in the prediction head
    vocab_path: str  # The path to the vocab file
    max_seq_len: int  # The maximum sequence length (fixed for a dataset)
    use_se_attention: bool = False


@wrapped_hyperparams(kw_only=True)
class BaseGNNConfig:
    embedding_dim: int | TuningOptions  # The embedding dimension for the features
    c_hidden: int | TuningOptions  # The hidden dimension in the GNN encoder
    num_layers: int | TuningOptions  # The num of hidden layers (number of message passing) in the GNN encoder
    c_hidden_head: int | TuningOptions  # The hidden dimension in the prediction head
    dp_rate_head: float | TuningOptions  # The dropout rate in the prediction head
    feature_provider: str = SupportedFeatureProviders.PYTORCH_GEOMETRIC.value  # Default features to use
    norm_func: Optional[str | TuningOptions] = None  # The norm function used in the GNN encoder
    dp_rate: Optional[float | TuningOptions] = None  # The dropout rate in the GNN encoder
    c_in: int = 0  # Will be set automatically. Depends on the number of node features
    scale_features: bool = False  # Whether the node features should be normalized via min-max scaling (deprecated!)


@wrapped_hyperparams
class GATConfig(BaseGNNConfig):
    heads: int | TuningOptions
    concat: bool = False
    negative_slope: float = 0.2
    dropout_gat_conv: float = 0.0


@wrapped_hyperparams
class GINConfig(BaseGNNConfig):
    mlp_act: str = "relu"
    mlp_act_first: bool = False
    mlp_norm: Optional[bool] = None
    mlp_norm_kwargs: Optional[Dict[str, Any]] = None


@wrapped_hyperparams(kw_only=True)
class TransformerConfig:
    d_model: int | TuningOptions  # The hidden dimension of the input. It is also the size of the embeddings
    heads: int | TuningOptions  # Number of heads. Used for HP tuning to get the d_model parameter
    # The dimension of the feed-forward net in each transformer layer. In HP tuning it will be set based on the d_model
    # parameter (dim_feedforward = 4 * d_model). In that case we do not need to set it.
    dim_feedforward: int | TuningOptions
    dp_rate: float | TuningOptions  # The dropout value in the feed-forward net
    num_layers: int | TuningOptions  # The number of encoding layers
    c_hidden_head: int | TuningOptions  # The hidden dimension in the prediction head
    dp_rate_head: float | TuningOptions  # The dropout rate in the prediction head
    max_seq_len: int  # The maximum sequence length (fixed for a dataset)
    vocab_path: str  # The path to the vocab file


@dataclass
class DNNConfig:
    training: TrainingConfig
    model: DNNModelConfig

    def get_config_fields(self) -> List[str]:
        all_fields = []
        for field in fields(self):
            all_fields.append(field.name)
        return all_fields


def get_config(config_file_path: str) -> DNNConfig:
    config_all = read_config_file(config_file_path)

    check_if_configs_are_set(config_all=config_all)

    try:
        training_config = TrainingConfig(**config_all['training'])
        model_config = get_dnn_config(model_name=training_config.model_name, model_params=config_all['model'])
    except TypeError as error:
        # Todo: Better error handling when creating config
        raise TypeError(f'Could not create the configs because of following error: \n {error}')
    # Todo: Make sure that the values are floats when setting them. Maybe after init go through the fields and cast it
    #  to float or int if we expect an int or float

    return DNNConfig(training=training_config, model=model_config)


def get_dnn_config(model_name: str, model_params: Dict[
    str, Any]) -> BaseGNNConfig | GATConfig | GINConfig | CNNConfig | TransformerConfig:
    if model_name == DNNModelNames.GCN.value:
        return BaseGNNConfig(**model_params)
    elif model_name == DNNModelNames.GAT.value:
        return GATConfig(**model_params)
    elif model_name == DNNModelNames.GIN.value:
        return GINConfig(**model_params)
    elif model_name == DNNModelNames.CNN.value:
        return CNNConfig(**model_params)
    elif model_name == DNNModelNames.Transformer.value:
        return TransformerConfig(**model_params)
    else:
        raise ValueError(
            f'Model {model_name} not supported. Choose one of following: {[e.value for e in DNNModelNames]}.')
