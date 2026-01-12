import enum
from typing import Dict, TypedDict, Optional


class ExplainerTypes(str, enum.Enum):
    GNN_EXPLAINER = 'gnn_explainer'


class ExplanationMethodArgs(TypedDict, total=False):
    explainer_args: Optional[Dict]  # Arguments for explainer initialization
    attribute_args: Optional[Dict]  # Arguments passed to attribute() call


class ExplanationMethod(str, enum.Enum):
    # Captum methods
    INTEGRATED_GRADIENTS = 'integrated_gradients'
    KERNEL_SHAP = 'kernel_shap'
    LIME = 'lime'
    GRAD_CAM = 'grad_cam'
    GUIDED_GRAD_CAM = 'guided_grad_cam'
    SHAP_SAMPLING = 'shap_sampling' # ShapleyValueSampling
    
    # PyG methods (Currently only GNN specfic methods)
    GNN_EXPLAINER = 'gnn_explainer'
    GRAPH_MASK = 'graph_mask'

    # Random explanations
    RANDOM_EXPLANATION = 'random_explanation'