from dataclasses import dataclass
from typing import Optional
import numpy as np


def normalize_attributions(attributions: np.ndarray) -> np.ndarray:
    sum_attributions = np.sum(np.abs(attributions))
    normalized_attributions = attributions / sum_attributions
    return normalized_attributions


@dataclass
class Attributions:
    atom_attributions: Optional[np.array] = None
    bond_attributions: Optional[np.array] = None