from collections import defaultdict

import numpy as np
import re
from abc import abstractmethod
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import Mol
from typing import Optional

from mol_faith.data.data_managers.smiles_data_manager import SMILESTokenizer
from mol_faith.explanations.explainer.utils import Attributions
from mol_faith.model.configs.dnn_configs import DNNConfig


@dataclass
class FilteredAttributions:
    target_attributions: Optional[Attributions]
    complete_attributions: Attributions
    target_indices: list[int]
    has_targets: bool
    n_target_atoms: int


class BaseExtractor:
    def __init__(self, config: Optional[DNNConfig] = None):
        self.config = config

    @abstractmethod
    def get_atom_and_bond_indices(self, molecule: Mol):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @staticmethod
    def _get_target_attributions(attributions: Attributions, target_atom_indices: list[int]) -> Attributions:
        target_attributions = Attributions()
        target_attributions.atom_attributions = attributions.atom_attributions[target_atom_indices]
        return target_attributions

    def _filter_sequence_attributions(self, molecule: Mol,
                                      attributions: Attributions,
                                      target_atom_indices: list[int]) -> tuple[Attributions, Attributions]:
        target_attributions = Attributions()
        complete_attributions = Attributions()

        if self.config is None:
            raise ValueError("Config required for sequence model tokenization")

        # Map atom indices to token indices and aggregate the token attributions according to the mapping
        atom_to_token_mapping = map_tokens_to_atoms(molecule, vocab_path=self.config.model.vocab_path)
        atom_level_attributions = aggregate_token_attributions_to_atoms(attributions.atom_attributions,
                                                                        atom_to_token_mapping)

        target_attributions.atom_attributions = atom_level_attributions[target_atom_indices]
        complete_attributions.atom_attributions = atom_level_attributions
        return target_attributions, complete_attributions

    def filter_attributions(self, molecule: Mol, attributions: Attributions) -> FilteredAttributions:
        target_atom_indices, _ = self.get_atom_and_bond_indices(molecule=molecule)

        if len(target_atom_indices) == 0:
            return FilteredAttributions(
                target_attributions=None,
                complete_attributions=attributions,
                target_indices=[],
                has_targets=False,
                n_target_atoms=0
            )

        if self.name == 'Non-fused Benzene':
            # Flatten nested list of atom indices (for each ring we get the atom indices in a separate list)
            target_atom_indices = [idx for sublist in target_atom_indices for idx in sublist]

        if attributions.atom_attributions is None:
            raise ValueError("Attribution object must contain atom_attributions")

        target_attributions = self._get_target_attributions(attributions=attributions,
                                                            target_atom_indices=target_atom_indices)
        return FilteredAttributions(
            target_attributions=target_attributions,  # Shape [n_target_atoms]
            complete_attributions=attributions,  # Shape [n_atoms]
            target_indices=target_atom_indices,
            has_targets=True,
            n_target_atoms=len(target_atom_indices)
        )


def is_ring_closure_token(token):
    """Check if token is any type of ring closure"""
    # Simple: 1, 2, 9
    if re.match(r'^\d+$', token):
        return True
    # Extended: %10, %11, %99
    if re.match(r'^%\d{2}$', token):
        return True
    return False


def map_tokens_to_atoms(mol: Mol, vocab_path: str) -> dict[int, list[int]]:
    """
    Map SMILES tokens to atoms, considering ONLY atomic symbol tokens.

    This approach maps only the actual element symbols (C, N, O, c, n, o, Br, Cl, etc.)
    and ignores all other tokens including charges, hydrogens, ring numbers, bonds, brackets, etc.

    Args:
        mol: RDKit molecule object
        vocab_path: Path to tokenizer vocabulary

    Returns:
        atom_to_token_mapping: Dict[atom_idx] -> List[token_indices] (atomic symbols only)
    """
    tokenizer = SMILESTokenizer.load_vocab(vocab_path)

    # As we work with canonicalized smiles, we can safely create the smiles from the mol object without changing the
    # input SMILES
    smiles = Chem.MolToSmiles(mol)

    atom_order = mol.GetPropsAsDict(includePrivate=True, includeComputed=True).get('_smilesAtomOutputOrder')
    atom_order = list(atom_order)

    tokenized_smiles = tokenizer.tokenize(smiles=smiles)

    atom_to_token_mapping = defaultdict(list)
    atom_count = 0
    num_atoms = mol.GetNumAtoms()
    in_brackets = False
    for token_idx, token in enumerate(tokenized_smiles):
        # Track if we're inside brackets
        if token == '[':
            in_brackets = True
            continue
        elif token == ']':
            in_brackets = False
            continue

        if re.fullmatch(tokenizer._ELEMENTS_STR, token):
            # Skip explicit hydrogens in brackets
            if in_brackets and token == 'H':
                continue  # This is an explicit hydrogen, not a separate atom

            # Check bounds before accessing atom_order
            if atom_count >= num_atoms:
                raise ValueError(f"Found more heavy atom tokens than atoms in molecule. "
                                 f"Token: '{token}' at index {token_idx}, "
                                 f"but molecule only has {num_atoms} atoms. "
                                 f"SMILES: {smiles}, Tokens: {tokenized_smiles}")

            # Map element symbol to atom
            atom_idx = atom_order[atom_count]
            atom_count += 1

            atom_to_token_mapping[atom_idx].append(token_idx)

        # Validation: ensure we've mapped all atoms
    if atom_count != num_atoms:
        raise ValueError(f'Atom count mismatch: found {atom_count} heavy atom tokens '
                         f'but molecule has {num_atoms} atoms. '
                         f'SMILES: {smiles}, Tokens: {tokenized_smiles}')

    return dict(atom_to_token_mapping)


def aggregate_token_attributions_to_atoms(token_attributions: np.ndarray,
                                          atom_to_token_mapping: dict[int, list[int]]) -> np.ndarray:
    # Even though the token_attributions also includes padding tokens, the atom_to_token_mapping only considers the
    # actual molecule token sequence. We only have padding at the end of the sequence. Thus, the
    # indices in the atom_to_token_mapping is still valid (e.g. index 5 of the mapping maps to the actual molecule
    # token and not the padding token
    n_atoms = len(atom_to_token_mapping)
    atom_attributions = np.zeros(n_atoms)

    for atom_idx, token_indices in atom_to_token_mapping.items():
        atom_attributions[atom_idx] = np.sum(np.abs(token_attributions[token_idx]) for token_idx in token_indices)

    return atom_attributions
