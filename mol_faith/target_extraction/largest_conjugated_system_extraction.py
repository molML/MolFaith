from collections import defaultdict

from pandas import DataFrame
from rdkit import Chem
from rdkit.Chem import Mol, ResonanceMolSupplier
from typing import Optional

from mol_faith.model.configs.dnn_configs import DNNConfig
from mol_faith.target_extraction.shared import BaseExtractor


class LargestConjugatedSystemExtractor(BaseExtractor):
    def __init__(self, config: Optional[DNNConfig] = None):
        super().__init__(config)

    @property
    def name(self) -> str:
        return 'Largest Conjugated System'

    @staticmethod
    def find_largest_conjugated_system(molecule: Mol) -> list[int]:
        """
        Find the largest conjugated system using RDKit's built-in methods.
        Returns the atom indices of the largest conjugated system.
        """
        if not molecule:
            return []

        try:
            # Create ResonanceMolSupplier to get conjugated groups
            suppl = ResonanceMolSupplier(molecule)
            num_groups = suppl.GetNumConjGrps()

            if num_groups == 0:
                return []

            # Group atoms by their conjugated group index
            conjugated_groups = defaultdict(list)
            for atom_idx in range(molecule.GetNumAtoms()):
                group_idx = suppl.GetAtomConjGrpIdx(atom_idx)
                # -1 means not conjugated. Somehow rdkit returns 4294967295 (probably overflow)
                if group_idx != 4294967295:
                    conjugated_groups[group_idx].append(atom_idx)

            # Find the group with the most atoms
            best_group = []
            max_atoms = 0
            for group_idx, atom_indices in conjugated_groups.items():
                if len(atom_indices) > max_atoms:
                    max_atoms = len(atom_indices)
                    best_group = atom_indices

            return best_group

        except Exception as e:
            print(f"Error finding conjugated system: {e}")
            return []

    def get_atom_and_bond_indices(self, molecule: Mol) -> tuple[list[int], list[int]]:
        """
        Get atom and bond indices for the largest conjugated system.
        """
        conjugated_atoms = self.find_largest_conjugated_system(molecule)

        if not conjugated_atoms:
            return [], []

        # Get bonds within the conjugated system
        conjugated_bonds = []
        atom_set = set(conjugated_atoms)

        for atom_idx in conjugated_atoms:
            atom = molecule.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in atom_set and atom_idx < neighbor_idx:  # Avoid duplicate bonds
                    bond = molecule.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                    if bond:
                        conjugated_bonds.append(bond.GetIdx())

        return conjugated_atoms, conjugated_bonds

    def get_conjugated_system_size(self, smiles: str) -> int:
        """
        Get the size (number of atoms) of the largest conjugated system in the molecule.
        """
        try:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                return 0
            conjugated_atoms = self.find_largest_conjugated_system(molecule)
            return len(conjugated_atoms)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return 0

    def filter_data(self, data: DataFrame) -> DataFrame:
        """
        Filter the dataset to include only molecules with conjugated systems.
        """
        data['conjugated_system_size'] = data['SMILES'].apply(self.get_conjugated_system_size)
        return data.loc[data['conjugated_system_size'] > 0].drop(columns=['conjugated_system_size'])
