from pandas import DataFrame
from rdkit import Chem
from rdkit.Chem import Mol
from typing import Optional

from mol_faith.model.configs.dnn_configs import DNNConfig
from mol_faith.target_extraction.shared import BaseExtractor


class HBondDonorExtractor(BaseExtractor):
    def __init__(self, config: Optional[DNNConfig] = None):
        super().__init__(config)

    @property
    def name(self) -> str:
        return 'H-Bond Donor (N, O, F)'

    @staticmethod
    def find_hbond_donor_atoms(molecule: Mol) -> list[int]:
        """
        Find atoms matching the SMARTS pattern [!H0;#7,#8,#9].
        This matches nitrogen, oxygen, or fluorine atoms that have at least one hydrogen.
        Returns the atom indices of all matching atoms.
        """
        if not molecule:
            return []

        try:
            # SMARTS pattern for N, O, or F atoms with at least one hydrogen
            pattern = Chem.MolFromSmarts("[!H0;#7,#8,#9]")
            if pattern is None:
                print("Error: Could not parse SMARTS pattern")
                return []

            # Get all matches
            matches = molecule.GetSubstructMatches(pattern)

            # Extract atom indices from matches
            # For single-atom SMARTS patterns, each match is a tuple with one element
            matching_atoms = [match[0] for match in matches]

            return matching_atoms

        except Exception as e:
            print(f"Error finding hbond donors: {e}")
            return []

    def get_atom_and_bond_indices(self, molecule: Mol) -> tuple[list[int], list[int]]:
        """
        Get atom and bond indices for h-bond donor.
        Returns atoms matching the pattern and empty list for bonds.
        """
        matching_atoms = self.find_hbond_donor_atoms(molecule)

        # Return empty list for bonds as requested since these are individual atoms
        return matching_atoms, []

    def get_hbond_donor_count(self, smiles: str) -> int:
        """
        Get the number of atoms matching the hbond donor pattern.
        """
        try:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                return 0

            matching_atoms = self.find_hbond_donor_atoms(molecule)
            return len(matching_atoms)

        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return 0

    def filter_data(self, data: DataFrame) -> DataFrame:
        """
        Filter the dataset to include only molecules with h-bond donor according to the SMART patterns.
        """
        data['hbond_donor_count'] = data['SMILES'].apply(self.get_hbond_donor_count)
        return data.loc[data['hbond_donor_count'] > 0].drop(columns=['hbond_donor_count'])
