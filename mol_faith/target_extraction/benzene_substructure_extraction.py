from pandas import DataFrame
from rdkit import Chem
from rdkit.Chem import Mol
from typing import Optional

from mol_faith.model.configs.dnn_configs import DNNConfig
from mol_faith.target_extraction.shared import BaseExtractor


class BenzeneSubstructureExtractor(BaseExtractor):
    def __init__(self, config: Optional[DNNConfig] = None):
        super().__init__(config)

    @property
    def name(self) -> str:
        return 'Non-fused Benzene'

    @staticmethod
    def find_non_fused_benzene_rings(molecule: Mol) -> list[list[int]]:
        benzene_pattern = Chem.MolFromSmarts('c1ccccc1')

        ring_info = molecule.GetRingInfo()

        all_rings = [set(ring) for ring in ring_info.AtomRings()]

        # Get benzene rings with original atom order
        benzene_rings = [list(match) for match in molecule.GetSubstructMatches(benzene_pattern)]

        fused_atoms = set()
        for i, ring1 in enumerate(all_rings):
            for j in range(i + 1, len(all_rings)):
                ring2 = all_rings[j]
                if ring1 & ring2:
                    fused_atoms.update(ring1)
                    fused_atoms.update(ring2)

        # Keep only benzene rings that do NOT contain fused atoms
        non_fused_benzene_rings = [ring for ring in benzene_rings if not set(ring) & fused_atoms]

        return non_fused_benzene_rings

    def get_atom_and_bond_indices(self, molecule: Mol) -> tuple[list[list[int]], list[list[int]]]:
        benzene_rings = self.find_non_fused_benzene_rings(molecule)

        non_fused_benzene_bonds = []
        for ring in benzene_rings:
            bond_indices = [molecule.GetBondBetweenAtoms(a1, a2).GetIdx()
                            for a1, a2 in zip(ring, ring[1:] + [ring[0]])]  # Circular ring connections
            non_fused_benzene_bonds.append(bond_indices)

        non_fused_benzene_atoms = [list(ring) for ring in benzene_rings]

        return non_fused_benzene_atoms, non_fused_benzene_bonds

    def get_substructure_count(self, smiles):
        molecule = Chem.MolFromSmiles(smiles)
        benzene_rings = self.find_non_fused_benzene_rings(molecule=molecule)
        return sum(1 for ring in benzene_rings)

    def filter_data(self, data: DataFrame) -> DataFrame:
        data['substructure_count'] = data['SMILES'].apply(self.get_substructure_count)
        return data.loc[data['substructure_count'] > 0].drop(columns=['substructure_count'])
