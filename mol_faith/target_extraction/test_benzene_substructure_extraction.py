import unittest

import numpy as np
from rdkit import Chem

from mol_faith.explanations.explainer.utils import Attributions
from mol_faith.target_extraction.benzene_substructure_extraction import BenzeneSubstructureExtractor


class TestBenzeneSubstructureExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = BenzeneSubstructureExtractor()

    def test_find_non_fused_benzene_rings(self):
        smiles = 'CCO'  # No benzene ring
        mol = Chem.MolFromSmiles(smiles)
        benzene_rings = self.extractor.find_non_fused_benzene_rings(mol)
        self.assertEqual(len(benzene_rings), 0)

        smiles = 'c1ccccc1'  # 1 non-fused benzene ring
        mol = Chem.MolFromSmiles(smiles)
        benzene_rings = self.extractor.find_non_fused_benzene_rings(mol)
        self.assertEqual(len(benzene_rings), 1)

        smiles = 'c1ccc(cc1)c2cc3ccccc3cc2'  # 1 non-fused benzene ring and 1 fused system
        mol = Chem.MolFromSmiles(smiles)
        benzene_rings = self.extractor.find_non_fused_benzene_rings(mol)
        self.assertEqual(len(benzene_rings), 1)

        smiles = 'c1ccc(cc1)c2ccccc2'  # 2 non-fused benzene rings
        mol = Chem.MolFromSmiles(smiles)
        benzene_rings = self.extractor.find_non_fused_benzene_rings(mol)
        self.assertEqual(len(benzene_rings), 2)

        smiles = 'c1ccc2ccccc2c1'  # Only 1 fused system
        mol = Chem.MolFromSmiles(smiles)
        benzene_rings = self.extractor.find_non_fused_benzene_rings(mol)
        self.assertEqual(len(benzene_rings), 0)

        smiles = 'c1ccc(cc1)c2cc3ccccc3c(c2)-c4ccccc4'  # 2 Non-fused benzene rings and 1 fused ones
        mol = Chem.MolFromSmiles(smiles)
        benzene_rings = self.extractor.find_non_fused_benzene_rings(mol)
        self.assertEqual(len(benzene_rings), 2)

    def test_get_atom_and_bond_indices(self):
        smiles = 'c1ccc(cc1)c2cc3ccccc3cc2'  # 1 non-fused benzene ring and 1 fused system
        mol = Chem.MolFromSmiles(smiles)

        atom_indices, bond_indices = self.extractor.get_atom_and_bond_indices(molecule=mol)

        self.assertEqual(len(atom_indices), 1)
        self.assertEqual(len(atom_indices[0]), 6)

        self.assertEqual(len(bond_indices), 1)
        self.assertEqual(len(bond_indices[0]), 6)

    def test_filter_atom_attributions(self):
        smiles = 'c1ccc(cc1)c2cc3ccccc3cc2'  # 1 non-fused benzene ring and 1 fused system
        mol = Chem.MolFromSmiles(smiles)

        num_atoms = mol.GetNumAtoms()
        atom_attr = np.random.rand(num_atoms)
        benzene_atom_indices = [0, 1, 2, 3, 4, 5]

        attributions = Attributions(atom_attributions=atom_attr)

        benzene_attributions, other_attributions = self.extractor.filter_attributions(molecule=mol,
                                                                                      attributions=attributions)

        self.assertEqual(len(benzene_attributions.atom_attributions), 6)
        self.assertListEqual(benzene_attributions.atom_attributions.tolist(), atom_attr[benzene_atom_indices].tolist())

        self.assertEqual(len(other_attributions.atom_attributions), num_atoms - len(benzene_atom_indices))
        self.assertListEqual(other_attributions.atom_attributions.tolist(),
                             np.delete(atom_attr, benzene_atom_indices).tolist())

    def test_filter_bond_attributions(self):
        smiles = 'c1ccc(cc1)c2cc3ccccc3cc2'  # 1 non-fused benzene ring and 1 fused system
        mol = Chem.MolFromSmiles(smiles)

        num_bonds = mol.GetNumBonds()
        bond_attr = np.random.rand(num_bonds)
        benzene_bond_indices = [0, 1, 2, 3, 4, 15]

        attributions = Attributions(bond_attributions=bond_attr)

        benzene_attributions, other_attributions = self.extractor.filter_attributions(molecule=mol,
                                                                                      attributions=attributions)

        self.assertEqual(len(benzene_attributions.bond_attributions), 6)
        self.assertListEqual(benzene_attributions.bond_attributions.tolist(), bond_attr[benzene_bond_indices].tolist())

        self.assertEqual(len(other_attributions.bond_attributions), num_bonds - len(benzene_bond_indices))
        self.assertListEqual(other_attributions.bond_attributions.tolist(),
                             np.delete(bond_attr, benzene_bond_indices).tolist())

    def test_no_isolated_benzene(self):
        smiles = 'CCO'  # No non-fused benzene ring
        mol = Chem.MolFromSmiles(smiles)

        atom_attr = np.random.rand(3)
        bond_attr = np.random.rand(2)

        attributions = Attributions(atom_attributions=atom_attr, bond_attributions=bond_attr)

        self.assertRaises(ValueError, self.extractor.filter_attributions, molecule=mol, attributions=attributions)


if __name__ == '__main__':
    unittest.main()
