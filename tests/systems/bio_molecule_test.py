import unittest

from tests.systems.four_atoms_plus_molecule import FourAtomsPlusInitTest
from tests.systems.two_atoms_molecule_test import MoleculesTest


class BioMoleculeInitTest(FourAtomsPlusInitTest):

    def setUp(self) -> None:
        from yeti.systems.molecules.molecules import BioMolecule

        super(BioMoleculeInitTest, self).setUp()
        self.molecule = BioMolecule

    def test_init(self):
        super(BioMoleculeInitTest, self).test_init()
        self.assertIsNone(self.working_molecule.dictionary)
        self.assertEqual(self.working_molecule.ensure_data_type.exception_class, self.exception)


class GetIdTest(MoleculesTest):
    def setUp(self) -> None:
        from yeti.systems.molecules.molecules import BioMolecule, BioMoleculeException

        residues, box_information = super(GetIdTest, self).setUp()
        simulation_information = dict(number_of_frames=3)
        hydrogen_bond_information = dict(distance_cutoff=0.25, angle_cutoff=2.0)

        self.molecule = BioMolecule(residues=residues, molecule_name='test', box_information=box_information,
                                    simulation_information=simulation_information,
                                    hydrogen_bond_information=hydrogen_bond_information)
        self.internal_exception = BioMoleculeException

    def test_get_atom_id(self):
        res = self.molecule.__get_atom_id__(name='B', residue_id=0)
        self.assertTupleEqual((0, 1), res)

    def test_get_atom_id_indistinguishable_atoms(self):
        self.molecule.residues[0].sequence = ('B', 'B')

        with self.assertRaises(self.internal_exception) as context:
            self.molecule.__get_atom_id__(name='B', residue_id=0)

        desired_msg = 'Atom names are not distinguishable. Check your naming or contact the developer.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_id_non_existing_atom(self):
        with self.assertRaises(self.internal_exception) as context:
            self.molecule.__get_atom_id__(name='C', residue_id=0)

        desired_msg = 'Atom does not exist in this residue.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_id_name_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.__get_atom_id__(name=3, residue_id=0)

        desired_msg = self.create_data_type_exception_messages(parameter_name='name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_id_residue_id_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.__get_atom_id__(name='C', residue_id=0.5)

        desired_msg = self.create_data_type_exception_messages(parameter_name='residue_id', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_ids(self):
        res = self.molecule.__get_atom_ids__(atom_names=('A', 'C'), residue_ids=(0, 1))
        self.assertTupleEqual(res, ((0, 0), (1, 0)))

    def test_get_atom_ids_atom_names_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.__get_atom_ids__(atom_names=['A', 'C'], residue_ids=(0, 1))

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_names', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_ids_atom_residue_ids_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.__get_atom_ids__(atom_names=('A', 'C'), residue_ids=[0, 1])

        desired_msg = self.create_data_type_exception_messages(parameter_name='residue_ids', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    # TODO: checks for inner tuple wrong data types
    # TODO: checks for dimensions


if __name__ == '__main__':
    unittest.main()
