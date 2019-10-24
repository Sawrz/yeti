import unittest

from tests.blueprints_test import BlueprintExceptionsTestCase
from tests.systems.four_atoms_plus_molecule_test import FourAtomsPlusMoleculeTestCase, TestFourAtomsPlusStandardMethods


class BioMoleculeTestCase(FourAtomsPlusMoleculeTestCase):
    def setUp(self) -> None:
        from yeti.systems.molecules.molecules import BioMolecule

        super(BioMoleculeTestCase, self).setUp()

        self.molecule = BioMolecule(residues=self.residues, molecule_name=self.molecule_name,
                                    box_information=self.box_information,
                                    simulation_information=self.simulation_information, periodic=True,
                                    hydrogen_bond_information=self.hydrogen_bond_information)


class TestBioStandardMethods(BioMoleculeTestCase, TestFourAtomsPlusStandardMethods):
    def test_init(self):
        from yeti.systems.molecules.molecules import MoleculeException

        super(TestBioStandardMethods, self).test_init()

        self.assertIsNone(self.molecule.dictionary)
        self.assertEqual(self.molecule.ensure_data_type.exception_class, MoleculeException)


class TestIdMethods(BioMoleculeTestCase):
    def test_get_atom_id(self):
        res = self.molecule.__get_atom_id__(name='B', residue_id=0)
        self.assertTupleEqual((0, 1), res)

    def test_get_atom_ids(self):
        res = self.molecule.__get_atom_ids__(atom_names=('A', 'C'), residue_ids=(0, 1))
        self.assertTupleEqual(res, ((0, 0), (1, 0)))


class BioMoleculeExceptionsTestCase(BioMoleculeTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.systems.molecules.molecules import MoleculeException, BioMoleculeException

        super(BioMoleculeExceptionsTestCase, self).setUp()

        self.molecule_exception = MoleculeException
        self.bio_exception = BioMoleculeException


class TestIdMethodsExceptions(BioMoleculeExceptionsTestCase):
    def test_get_atom_id_indistinguishable_atoms(self):
        self.molecule.residues[0].sequence = ('B', 'B')

        with self.assertRaises(self.bio_exception) as context:
            self.molecule.__get_atom_id__(name='B', residue_id=0)

        desired_msg = 'Atom names are not distinguishable. Check your naming or contact the developer.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_id_non_existing_atom(self):
        with self.assertRaises(self.bio_exception) as context:
            self.molecule.__get_atom_id__(name='C', residue_id=0)

        desired_msg = 'Atom does not exist in this residue.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_id_name_wrong_data_type(self):
        with self.assertRaises(self.molecule_exception) as context:
            self.molecule.__get_atom_id__(name=3, residue_id=0)

        desired_msg = self.create_data_type_exception_messages(parameter_name='name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_id_residue_id_wrong_data_type(self):
        with self.assertRaises(self.molecule_exception) as context:
            self.molecule.__get_atom_id__(name='C', residue_id=0.5)

        desired_msg = self.create_data_type_exception_messages(parameter_name='residue_id', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_ids_atom_names_wrong_data_type(self):
        with self.assertRaises(self.molecule_exception) as context:
            self.molecule.__get_atom_ids__(atom_names=['A', 'C'], residue_ids=(0, 1))

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_names', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_ids_atom_residue_ids_wrong_data_type(self):
        with self.assertRaises(self.molecule_exception) as context:
            self.molecule.__get_atom_ids__(atom_names=('A', 'C'), residue_ids=[0, 1])

        desired_msg = self.create_data_type_exception_messages(parameter_name='residue_ids', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    # TODO: checks for inner tuple wrong data types
    # TODO: checks for dimensions


if __name__ == '__main__':
    unittest.main()
