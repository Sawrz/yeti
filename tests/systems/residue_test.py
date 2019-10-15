import unittest

import numpy as np

from test_utils.blueprints import BlueprintTestCase, BlueprintExceptionsTestCase


class ResidueTestCase(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Atom, Residue

        self.atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A',
                            xyz_trajectory=np.arange(6).reshape((2, 3)))
        self.atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                            xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        self.residue = Residue(subsystem_index=0, structure_file_index=1, name='test')


class TestStandardMethods(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Residue

        self.residue = Residue(subsystem_index=0, structure_file_index=1, name='test')

    def test_init(self):
        from yeti.systems.building_blocks import ResidueException

        self.assertEqual(self.residue.ensure_data_type.exception_class, ResidueException)
        self.assertListEqual(self.residue.atoms, [])
        self.assertListEqual(self.residue.sequence, [])
        self.assertEqual(self.residue.name, 'test')
        self.assertEqual(self.residue.structure_file_index, 1)
        self.assertEqual(self.residue.subsystem_index, 0)
        self.assertEqual(self.residue.number_of_atoms, 0)

    def test_str(self):
        self.assertEqual(str(self.residue), 'test0')


class TestAtomInteractionMethods(ResidueTestCase):
    def test_add_atom(self):
        self.residue.add_atom(atom=self.atom_01)

        self.assertListEqual(self.residue.atoms, [self.atom_01])
        self.assertListEqual(self.residue.sequence, ['A'])
        self.assertEqual(self.residue.number_of_atoms, 1)

    def test_add_atom_two_atoms(self):
        self.residue.add_atom(atom=self.atom_01)
        self.residue.add_atom(atom=self.atom_02)

        self.assertListEqual(self.residue.atoms, [self.atom_01, self.atom_02])
        self.assertListEqual(self.residue.sequence, ['A', 'B'])
        self.assertEqual(self.residue.number_of_atoms, 2)

    def test_finalize_single_atom(self):
        self.residue.add_atom(atom=self.atom_01)
        self.residue.finalize()

        self.assertTupleEqual(self.residue.atoms, (self.atom_01,))
        self.assertTupleEqual(self.residue.sequence, ('A',))

    def test_finalize_two_atoms(self):
        self.residue.add_atom(atom=self.atom_01)
        self.residue.add_atom(atom=self.atom_02)
        self.residue.finalize()

        self.assertTupleEqual(self.residue.atoms, (self.atom_01, self.atom_02))
        self.assertTupleEqual(self.residue.sequence, ('A', 'B'))


class ResidueExceptionsTestCase(ResidueTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import ResidueException

        super(ResidueExceptionsTestCase, self).setUp()

        self.exception = ResidueException


class TestStandardMethodExceptions(ResidueExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Residue

        super(TestStandardMethodExceptions, self).setUp()
        self.residue = Residue

    def test_init_subsystem_index_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.residue(subsystem_index='0', structure_file_index=1, name='test')

        desired_msg = self.create_data_type_exception_messages(parameter_name='subsystem_index',
                                                               data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_structure_file_index_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.residue(subsystem_index=0, structure_file_index=1.0, name='test')

        desired_msg = self.create_data_type_exception_messages(parameter_name='structure_file_index',
                                                               data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_name_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.residue(subsystem_index=0, structure_file_index=1, name=[])

        desired_msg = self.create_data_type_exception_messages(parameter_name='name',
                                                               data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))


class TestAtomInteractionExceptions(ResidueExceptionsTestCase):
    def test_add_atom_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.residue.add_atom(atom=42)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom',
                                                               data_type_name='Atom')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
