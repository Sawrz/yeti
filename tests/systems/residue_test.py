import unittest

import numpy as np

from test_utils.test_utils import create_data_type_exception_messages


class ResidueTest(unittest.TestCase):
    def test_init(self):
        from yeti.systems.building_blocks import Residue

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')

        self.assertListEqual(residue.atoms, [])
        self.assertListEqual(residue.sequence, [])
        self.assertEqual(residue.name, 'test')
        self.assertEqual(residue.structure_file_index, 1)
        self.assertEqual(residue.subsystem_index, 0)
        self.assertEqual(residue.number_of_atoms, 0)

    def test_init_subsystem_index_wrong_data_type(self):
        from yeti.systems.building_blocks import Residue, ResidueException

        with self.assertRaises(ResidueException) as context:
            Residue(subsystem_index='0', structure_file_index=1, name='test')

        desired_msg = create_data_type_exception_messages(parameter_name='subsystem_index',
                                                          data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_structure_file_index_wrong_data_type(self):
        from yeti.systems.building_blocks import Residue, ResidueException

        with self.assertRaises(ResidueException) as context:
            Residue(subsystem_index=0, structure_file_index=1.0, name='test')

        desired_msg = create_data_type_exception_messages(parameter_name='structure_file_index',
                                                          data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_name_wrong_data_type(self):
        from yeti.systems.building_blocks import Residue, ResidueException

        with self.assertRaises(ResidueException) as context:
            Residue(subsystem_index=0, structure_file_index=1, name=[])

        desired_msg = create_data_type_exception_messages(parameter_name='name',
                                                          data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_str(self):
        from yeti.systems.building_blocks import Residue

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')

        self.assertEqual(str(residue), 'test0')

    def test_add_atom(self):
        from yeti.systems.building_blocks import Atom, Residue

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')
        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        residue.add_atom(atom=atom)

        self.assertListEqual(residue.atoms, [atom])
        self.assertListEqual(residue.sequence, ['A'])
        self.assertEqual(residue.number_of_atoms, 1)

    def test_add_atom_two_atoms(self):
        from yeti.systems.building_blocks import Atom, Residue

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')
        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        residue.add_atom(atom=atom_01)
        residue.add_atom(atom=atom_02)

        self.assertListEqual(residue.atoms, [atom_01, atom_02])
        self.assertListEqual(residue.sequence, ['A', 'B'])
        self.assertEqual(residue.number_of_atoms, 2)

    def test_add_atom_wrong_data_type(self):
        from yeti.systems.building_blocks import Residue, ResidueException

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')

        with self.assertRaises(ResidueException) as context:
            residue.add_atom(atom=42)

        desired_msg = create_data_type_exception_messages(parameter_name='atom',
                                                          data_type_name='Atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_finalize_single_atom(self):
        from yeti.systems.building_blocks import Atom, Residue

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')
        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        residue.add_atom(atom=atom)
        residue.finalize()

        self.assertTupleEqual(residue.atoms, (atom,))
        self.assertTupleEqual(residue.sequence, ('A',))

    def test_finalize_two_atoms(self):
        from yeti.systems.building_blocks import Atom, Residue

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')
        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        residue.add_atom(atom=atom_01)
        residue.add_atom(atom=atom_02)
        residue.finalize()

        self.assertTupleEqual(residue.atoms, (atom_01, atom_02))
        self.assertTupleEqual(residue.sequence, ('A', 'B'))


if __name__ == '__main__':
    unittest.main()
