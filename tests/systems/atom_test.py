import unittest

import numpy as np
import numpy.testing as npt

from yeti.systems.residue import Residue

residue = Residue(subsystem_index=0, structure_file_index=1, name='helper')


def create_data_type_exception_messages(parameter_name, data_type_name):
    return 'Wrong data type for parameter "{name}". Desired type is {data_type}'.format(name=parameter_name,
                                                                                        data_type=data_type_name)


class AtomTest(unittest.TestCase):
    def test_init(self):
        from yeti.systems.atom import Atom

        xyz_trajectory = np.arange(6).reshape((3, 2))

        atom = Atom(name='test', subsystem_index=16, structure_file_index=42, residue=residue,
                    xyz_trajectory=xyz_trajectory)

        self.assertEqual(atom.name, 'test')
        self.assertEqual(atom.subsystem_index, 16)
        self.assertEqual(atom.structure_file_index, 42)
        self.assertEqual(atom.residue, residue)
        npt.assert_equal(atom.xyz_trajectory, xyz_trajectory)

        self.assertIsNone(atom.element)
        self.assertTupleEqual(atom.covalent_bond_partners, ())
        self.assertFalse(atom.is_donor_atom)
        self.assertFalse(atom.is_acceptor)
        self.assertEqual(atom.donor_slots, 0)
        self.assertEqual(atom.acceptor_slots, 0)
        self.assertIsNone(atom.hydrogen_bond_partners)

    def test_init_name_wrong_data_type(self):
        from yeti.systems.atom import Atom, AtomException

        with self.assertRaises(AtomException) as context:
            Atom(name=12, subsystem_index=16, structure_file_index=42, residue=residue,
                 xyz_trajectory=np.arange(6).reshape((3, 2)))

        desired_msg = create_data_type_exception_messages(parameter_name='name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_subsystem_index_wrong_data_type(self):
        from yeti.systems.atom import Atom, AtomException

        with self.assertRaises(AtomException) as context:
            Atom(name='test', subsystem_index=16.1, structure_file_index=42, residue=residue,
                 xyz_trajectory=np.arange(6).reshape((3, 2)))

        desired_msg = create_data_type_exception_messages(parameter_name='subsystem_index', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_structure_file_wrong_data_type(self):
        from yeti.systems.atom import Atom, AtomException

        with self.assertRaises(AtomException) as context:
            Atom(name='test', subsystem_index=16, structure_file_index=42.0, residue=residue,
                 xyz_trajectory=np.arange(6).reshape((3, 2)))

        desired_msg = create_data_type_exception_messages(parameter_name='structure_file_index',
                                                          data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_residue_wrong_data_type(self):
        from yeti.systems.atom import Atom, AtomException

        with self.assertRaises(AtomException) as context:
            Atom(name='test', subsystem_index=16, structure_file_index=42, residue='residue',
                 xyz_trajectory=np.arange(6).reshape((3, 2)))

        desired_msg = create_data_type_exception_messages(parameter_name='residue', data_type_name='Residue')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_xyz_wrong_shape(self):
        from yeti.systems.atom import Atom, AtomException

        xyz_trajectory = np.arange(8).reshape((4, 2))

        with self.assertRaises(AtomException) as context:
            Atom(name='test', subsystem_index=16, structure_file_index=42, residue=residue,
                 xyz_trajectory=xyz_trajectory)

        self.assertEqual(str(context.exception), 'Wrong shape for parameter "xyz_trajectory". Desired shape: (3, None).')

    def test_init_xyz_wrong_data_type(self):
        from yeti.systems.atom import Atom, AtomException

        xyz_trajectory = list(np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            Atom(name='test', subsystem_index=16, structure_file_index=42, residue=residue,
                 xyz_trajectory=xyz_trajectory)

        desired_msg = create_data_type_exception_messages(parameter_name='xyz_trajectory', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_covalent_bond(self):
        from yeti.systems.atom import Atom

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                       xyz_trajectory=np.arange(6).reshape((3, 2)))
        atom_02 = Atom(structure_file_index=3, subsystem_index=0, name='A', residue=residue,
                       xyz_trajectory=np.arange(6, 12).reshape((3, 2)))

        atom_01.__update_covalent_bond__(atom=atom_02)

        self.assertTupleEqual(atom_01.covalent_bond_partners, (atom_02,))

    def test_update_covalent_bond_wrong_data_type(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.__update_covalent_bond__(atom=42)

        desired_msg = create_data_type_exception_messages(parameter_name='atom', data_type_name='atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_covalent_bond_exists_already(self):
        from yeti.systems.atom import Atom, AtomWarning

        atom_01 = Atom(structure_file_index=4, subsystem_index=0, name='A', residue=residue,
                       xyz_trajectory=np.arange(6).reshape((3, 2)))
        atom_02 = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                       xyz_trajectory=np.arange(6, 12).reshape((3, 2)))

        atom_01.__update_covalent_bond__(atom=atom_02)

        with self.assertRaises(AtomWarning) as context:
            atom_01.__update_covalent_bond__(atom=atom_02)

        desired_msg = 'Covalent bond already exists. Skipping...'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_covalent_bond_with_itself(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.__update_covalent_bond__(atom=atom)

        desired_msg = 'Atom can not have a covalent bond with itself.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_add_covalent_bonds(self):
        from yeti.systems.atom import Atom

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                       xyz_trajectory=np.arange(6).reshape((3, 2)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=0, name='A', residue=residue,
                       xyz_trajectory=np.arange(6, 12).reshape((3, 2)))

        atom_01.add_covalent_bond(atom=atom_02)

        self.assertTupleEqual(atom_01.covalent_bond_partners, (atom_02,))
        self.assertTupleEqual(atom_02.covalent_bond_partners, (atom_01,))

    def test_add_covalent_bonds_wrong_data_type(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.add_covalent_bond(atom=42)

        desired_msg = create_data_type_exception_messages(parameter_name='atom', data_type_name='atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partners_false(self):
        from yeti.systems.atom import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        atom.__update_hydrogen_bond_partners__(is_hydrogen_bond_active=False)
        self.assertIsNone(atom.hydrogen_bond_partners)

    def test_update_hydrogen_bond_partners_true(self):
        from yeti.systems.atom import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        atom.__update_hydrogen_bond_partners__(is_hydrogen_bond_active=True)
        self.assertDictEqual(atom.hydrogen_bond_partners, dict(subsystem=[[], []]))
        self.assertListEqual(atom.hydrogen_bond_partners['subsystem'], [[], []])

    def test_update_hydrogen_bond_partners_wrong_data_type(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.__update_hydrogen_bond_partners__(is_hydrogen_bond_active=42)

        desired_msg = create_data_type_exception_messages(parameter_name='is_hydrogen_bond_active',
                                                          data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_true(self):
        from yeti.systems.atom import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        atom.update_donor_state(is_donor_atom=True, donor_slots=2)

        self.assertTrue(atom.is_donor_atom)
        self.assertEqual(atom.donor_slots, 2)
        self.assertListEqual(atom.hydrogen_bond_partners['subsystem'], [[], []])

    def test_update_donor_state_false(self):
        from yeti.systems.atom import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        atom.update_donor_state(is_donor_atom=False, donor_slots=0)

        self.assertFalse(atom.is_donor_atom)
        self.assertEqual(atom.donor_slots, 0)
        self.assertIsNone(atom.hydrogen_bond_partners)

    def test_update_donor_state_false_but_slots(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.update_donor_state(is_donor_atom=False, donor_slots=2)

        desired_msg = 'A non-donor atom does not have any donor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_true_but_none_slots(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.update_donor_state(is_donor_atom=True, donor_slots=0)

        desired_msg = 'Donor atom need donor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_true_but_no_slots(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.update_donor_state(is_donor_atom=True, donor_slots=0)

        desired_msg = 'Donor atom need donor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_true_but_negative_slots(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.update_donor_state(is_donor_atom=True, donor_slots=-1)

        desired_msg = 'Donor atom need donor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_is_donor_atom_wrong_data_type(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.update_donor_state(is_donor_atom=42, donor_slots=2)

        desired_msg = create_data_type_exception_messages(parameter_name='is_donor_atom', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_donor_slots_wrong_data_type(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.update_donor_state(is_donor_atom=True, donor_slots=2.5)

        desired_msg = create_data_type_exception_messages(parameter_name='donor_slots', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_true(self):
        from yeti.systems.atom import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        atom.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        self.assertTrue(atom.is_acceptor)
        self.assertEqual(atom.acceptor_slots, 2)
        self.assertListEqual(atom.hydrogen_bond_partners['subsystem'], [[], []])

    def test_update_acceptor_state_false(self):
        from yeti.systems.atom import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        atom.update_acceptor_state(is_acceptor=False, acceptor_slots=0)

        self.assertFalse(atom.is_acceptor)
        self.assertEqual(atom.acceptor_slots, 0)
        self.assertIsNone(atom.hydrogen_bond_partners)

    def test_update_acceptor_state_false_but_slots(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.update_acceptor_state(is_acceptor=False, acceptor_slots=2)

        desired_msg = 'A non-acceptor atom does not have any acceptor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_true_but_none_slots(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.update_acceptor_state(is_acceptor=True, acceptor_slots=0)

        desired_msg = 'Acceptor atom need acceptor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_true_but_no_slots(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.update_acceptor_state(is_acceptor=True, acceptor_slots=0)

        desired_msg = 'Acceptor atom need acceptor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_true_but_negative_slots(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.update_acceptor_state(is_acceptor=True, acceptor_slots=-1)

        desired_msg = 'Acceptor atom need acceptor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_is_donor_atom_wrong_data_type(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.update_acceptor_state(is_acceptor=42, acceptor_slots=2)

        desired_msg = create_data_type_exception_messages(parameter_name='is_acceptor', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_donor_slots_wrong_data_type(self):
        from yeti.systems.atom import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', residue=residue,
                    xyz_trajectory=np.arange(6).reshape((3, 2)))

        with self.assertRaises(AtomException) as context:
            atom.update_acceptor_state(is_acceptor=True, acceptor_slots=2.5)

        desired_msg = create_data_type_exception_messages(parameter_name='acceptor_slots', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
