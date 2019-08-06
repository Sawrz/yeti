import unittest

import numpy as np
import numpy.testing as npt

from test_utils.test_utils import create_data_type_exception_messages
from yeti.systems.building_blocks import Residue

residue = Residue(subsystem_index=0, structure_file_index=1, name='helper')


class AtomTest(unittest.TestCase):
    def test_init(self):
        from yeti.systems.building_blocks import Atom

        xyz_trajectory = np.arange(6).reshape((2, 3))

        atom = Atom(name='test', subsystem_index=16, structure_file_index=42, xyz_trajectory=xyz_trajectory)

        self.assertEqual(atom.name, 'test')
        self.assertEqual(atom.subsystem_index, 16)
        self.assertEqual(atom.structure_file_index, 42)
        self.assertIsNone(atom.residue)
        npt.assert_equal(atom.xyz_trajectory, xyz_trajectory)

        self.assertIsNone(atom.element)
        self.assertTupleEqual(atom.covalent_bond_partners, ())
        self.assertFalse(atom.is_donor_atom)
        self.assertFalse(atom.is_acceptor)
        self.assertEqual(atom.donor_slots, 0)
        self.assertEqual(atom.acceptor_slots, 0)
        self.assertIsNone(atom.hydrogen_bond_partners)

    def test_init_name_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        with self.assertRaises(AtomException) as context:
            Atom(name=12, subsystem_index=16, structure_file_index=42, xyz_trajectory=np.arange(6).reshape((2, 3)))

        desired_msg = create_data_type_exception_messages(parameter_name='name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_subsystem_index_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        with self.assertRaises(AtomException) as context:
            Atom(name='test', subsystem_index=16.1, structure_file_index=42,
                 xyz_trajectory=np.arange(6).reshape((2, 3)))

        desired_msg = create_data_type_exception_messages(parameter_name='subsystem_index', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_structure_file_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        with self.assertRaises(AtomException) as context:
            Atom(name='test', subsystem_index=16, structure_file_index=42.0,
                 xyz_trajectory=np.arange(6).reshape((2, 3)))

        desired_msg = create_data_type_exception_messages(parameter_name='structure_file_index',
                                                          data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_xyz_wrong_shape(self):
        from yeti.systems.building_blocks import Atom, AtomException

        xyz_trajectory = np.arange(8).reshape((4, 2))

        with self.assertRaises(AtomException) as context:
            Atom(name='test', subsystem_index=16, structure_file_index=42, xyz_trajectory=xyz_trajectory)

        self.assertEqual(str(context.exception),
                         'Wrong shape for parameter "xyz_trajectory". Desired shape: (None, 3).')

    def test_init_xyz_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        xyz_trajectory = list(np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            Atom(name='test', subsystem_index=16, structure_file_index=42, xyz_trajectory=xyz_trajectory)

        desired_msg = create_data_type_exception_messages(parameter_name='xyz_trajectory', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_set_residue(self):
        from yeti.systems.building_blocks import Atom, Residue

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        res = Residue(subsystem_index=0, structure_file_index=1, name='RES')

        atom.set_residue(residue=res)

        self.assertEqual(atom.residue, res)

    def test_set_residue_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(name='test', subsystem_index=16, structure_file_index=42,
                    xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.set_residue(residue=24)

        desired_msg = create_data_type_exception_messages(parameter_name='residue', data_type_name='Residue')
        self.assertEqual(desired_msg, str(context.exception))

    def test_set_residue_already_assigned(self):
        from yeti.systems.building_blocks import Atom, AtomWarning

        atom = Atom(name='test', subsystem_index=16, structure_file_index=42,
                    xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom.set_residue(residue=residue)

        with self.assertRaises(AtomWarning) as context:
            atom.set_residue(residue=residue)

        desired_msg = 'This atom belongs already to a residue. Changing relationship...'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_covalent_bond(self):
        from yeti.systems.building_blocks import Atom

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=3, subsystem_index=0, name='A',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        atom_01.__update_covalent_bond__(atom=atom_02)

        self.assertTupleEqual(atom_01.covalent_bond_partners, (atom_02,))

    def test_update_covalent_bond_more_atoms(self):
        from yeti.systems.building_blocks import Atom

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_03 = Atom(structure_file_index=3, subsystem_index=2, name='C',
                       xyz_trajectory=np.arange(12, 18).reshape((2, 3)))

        atom_01.__update_covalent_bond__(atom=atom_02)
        atom_01.__update_covalent_bond__(atom=atom_03)

        self.assertTupleEqual(atom_01.covalent_bond_partners, (atom_02, atom_03))

    def test_update_covalent_bond_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.__update_covalent_bond__(atom=42)

        desired_msg = create_data_type_exception_messages(parameter_name='atom', data_type_name='atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_covalent_bond_exists_already(self):
        from yeti.systems.building_blocks import Atom, AtomWarning

        atom_01 = Atom(structure_file_index=4, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=1, subsystem_index=0, name='A',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        atom_01.__update_covalent_bond__(atom=atom_02)

        with self.assertRaises(AtomWarning) as context:
            atom_01.__update_covalent_bond__(atom=atom_02)

        desired_msg = 'Covalent bond already exists. Skipping...'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_covalent_bond_with_itself(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.__update_covalent_bond__(atom=atom)

        desired_msg = 'Atom can not have a covalent bond with itself.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_add_covalent_bonds(self):
        from yeti.systems.building_blocks import Atom

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=0, name='A',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        atom_01.add_covalent_bond(atom=atom_02)

        self.assertTupleEqual(atom_01.covalent_bond_partners, (atom_02,))
        self.assertTupleEqual(atom_02.covalent_bond_partners, (atom_01,))

    def test_add_covalent_bonds_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.add_covalent_bond(atom=42)

        desired_msg = create_data_type_exception_messages(parameter_name='atom', data_type_name='atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partners_false(self):
        from yeti.systems.building_blocks import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        atom.__reset_hydrogen_bond_partners__(is_hydrogen_bond_active=False)
        self.assertIsNone(atom.hydrogen_bond_partners)

    def test_update_hydrogen_bond_partners_true(self):
        from yeti.systems.building_blocks import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        atom.__reset_hydrogen_bond_partners__(is_hydrogen_bond_active=True)
        self.assertDictEqual(atom.hydrogen_bond_partners, dict(subsystem=[[], []]))
        self.assertListEqual(atom.hydrogen_bond_partners['subsystem'], [[], []])

    def test_update_hydrogen_bond_partners_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.__reset_hydrogen_bond_partners__(is_hydrogen_bond_active=42)

        desired_msg = create_data_type_exception_messages(parameter_name='is_hydrogen_bond_active',
                                                          data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_true(self):
        from yeti.systems.building_blocks import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        atom.update_donor_state(is_donor_atom=True, donor_slots=2)

        self.assertTrue(atom.is_donor_atom)
        self.assertEqual(atom.donor_slots, 2)
        self.assertListEqual(atom.hydrogen_bond_partners['subsystem'], [[], []])

    def test_update_donor_state_false(self):
        from yeti.systems.building_blocks import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        atom.update_donor_state(is_donor_atom=False, donor_slots=0)

        self.assertFalse(atom.is_donor_atom)
        self.assertEqual(atom.donor_slots, 0)
        self.assertIsNone(atom.hydrogen_bond_partners)

    def test_update_donor_state_false_but_slots(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.update_donor_state(is_donor_atom=False, donor_slots=2)

        desired_msg = 'A non-donor atom does not have any donor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_true_but_none_slots(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.update_donor_state(is_donor_atom=True, donor_slots=0)

        desired_msg = 'Donor atom need donor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_true_but_no_slots(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.update_donor_state(is_donor_atom=True, donor_slots=0)

        desired_msg = 'Donor atom need donor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_true_but_negative_slots(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.update_donor_state(is_donor_atom=True, donor_slots=-1)

        desired_msg = 'Donor atom need donor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_is_donor_atom_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.update_donor_state(is_donor_atom=42, donor_slots=2)

        desired_msg = create_data_type_exception_messages(parameter_name='is_donor_atom', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_donor_slots_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.update_donor_state(is_donor_atom=True, donor_slots=2.5)

        desired_msg = create_data_type_exception_messages(parameter_name='donor_slots', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_true(self):
        from yeti.systems.building_blocks import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        atom.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        self.assertTrue(atom.is_acceptor)
        self.assertEqual(atom.acceptor_slots, 2)
        self.assertListEqual(atom.hydrogen_bond_partners['subsystem'], [[], []])

    def test_update_acceptor_state_false(self):
        from yeti.systems.building_blocks import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        atom.update_acceptor_state(is_acceptor=False, acceptor_slots=0)

        self.assertFalse(atom.is_acceptor)
        self.assertEqual(atom.acceptor_slots, 0)
        self.assertIsNone(atom.hydrogen_bond_partners)

    def test_update_acceptor_state_false_but_slots(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.update_acceptor_state(is_acceptor=False, acceptor_slots=2)

        desired_msg = 'A non-acceptor atom does not have any acceptor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_true_but_none_slots(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.update_acceptor_state(is_acceptor=True, acceptor_slots=0)

        desired_msg = 'Acceptor atom need acceptor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_true_but_no_slots(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.update_acceptor_state(is_acceptor=True, acceptor_slots=0)

        desired_msg = 'Acceptor atom need acceptor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_true_but_negative_slots(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.update_acceptor_state(is_acceptor=True, acceptor_slots=-1)

        desired_msg = 'Acceptor atom need acceptor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_is_donor_atom_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.update_acceptor_state(is_acceptor=42, acceptor_slots=2)

        desired_msg = create_data_type_exception_messages(parameter_name='is_acceptor', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_donor_slots_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.update_acceptor_state(is_acceptor=True, acceptor_slots=2.5)

        desired_msg = create_data_type_exception_messages(parameter_name='acceptor_slots', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partner_donor(self):
        from yeti.systems.building_blocks import Atom

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)

        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_02.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        atom_01.__update_hydrogen_bond_partner__(atom=atom_02, frame=1, system_name='subsystem')

        self.assertDictEqual(atom_01.hydrogen_bond_partners, dict(subsystem=[[], [atom_02]]))

    def test_update_hydrogen_bond_partner_acceptor(self):
        from yeti.systems.building_blocks import Atom

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)

        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_02.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        atom_02.__update_hydrogen_bond_partner__(atom=atom_01, frame=1, system_name='subsystem')

        self.assertDictEqual(atom_02.hydrogen_bond_partners, dict(subsystem=[[], [atom_01]]))

    def test_update_hydrogen_bond_partner_more_atoms(self):
        from yeti.systems.building_blocks import Atom

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)

        atom_02 = Atom(structure_file_index=1, subsystem_index=0, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_02.update_donor_state(is_donor_atom=True, donor_slots=1)

        atom_03 = Atom(structure_file_index=2, subsystem_index=1, name='C',
                       xyz_trajectory=np.arange(12, 18).reshape((2, 3)))
        atom_03.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        atom_03.__update_hydrogen_bond_partner__(atom=atom_01, frame=1, system_name='subsystem')
        atom_03.__update_hydrogen_bond_partner__(atom=atom_02, frame=1, system_name='subsystem')

        self.assertDictEqual(atom_03.hydrogen_bond_partners, dict(subsystem=[[], [atom_01, atom_02]]))

    def test_update_hydrogen_bond_partner_atom_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)

        with self.assertRaises(AtomException) as context:
            atom_01.__update_hydrogen_bond_partner__(atom=5, frame=1, system_name='subsystem')

        desired_msg = create_data_type_exception_messages(parameter_name='atom', data_type_name='atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partner_frame_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)

        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_02.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        with self.assertRaises(AtomException) as context:
            atom_01.__update_hydrogen_bond_partner__(atom=atom_02, frame='1', system_name='subsystem')

        desired_msg = create_data_type_exception_messages(parameter_name='frame', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partner_frame_negative(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)

        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_02.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        with self.assertRaises(AtomException) as context:
            atom_01.__update_hydrogen_bond_partner__(atom=atom_02, frame=-1, system_name='subsystem')

        desired_msg = 'Frame has to be a positive integer.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partner_system_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)

        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_02.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        with self.assertRaises(AtomException) as context:
            atom_01.__update_hydrogen_bond_partner__(atom=atom_02, frame=1, system_name=6.8)

        desired_msg = create_data_type_exception_messages(parameter_name='system_name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partner_system_not_exist(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)

        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_02.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        with self.assertRaises(AtomException) as context:
            atom_01.__update_hydrogen_bond_partner__(atom=atom_02, frame=1, system_name='wrong')

        desired_msg = 'Subsystem does not exist. Create it first!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partner_bond_already_exists(self):
        from yeti.systems.building_blocks import Atom, AtomWarning

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)

        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_02.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        atom_01.__update_hydrogen_bond_partner__(atom=atom_02, frame=1, system_name='subsystem')

        with self.assertRaises(AtomWarning) as context:
            atom_01.__update_hydrogen_bond_partner__(atom=atom_02, frame=1, system_name='subsystem')

        desired_msg = 'Hydrogen bond already exists. Skipping...'
        self.assertEqual(desired_msg, str(context.exception))

    def test_add_hydrogen_bond_partner(self):
        from yeti.systems.building_blocks import Atom

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)

        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_02.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        atom_01.add_hydrogen_bond_partner(frame=0, atom=atom_02, system_name='subsystem')

        self.assertDictEqual(atom_01.hydrogen_bond_partners, dict(subsystem=[[atom_02], []]))
        self.assertDictEqual(atom_02.hydrogen_bond_partners, dict(subsystem=[[atom_01], []]))

    def test_add_hydrogen_bond_partner_is_None(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_02.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        with self.assertRaises(AtomException) as context:
            atom_01.add_hydrogen_bond_partner(frame=0, atom=atom_02, system_name='subsystem')

        desired_msg = 'This atom is neither acceptor nor a donor atom. Update its state first!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_add_hydrogen_bond_partner_atom_is_None(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)

        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom_01.add_hydrogen_bond_partner(frame=0, atom=atom_02, system_name='subsystem')

        desired_msg = 'Parameter atom is neither acceptor nor a donor atom. Update its state first!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_purge_hydrogen_bond_partner_history_donor(self):
        from yeti.systems.building_blocks import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom.update_donor_state(is_donor_atom=True, donor_slots=1)
        atom.hydrogen_bond_partners['subsystem'][1].append(5)
        atom.purge_hydrogen_bond_partner_history(system_name='subsystem')

        self.assertDictEqual(atom.hydrogen_bond_partners, dict(subsystem=[[], []]))

    def test_purge_hydrogen_bond_partner_history_acceptor(self):
        from yeti.systems.building_blocks import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom.update_acceptor_state(is_acceptor=True, acceptor_slots=1)
        atom.hydrogen_bond_partners['subsystem'][1].append(5)
        atom.purge_hydrogen_bond_partner_history(system_name='subsystem')

        self.assertDictEqual(atom.hydrogen_bond_partners, dict(subsystem=[[], []]))

    def test_purge_hydrogen_bond_partner_history_subsystem_wrong_data_type(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom.update_acceptor_state(is_acceptor=True, acceptor_slots=1)

        with self.assertRaises(AtomException) as context:
            atom.purge_hydrogen_bond_partner_history(system_name=42)

        desired_msg = create_data_type_exception_messages(parameter_name='system_name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_purge_hydrogen_bond_partner_history_system_name_not_exist(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom.update_acceptor_state(is_acceptor=True, acceptor_slots=1)

        with self.assertRaises(AtomException) as context:
            atom.purge_hydrogen_bond_partner_history(system_name='non-existent')

        desired_msg = 'Subsystem does not exist. Create it first!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_purge_hydrogen_bond_partner_history_hydrogen_bond_partners_is_none(self):
        from yeti.systems.building_blocks import Atom, AtomException

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        with self.assertRaises(AtomException) as context:
            atom.purge_hydrogen_bond_partner_history(system_name='subsystem')

        desired_msg = 'The given atom is neither donor nor acceptor. Purging does not make sense!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_str(self):
        from yeti.systems.building_blocks import Atom

        atom = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))

        self.assertEqual(str(atom), atom.name)


if __name__ == '__main__':
    unittest.main()
