import unittest

import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintTestCase, BlueprintExceptionsTestCase


class AtomTestCase(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Atom

        self.atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A',
                            xyz_trajectory=np.arange(6).reshape((2, 3)))
        self.atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                            xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        self.atom_03 = Atom(structure_file_index=3, subsystem_index=2, name='C',
                            xyz_trajectory=np.arange(12, 18).reshape((2, 3)))


class TestStandardMethods(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Atom

        xyz_trajectory = np.arange(6).reshape((3, 2))

        self.atom = Atom(name='test', subsystem_index=16, structure_file_index=42, xyz_trajectory=xyz_trajectory)

    def test_init(self):
        from yeti.systems.building_blocks import AtomException

        self.assertEqual(self.atom.ensure_data_type.exception_class, AtomException)
        self.assertEqual(self.atom.name, 'test')
        self.assertEqual(self.atom.subsystem_index, 16)
        self.assertEqual(self.atom.structure_file_index, 42)
        self.assertIsNone(self.atom.residue)
        npt.assert_equal(self.atom.xyz_trajectory, np.arange(6).reshape((2, 3)))

        self.assertIsNone(self.atom.element)
        self.assertTupleEqual(self.atom.covalent_bond_partners, ())
        self.assertFalse(self.atom.is_donor_atom)
        self.assertFalse(self.atom.is_acceptor)
        self.assertEqual(self.atom.donor_slots, 0)
        self.assertEqual(self.atom.acceptor_slots, 0)
        self.assertIsNone(self.atom.hydrogen_bond_partners)

    def test_str(self):
        self.assertEqual(str(self.atom), self.atom.name)


class TestAtomManipulationMethods(AtomTestCase):
    def test_add_frame(self):
        self.atom_03._add_frame(frame=[42, 43, 44])

        npt.assert_array_equal(self.atom_03.xyz_trajectory, np.array([[12, 13, 14], [15, 16, 17], [42, 43, 44]]))

    def test_delete_frame(self):
        self.atom_03._delete_frame(frame_index=1)

        npt.assert_array_equal(self.atom_03.xyz_trajectory, np.array([[12, 13, 14]]))


class TestIntraMoleculeConnectionMethods(AtomTestCase):
    def test_set_residue(self):
        from yeti.systems.building_blocks import Residue

        res = Residue(subsystem_index=0, structure_file_index=1, name='RES')
        self.atom_01.set_residue(residue=res)

        self.assertEqual(self.atom_01.residue, res)

    def test_add_system(self):
        self.atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)
        self.atom_01.add_system('test')

        self.assertDictEqual({'test': [[], []]}, self.atom_01.hydrogen_bond_partners)


class TestCovalentBondMethods(AtomTestCase):
    def test_update_covalent_bond(self):
        self.atom_01.__update_covalent_bond__(atom=self.atom_02)

        self.assertTupleEqual(self.atom_01.covalent_bond_partners, (self.atom_02,))
        self.assertTupleEqual(self.atom_02.covalent_bond_partners, ())
        self.assertTupleEqual(self.atom_03.covalent_bond_partners, ())

    def test_update_covalent_bond_more_atoms(self):
        self.atom_01.__update_covalent_bond__(atom=self.atom_02)
        self.atom_01.__update_covalent_bond__(atom=self.atom_03)

        self.assertTupleEqual(self.atom_01.covalent_bond_partners, (self.atom_02, self.atom_03))
        self.assertTupleEqual(self.atom_02.covalent_bond_partners, ())
        self.assertTupleEqual(self.atom_03.covalent_bond_partners, ())

    def test_add_covalent_bonds(self):
        self.atom_01.add_covalent_bond(atom=self.atom_02)

        self.assertTupleEqual(self.atom_01.covalent_bond_partners, (self.atom_02,))
        self.assertTupleEqual(self.atom_02.covalent_bond_partners, (self.atom_01,))


class TestUpdateChargeStateMethods(AtomTestCase):
    def test_update_donor_state_true(self):
        self.atom_01.update_donor_state(is_donor_atom=True, donor_slots=2)

        self.assertTrue(self.atom_01.is_donor_atom)
        self.assertEqual(self.atom_01.donor_slots, 2)
        self.assertDictEqual(self.atom_01.hydrogen_bond_partners, {})

    def test_update_donor_state_false(self):
        self.atom_01.update_donor_state(is_donor_atom=False, donor_slots=0)

        self.assertFalse(self.atom_01.is_donor_atom)
        self.assertEqual(self.atom_01.donor_slots, 0)
        self.assertIsNone(self.atom_01.hydrogen_bond_partners)

    def test_update_acceptor_state_true(self):
        self.atom_01.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        self.assertTrue(self.atom_01.is_acceptor)
        self.assertEqual(self.atom_01.acceptor_slots, 2)
        self.assertDictEqual(self.atom_01.hydrogen_bond_partners, {})

    def test_update_acceptor_state_false(self):
        self.atom_01.update_acceptor_state(is_acceptor=False, acceptor_slots=0)

        self.assertFalse(self.atom_01.is_acceptor)
        self.assertEqual(self.atom_01.acceptor_slots, 0)
        self.assertIsNone(self.atom_01.hydrogen_bond_partners)


class TestHydrogenBondMethods(AtomTestCase):
    def setUp(self) -> None:
        super(TestHydrogenBondMethods, self).setUp()

        self.atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)
        self.atom_01.add_system(system_name='subsystem')

        self.atom_02.update_acceptor_state(is_acceptor=True, acceptor_slots=2)
        self.atom_02.add_system(system_name='subsystem')

        self.atom_03.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

    def test_update_hydrogen_bond_partners_false(self):
        self.atom_01.__reset_hydrogen_bond_partners__(is_hydrogen_bond_active=False)
        self.assertIsNone(self.atom_01.hydrogen_bond_partners)

    def test_reset_hydrogen_bond_partners_true_no_entries(self):
        self.atom_03.__reset_hydrogen_bond_partners__(is_hydrogen_bond_active=True)
        self.assertDictEqual(self.atom_03.hydrogen_bond_partners, {})

    def test_reset_hydrogen_bond_partners_true_entries(self):
        self.atom_01.__reset_hydrogen_bond_partners__(is_hydrogen_bond_active=True)
        self.assertDictEqual(self.atom_01.hydrogen_bond_partners, dict(subsystem=[[], []]))
        self.assertListEqual(self.atom_01.hydrogen_bond_partners['subsystem'], [[], []])

    def test_update_hydrogen_bond_partner_donor(self):
        self.atom_01.__update_hydrogen_bond_partner__(atom=self.atom_02, frame=1, system_name='subsystem', add=True)

        self.assertDictEqual(self.atom_01.hydrogen_bond_partners, dict(subsystem=[[], [self.atom_02]]))

    def test_update_hydrogen_bond_partner_acceptor(self):
        self.atom_02.__update_hydrogen_bond_partner__(atom=self.atom_01, frame=1, system_name='subsystem', add=True)

        self.assertDictEqual(self.atom_02.hydrogen_bond_partners, dict(subsystem=[[], [self.atom_01]]))

    def test_update_hydrogen_bond_partner_more_atoms(self):
        self.atom_03.add_system(system_name='subsystem')

        self.atom_03.__update_hydrogen_bond_partner__(atom=self.atom_01, frame=1, system_name='subsystem', add=True)
        self.atom_03.__update_hydrogen_bond_partner__(atom=self.atom_02, frame=1, system_name='subsystem', add=True)

        self.assertDictEqual(self.atom_01.hydrogen_bond_partners, dict(subsystem=[[], []]))
        self.assertDictEqual(self.atom_02.hydrogen_bond_partners, dict(subsystem=[[], []]))
        self.assertDictEqual(self.atom_03.hydrogen_bond_partners, dict(subsystem=[[], [self.atom_01, self.atom_02]]))

    def test_update_hydrogen_bond_partner_remove(self):
        self.atom_01.add_hydrogen_bond_partner(frame=0, atom=self.atom_02, system_name='subsystem')
        self.atom_01.__update_hydrogen_bond_partner__(atom=self.atom_02, frame=0, system_name='subsystem', add=False)

        self.assertDictEqual(self.atom_01.hydrogen_bond_partners, dict(subsystem=[[], []]))
        self.assertDictEqual(self.atom_02.hydrogen_bond_partners, dict(subsystem=[[self.atom_01], []]))

    def test_add_hydrogen_bond_partner(self):
        self.atom_01.add_hydrogen_bond_partner(frame=0, atom=self.atom_02, system_name='subsystem')

        self.assertDictEqual(self.atom_01.hydrogen_bond_partners, dict(subsystem=[[self.atom_02], []]))
        self.assertDictEqual(self.atom_02.hydrogen_bond_partners, dict(subsystem=[[self.atom_01], []]))

    def test_remove_hydrogen_bond_partner(self):
        self.atom_01.add_hydrogen_bond_partner(frame=0, atom=self.atom_02, system_name='subsystem')
        self.atom_01.remove_hydrogen_bond_partner(frame=0, atom=self.atom_02, system_name='subsystem')

        self.assertDictEqual(self.atom_01.hydrogen_bond_partners, dict(subsystem=[[], []]))
        self.assertDictEqual(self.atom_02.hydrogen_bond_partners, dict(subsystem=[[], []]))

    def test_purge_hydrogen_bond_partner_history_donor(self):
        self.atom_01.hydrogen_bond_partners['subsystem'][1].append(5)
        self.atom_01.purge_hydrogen_bond_partner_history(system_name='subsystem')

        self.assertDictEqual(self.atom_01.hydrogen_bond_partners, dict(subsystem=[[], []]))

    def test_purge_hydrogen_bond_partner_history_acceptor(self):
        self.atom_02.hydrogen_bond_partners['subsystem'][1].append(5)
        self.atom_02.purge_hydrogen_bond_partner_history(system_name='subsystem')

        self.assertDictEqual(self.atom_02.hydrogen_bond_partners, dict(subsystem=[[], []]))


class AtomExceptionsTestCase(AtomTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import AtomException

        super(AtomExceptionsTestCase, self).setUp()

        self.exception = AtomException
        self.warning = Warning


class TestStandardMethodExceptions(AtomExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Atom

        super(TestStandardMethodExceptions, self).setUp()
        self.atom = Atom

    def test_init_name_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom(name=12, subsystem_index=16, structure_file_index=42, xyz_trajectory=np.arange(6).reshape((2, 3)))

        desired_msg = self.create_data_type_exception_messages(parameter_name='name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_subsystem_index_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom(name='test', subsystem_index=16.1, structure_file_index=42,
                      xyz_trajectory=np.arange(6).reshape((2, 3)))

        desired_msg = self.create_data_type_exception_messages(parameter_name='subsystem_index', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_structure_file_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom(name='test', subsystem_index=16, structure_file_index=42.0,
                      xyz_trajectory=np.arange(6).reshape((2, 3)))

        desired_msg = self.create_data_type_exception_messages(parameter_name='structure_file_index',
                                                               data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_xyz_wrong_shape(self):
        xyz_trajectory = np.arange(8).reshape((4, 2))

        with self.assertRaises(self.exception) as context:
            self.atom(name='test', subsystem_index=16, structure_file_index=42, xyz_trajectory=xyz_trajectory)

        self.assertEqual(str(context.exception),
                         'Wrong shape for parameter "xyz_trajectory". Desired shape: (None, 3).')

    def test_init_xyz_wrong_data_type(self):
        xyz_trajectory = list(np.arange(6).reshape((2, 3)))

        with self.assertRaises(self.exception) as context:
            self.atom(name='test', subsystem_index=16, structure_file_index=42, xyz_trajectory=xyz_trajectory)

        desired_msg = self.create_data_type_exception_messages(parameter_name='xyz_trajectory',
                                                               data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))


class TestIntraMoleculeConnectionMethodExceptions(AtomExceptionsTestCase):
    def test_set_residue_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.set_residue(residue=24)

        desired_msg = self.create_data_type_exception_messages(parameter_name='residue', data_type_name='Residue')
        self.assertEqual(desired_msg, str(context.exception))

    def test_set_residue_already_assigned(self):
        from yeti.systems.building_blocks import Residue

        residue = Residue(subsystem_index=0, structure_file_index=1, name='RES')
        self.atom_01.set_residue(residue=residue)

        with self.assertWarns(self.warning) as context:
            self.atom_01.set_residue(residue=residue)

        desired_msg = 'This atom belongs already to a residue. Changing relationship...'
        self.assertEqual(desired_msg, str(context.warning))

    def test_add_system_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.add_system(system_name=5)

        desired_msg = self.create_data_type_exception_messages(parameter_name='system_name', data_type_name='str')
        self.assertEqual(str(context.exception), desired_msg)

    def test_add_system_no_donor_or_acceptor(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.add_system('test')

        desired_msg = 'This atom is neither acceptor nor a donor atom. Update its state first!'
        self.assertEqual(str(context.exception), desired_msg)


class TestCovalentbondMethodsExceptions(AtomExceptionsTestCase):
    def test_update_covalent_bond_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.__update_covalent_bond__(atom=42)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom', data_type_name='atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_covalent_bond_exists_already(self):
        self.atom_01.__update_covalent_bond__(atom=self.atom_02)

        with self.assertWarns(self.warning) as context:
            self.atom_01.__update_covalent_bond__(atom=self.atom_02)

        desired_msg = 'Covalent bond already exists. Skipping...'
        self.assertEqual(desired_msg, str(context.warning))

    def test_update_covalent_bond_with_itself(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.__update_covalent_bond__(atom=self.atom_01)

        desired_msg = 'Atom can not have a covalent bond with itself.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_add_covalent_bonds_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.add_covalent_bond(atom=42)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom', data_type_name='atom')
        self.assertEqual(desired_msg, str(context.exception))


class TestUpdateChargeStateMethodExceptions(AtomExceptionsTestCase):
    def test_update_donor_state_false_but_slots(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.update_donor_state(is_donor_atom=False, donor_slots=2)

        desired_msg = 'A non-donor atom does not have any donor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_true_but_none_slots(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.update_donor_state(is_donor_atom=True, donor_slots=0)

        desired_msg = 'Donor atom need donor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_true_but_no_slots(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.update_donor_state(is_donor_atom=True, donor_slots=0)

        desired_msg = 'Donor atom need donor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_true_but_negative_slots(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.update_donor_state(is_donor_atom=True, donor_slots=-1)

        desired_msg = 'Donor atom need donor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_is_donor_atom_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.update_donor_state(is_donor_atom=42, donor_slots=2)

        desired_msg = self.create_data_type_exception_messages(parameter_name='is_donor_atom', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_donor_state_donor_slots_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.update_donor_state(is_donor_atom=True, donor_slots=2.5)

        desired_msg = self.create_data_type_exception_messages(parameter_name='donor_slots', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_false_but_slots(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.update_acceptor_state(is_acceptor=False, acceptor_slots=2)

        desired_msg = 'A non-acceptor atom does not have any acceptor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_true_but_none_slots(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.update_acceptor_state(is_acceptor=True, acceptor_slots=0)

        desired_msg = 'Acceptor atom need acceptor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_true_but_no_slots(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.update_acceptor_state(is_acceptor=True, acceptor_slots=0)

        desired_msg = 'Acceptor atom need acceptor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_true_but_negative_slots(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.update_acceptor_state(is_acceptor=True, acceptor_slots=-1)

        desired_msg = 'Acceptor atom need acceptor slots.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_is_donor_atom_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.update_acceptor_state(is_acceptor=42, acceptor_slots=2)

        desired_msg = self.create_data_type_exception_messages(parameter_name='is_acceptor', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_acceptor_state_donor_slots_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.update_acceptor_state(is_acceptor=True, acceptor_slots=2.5)

        desired_msg = self.create_data_type_exception_messages(parameter_name='acceptor_slots', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))


class TestHydrogenBondMethodExceptions(AtomExceptionsTestCase):
    def setUp(self) -> None:
        super(TestHydrogenBondMethodExceptions, self).setUp()

        self.atom_01.update_donor_state(is_donor_atom=True, donor_slots=1)
        self.atom_01.add_system(system_name='subsystem')

        self.atom_02.update_acceptor_state(is_acceptor=True, acceptor_slots=1)
        self.atom_02.add_system(system_name='subsystem')

    def test_update_hydrogen_bond_partners_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.__reset_hydrogen_bond_partners__(is_hydrogen_bond_active=42)

        desired_msg = self.create_data_type_exception_messages(parameter_name='is_hydrogen_bond_active',
                                                               data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partner_atom_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.__update_hydrogen_bond_partner__(atom=5, frame=1, system_name='subsystem', add=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom', data_type_name='atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partner_frame_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.__update_hydrogen_bond_partner__(atom=self.atom_02, frame='1', system_name='subsystem',
                                                          add=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='frame', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partner_frame_negative(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.__update_hydrogen_bond_partner__(atom=self.atom_02, frame=-1, system_name='subsystem',
                                                          add=True)

        desired_msg = 'Frame has to be a positive integer.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partner_system_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.__update_hydrogen_bond_partner__(atom=self.atom_02, frame=1, system_name=6.8, add=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='system_name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partner_system_not_exist(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.__update_hydrogen_bond_partner__(atom=self.atom_02, frame=1, system_name='wrong', add=True)

        desired_msg = 'Subsystem does not exist. Create it first!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partner_add_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.__update_hydrogen_bond_partner__(atom=self.atom_02, frame=1, system_name='subsystem', add=42)

        desired_msg = self.create_data_type_exception_messages(parameter_name='add', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_update_hydrogen_bond_partner_bond_already_exists(self):
        self.atom_01.__update_hydrogen_bond_partner__(atom=self.atom_02, frame=1, system_name='subsystem', add=True)

        with self.assertWarns(self.warning) as context:
            self.atom_01.__update_hydrogen_bond_partner__(atom=self.atom_02, frame=1, system_name='subsystem', add=True)

        desired_msg = 'Hydrogen bond already exists. Skipping...'
        self.assertEqual(desired_msg, str(context.warning))

    def test_update_hydrogen_bond_partner_bond_already_deleted(self):
        with self.assertWarns(self.warning) as context:
            self.atom_01.__update_hydrogen_bond_partner__(atom=self.atom_02, frame=1, system_name='subsystem',
                                                          add=False)

        desired_msg = 'Atom not found. Skipping deletion...'
        self.assertEqual(desired_msg, str(context.warning))

    def test_add_hydrogen_bond_partner_is_None(self):
        self.atom_01.update_donor_state(is_donor_atom=False, donor_slots=0)

        with self.assertRaises(self.exception) as context:
            self.atom_01.add_hydrogen_bond_partner(frame=0, atom=self.atom_02, system_name='subsystem')

        desired_msg = 'This atom is neither acceptor nor a donor atom. Update its state first!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_add_hydrogen_bond_partner_atom_is_None(self):
        self.atom_02.update_acceptor_state(is_acceptor=False, acceptor_slots=0)

        with self.assertRaises(self.exception) as context:
            self.atom_01.add_hydrogen_bond_partner(frame=0, atom=self.atom_02, system_name='subsystem')

        desired_msg = 'Parameter atom is neither acceptor nor a donor atom. Update its state first!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_add_hydrogen_bond_partner_donor_atom_has_already_max_bonds(self):
        self.atom_01.add_hydrogen_bond_partner(frame=0, atom=self.atom_02, system_name='subsystem')
        self.atom_03.update_acceptor_state(is_acceptor=True, acceptor_slots=1)
        self.atom_03.add_system(system_name='subsystem')

        with self.assertRaises(self.exception) as context:
            self.atom_01.add_hydrogen_bond_partner(frame=0, atom=self.atom_03, system_name='subsystem')

        desired_msg = 'This atom has already the maximum amount of hydrogen bond partners. You need to remove one first!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_add_hydrogen_bond_partner_acceptor_has_already_max_bonds(self):
        self.atom_01.add_hydrogen_bond_partner(frame=0, atom=self.atom_02, system_name='subsystem')
        self.atom_03.update_donor_state(is_donor_atom=True, donor_slots=1)
        self.atom_03.add_system(system_name='subsystem')

        with self.assertRaises(self.exception) as context:
            self.atom_02.add_hydrogen_bond_partner(frame=0, atom=self.atom_03, system_name='subsystem')

        desired_msg = 'This atom has already the maximum amount of hydrogen bond partners. You need to remove one first!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_remove_hydrogen_bond_partner_is_None(self):
        self.atom_01.update_donor_state(is_donor_atom=False, donor_slots=0)

        with self.assertRaises(self.exception) as context:
            self.atom_01.remove_hydrogen_bond_partner(frame=0, atom=self.atom_02, system_name='subsystem')

        desired_msg = 'This atom is neither acceptor nor a donor atom. Update its state first!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_remove_hydrogen_bond_partner_atom_is_None(self):
        self.atom_02.update_acceptor_state(is_acceptor=False, acceptor_slots=0)

        with self.assertRaises(self.exception) as context:
            self.atom_01.remove_hydrogen_bond_partner(frame=0, atom=self.atom_02, system_name='subsystem')

        desired_msg = 'Parameter atom is neither acceptor nor a donor atom. Update its state first!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_purge_hydrogen_bond_partner_history_subsystem_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.purge_hydrogen_bond_partner_history(system_name=42)

        desired_msg = self.create_data_type_exception_messages(parameter_name='system_name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_purge_hydrogen_bond_partner_history_system_name_not_exist(self):
        with self.assertRaises(self.exception) as context:
            self.atom_01.purge_hydrogen_bond_partner_history(system_name='non-existent')

        desired_msg = 'Subsystem does not exist. Create it first!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_purge_hydrogen_bond_partner_history_hydrogen_bond_partners_is_none(self):
        self.atom_01.update_donor_state(is_donor_atom=False, donor_slots=0)

        with self.assertRaises(self.exception) as context:
            self.atom_01.purge_hydrogen_bond_partner_history(system_name='subsystem')

        desired_msg = 'The given atom is neither donor nor acceptor. Purging does not make sense!'
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
