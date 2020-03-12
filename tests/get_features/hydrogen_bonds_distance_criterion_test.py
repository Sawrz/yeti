import unittest

import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintTestCase, BlueprintExceptionsTestCase
from tests.get_features.hydrogen_bonds_test import HydrogenBondTestCase


class StaticHelperMethodsTestCase(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.get_features.hydrogen_bonds import HydrogenBondsDistanceCriterion

        self.donor = Atom(structure_file_index=8, subsystem_index=10, name='G',
                          xyz_trajectory=np.array([[0.1, 0.4, 0.3], [0.1, 0.4, 0.3], [0.1, 0.4, 0.3]]))
        self.donor_atom = Atom(structure_file_index=9, subsystem_index=11, name='H',
                               xyz_trajectory=np.array([[0.1, 0.5, 0.2], [0.1, 0.5, 0.2], [0.5, 0.5, 0.2]]))
        self.acceptor = Atom(structure_file_index=10, subsystem_index=11, name='I',
                             xyz_trajectory=np.array([[0.1, 0.6, 0.4], [0.1, 0.7, 0.4], [0.1, 0.6, 0.4]]))

        self.donor.add_covalent_bond(atom=self.donor_atom)
        self.donor_atom.update_donor_state(is_donor_atom=True, donor_slots=1)
        self.acceptor.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        frames = len(self.donor.xyz_trajectory)
        unit_cell_angles = np.array([[90, 90, 90]], dtype=np.float32)
        unit_cell_vectors = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float32)

        self.unit_cell_angles = np.repeat(unit_cell_angles, repeats=frames, axis=0)
        self.unit_cell_vectors = np.repeat(unit_cell_vectors, repeats=frames, axis=0)

        self.frame = 0
        self.hydrogen_bonds = HydrogenBondsDistanceCriterion


class HelperMethodsTestCase(StaticHelperMethodsTestCase):
    def setUp(self) -> None:
        super(HelperMethodsTestCase, self).setUp()

        self.hydrogen_bonds = self.hydrogen_bonds(atoms=(self.donor, self.donor_atom, self.acceptor),
                                                  periodic=True, unit_cell_angles=self.unit_cell_angles,
                                                  unit_cell_vectors=self.unit_cell_vectors,
                                                  system_name='test',
                                                  number_of_frames=len(self.unit_cell_angles))


class TestSortDistances(StaticHelperMethodsTestCase):
    def setUp(self) -> None:
        super(TestSortDistances, self).setUp()

        self.xyz = self.acceptor.xyz_trajectory[self.frame]
        self.hydrogen_bond_partners = [self.donor, self.acceptor]

    def test_trivial_one_entry(self):
        res_distances, res_args = self.hydrogen_bonds.__get_sorted_distances__(xyz_triplet=self.xyz,
                                                                               hydrogen_bond_partners=[
                                                                                   self.hydrogen_bond_partners[1]],
                                                                               frame=self.frame)

        exp_dist = np.array([0])
        npt.assert_almost_equal(res_distances, exp_dist, decimal=5)

        exp_args = np.array([0])
        npt.assert_array_equal(res_args, exp_args)

    def test_trivial_multi_entries(self):
        res_distances, res_args = self.hydrogen_bonds.__get_sorted_distances__(xyz_triplet=self.xyz,
                                                                               hydrogen_bond_partners=self.hydrogen_bond_partners,
                                                                               frame=self.frame)

        exp_dist = np.array([0.22361, 0])
        npt.assert_almost_equal(res_distances, exp_dist, decimal=5)

        exp_args = np.array([1, 0])
        npt.assert_array_equal(res_args, exp_args)


class TestCheck(StaticHelperMethodsTestCase):
    def setUp(self) -> None:
        from yeti.get_features.hydrogen_bonds import Triplet

        super(TestCheck, self).setUp()

        self.triplet = Triplet(donor_atom=self.donor_atom, acceptor=self.acceptor, periodic=True,
                               unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)

    def test_old_distances_smaller_than_new_triplet_dist(self):
        distances = np.array([0.14, 0.15])
        indices = np.array([1, 0])

        res_index = self.hydrogen_bonds.__check__(sorted_args=indices, distances=distances, triplet=self.triplet,
                                                  frame=self.frame)

        self.assertIsNone(res_index)

    def test_old_distances_greater_than_new_triplet_dist(self):
        distances = np.array([0.14, 0.35, 0.20])
        indices = np.array([0, 2, 1])

        res_index = self.hydrogen_bonds.__check__(sorted_args=indices, distances=distances, triplet=self.triplet,
                                                  frame=self.frame)

        self.assertEqual(1, res_index)


class TestReplace(HelperMethodsTestCase):
    pass





class HydrogenBondDistanceTestCase(HydrogenBondTestCase):
    def setUpAtoms(self) -> None:
        self.atoms = (self.donor_01, self.donor_atom_01, self.acceptor_01,
                      self.donor_02, self.donor_atom_02, self.acceptor_02,
                      self.donor_03, self.donor_atom_03, self.acceptor_03,
                      self.donor_04, self.donor_atom_04, self.acceptor_04)

    def setUp(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.get_features.hydrogen_bonds import HydrogenBondsDistanceCriterion

        super(HydrogenBondDistanceTestCase, self).setUp()

        # create new objects
        self.donor_02.xyz_trajectory -= 0.2
        self.donor_atom_02.xyz_trajectory -= 0.2
        self.acceptor_02.xyz_trajectory -= 0.2

        self.donor_03 = Atom(structure_file_index=8, subsystem_index=10, name='G',
                             xyz_trajectory=np.array([[0.1, 0.4, 0.3], [0.1, 0.4, 0.3], [0.1, 0.4, 0.3]]))
        self.donor_atom_03 = Atom(structure_file_index=9, subsystem_index=11, name='H',
                                  xyz_trajectory=np.array([[0.1, 0.5, 0.2], [0.1, 0.5, 0.2], [0.5, 0.5, 0.2]]))
        self.acceptor_03 = Atom(structure_file_index=10, subsystem_index=11, name='I',
                                xyz_trajectory=np.array([[0.1, 0.6, 0.4], [0.1, 0.7, 0.4], [0.1, 0.6, 0.4]]))

        self.donor_03.add_covalent_bond(atom=self.donor_atom_03)
        self.donor_atom_03.update_donor_state(is_donor_atom=True, donor_slots=1)
        self.acceptor_03.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        self.donor_04 = Atom(structure_file_index=11, subsystem_index=10, name='J',
                             xyz_trajectory=np.array([[0.1, 0.5, 0.3], [0.1, 0.4, 0.3], [0.1, 0.4, 0.3]]))
        self.donor_atom_04 = Atom(structure_file_index=12, subsystem_index=11, name='K',
                                  xyz_trajectory=np.array([[0.1, 0.6, 0.3], [0.1, 0.5, 0.2], [0.5, 0.5, 0.2]]))
        self.acceptor_04 = Atom(structure_file_index=13, subsystem_index=11, name='L',
                                xyz_trajectory=np.array([[0.1, 0.6, 0.39], [0.1, 0.7, 0.4], [0.1, 0.6, 0.4]]))

        self.donor_04.add_covalent_bond(atom=self.donor_atom_04)
        self.donor_atom_04.update_donor_state(is_donor_atom=True, donor_slots=1)
        self.acceptor_04.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        self.setUpAtoms()
        self.hydrogen_bonds = HydrogenBondsDistanceCriterion(atoms=self.atoms, periodic=True,
                                                             unit_cell_angles=self.unit_cell_angles,
                                                             unit_cell_vectors=self.unit_cell_vectors,
                                                             system_name='test',
                                                             number_of_frames=self.number_of_frames)

        self.triplets = self.hydrogen_bonds.__build_triplets__(distance_cutoff=0.25, angle_cutoff=2.0)


class TestHydrogenBondRepresentationMethodsTooManyDonors(HydrogenBondDistanceTestCase):
    def setUpAtoms(self) -> None:
        self.atoms = (self.donor_01, self.donor_atom_01, self.acceptor_01,
                      self.donor_02, self.donor_atom_02,
                      self.donor_03, self.donor_atom_03,
                      self.donor_04, self.donor_atom_04)

    def test_get_hydrogen_bonds_in_frame(self):
        self.hydrogen_bonds.__get_hydrogen_bonds_in_frame__(triplets=self.triplets, frame=0)

        self.assertIsNone(self.hydrogen_bonds.atoms[0].hydrogen_bond_partners)
        self.assertIsNone(self.hydrogen_bonds.atoms[3].hydrogen_bond_partners)
        self.assertIsNone(self.hydrogen_bonds.atoms[5].hydrogen_bond_partners)
        self.assertIsNone(self.hydrogen_bonds.atoms[7].hydrogen_bond_partners)
        self.assertListEqual([[], [], []], self.hydrogen_bonds.atoms[1].hydrogen_bond_partners['test'])
        self.assertListEqual([[self.donor_atom_02, self.donor_atom_04], [], []],
                             self.hydrogen_bonds.atoms[2].hydrogen_bond_partners['test'])
        self.assertListEqual([[self.acceptor_01], [], []], self.hydrogen_bonds.atoms[4].hydrogen_bond_partners['test'])
        self.assertListEqual([[], [], []], self.hydrogen_bonds.atoms[6].hydrogen_bond_partners['test'])
        self.assertListEqual([[self.acceptor_01], [], []], self.hydrogen_bonds.atoms[8].hydrogen_bond_partners['test'])


class TestHydrogenBondRepresentationMethodsTooManyAcceptors(HydrogenBondDistanceTestCase):
    def setUpAtoms(self) -> None:
        self.atoms = (self.donor_01, self.donor_atom_01, self.acceptor_01, self.acceptor_02, self.acceptor_03,
                      self.acceptor_04)

    def test_get_hydrogen_bonds_in_frame(self):
        self.hydrogen_bonds.__get_hydrogen_bonds_in_frame__(triplets=self.triplets, frame=0)

        self.assertIsNone(self.donor_01.hydrogen_bond_partners)
        self.assertListEqual([[self.acceptor_04], [], []], self.hydrogen_bonds.atoms[1].hydrogen_bond_partners['test'])
        self.assertListEqual([[], [], []],
                             self.hydrogen_bonds.atoms[2].hydrogen_bond_partners['test'])
        self.assertListEqual([[], [], []], self.hydrogen_bonds.atoms[3].hydrogen_bond_partners['test'])
        self.assertListEqual([[], [], []], self.hydrogen_bonds.atoms[4].hydrogen_bond_partners['test'])
        self.assertListEqual([[self.donor_atom_01], [], []],
                             self.hydrogen_bonds.atoms[5].hydrogen_bond_partners['test'])


class TestHydrogenBondRepresentationMethodsTooManyAcceptorsAndDonors(HydrogenBondDistanceTestCase):
    def setUpAtoms(self) -> None:
        self.atoms = (self.donor_01, self.donor_atom_01, self.acceptor_03,
                      self.donor_02, self.donor_atom_02, self.acceptor_04,
                      self.donor_03, self.donor_atom_03,
                      self.donor_04, self.donor_atom_04)

    def setUp(self) -> None:
        super(TestHydrogenBondRepresentationMethodsTooManyAcceptorsAndDonors, self).setUp()

        self.acceptor_03.add_hydrogen_bond_partner(frame=0, atom=self.donor_atom_02, system_name='test')
        self.acceptor_03.add_hydrogen_bond_partner(frame=0, atom=self.donor_atom_04, system_name='test')
        self.acceptor_04.add_hydrogen_bond_partner(frame=0, atom=self.donor_atom_01, system_name='test')
        self.acceptor_04.add_hydrogen_bond_partner(frame=0, atom=self.donor_atom_03, system_name='test')

    def test_get_hydrogen_bonds_in_frame(self):
        self.hydrogen_bonds.__get_hydrogen_bonds_in_frame__(triplets=self.triplets, frame=0)

        # Donors
        self.assertIsNone(self.donor_01.hydrogen_bond_partners)
        self.assertIsNone(self.donor_02.hydrogen_bond_partners)
        self.assertIsNone(self.donor_03.hydrogen_bond_partners)
        self.assertIsNone(self.donor_04.hydrogen_bond_partners)

        # Acceptors
        self.assertListEqual([[self.donor_atom_02], [], []],
                             self.hydrogen_bonds.atoms[2].hydrogen_bond_partners['test'])
        self.assertListEqual([[self.donor_atom_01, self.donor_atom_04], [], []],
                             self.hydrogen_bonds.atoms[5].hydrogen_bond_partners['test'])

        # Donor Atoms
        self.assertListEqual([[self.acceptor_04], [], []], self.hydrogen_bonds.atoms[1].hydrogen_bond_partners['test'])
        self.assertListEqual([[self.acceptor_03], [], []], self.hydrogen_bonds.atoms[4].hydrogen_bond_partners['test'])
        self.assertListEqual([[], [], []], self.hydrogen_bonds.atoms[7].hydrogen_bond_partners['test'])
        self.assertListEqual([[self.acceptor_04], [], []], self.hydrogen_bonds.atoms[9].hydrogen_bond_partners['test'])


class HydrogenBondExceptionsDistanceTestCase(HydrogenBondDistanceTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.get_features.hydrogen_bonds import HydrogenBondsException

        super(HydrogenBondExceptionsDistanceTestCase, self).setUp()

        self.exception = HydrogenBondsException


class TestHydrogenBondRepresentationMethodExceptions(HydrogenBondExceptionsDistanceTestCase):
    def test_get_hydrogen_bonds_in_frame_triplets_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds.__get_hydrogen_bonds_in_frame__(triplets=list(self.triplets), frame=0)

        desired_msg = self.create_data_type_exception_messages(parameter_name='triplets', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_hydrogen_bonds_in_frame_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds.__get_hydrogen_bonds_in_frame__(triplets=self.triplets, frame=0.)

        desired_msg = self.create_data_type_exception_messages(parameter_name='frame', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
