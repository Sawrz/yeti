import copy
import unittest

import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintExceptionsTestCase
from tests.get_features.triplet_test import TripletTestCase


class HydrogenBondTestCase(TripletTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.get_features.hydrogen_bonds import Triplet, HydrogenBonds

        super(HydrogenBondTestCase, self).setUp()

        self.number_of_frames = self.donor.xyz_trajectory.shape[0]

        # remap existing objects
        self.donor_01 = self.donor
        self.donor_atom_01 = self.donor_atom
        self.acceptor_01 = self.acceptor

        self.triplet_01 = self.triplet

        del self.donor
        del self.donor_atom
        del self.acceptor
        del self.triplet

        # create new objects
        self.donor_02 = Atom(structure_file_index=5, subsystem_index=3, name='D',
                             xyz_trajectory=np.array([[0.1, 0.4, 0.3], [0.1, 0.4, 0.3], [0.1, 0.4, 0.3]]) + 0.2)
        self.donor_atom_02 = Atom(structure_file_index=6, subsystem_index=4, name='E',
                                  xyz_trajectory=np.array([[0.1, 0.5, 0.2], [0.1, 0.5, 0.2], [0.5, 0.5, 0.2]]) + 0.2)
        self.acceptor_02 = Atom(structure_file_index=7, subsystem_index=5, name='F',
                                xyz_trajectory=np.array([[0.1, 0.6, 0.4], [0.1, 0.7, 0.4], [0.1, 0.6, 0.4]]) + 0.2)

        self.donor_02.add_covalent_bond(atom=self.donor_atom_02)
        self.donor_atom_02.update_donor_state(is_donor_atom=True, donor_slots=1)
        self.acceptor_02.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        self.triplet_02 = Triplet(donor_atom=self.donor_atom_02, acceptor=self.acceptor_02, periodic=True,
                                  unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)

        self.atoms = (self.donor_01, self.donor_atom_01, self.acceptor_01,
                      self.donor_02, self.donor_atom_02, self.acceptor_02)

        self.triplets = (self.triplet_01, self.triplet_02)

        self.hydrogen_bonds = HydrogenBonds(atoms=self.atoms, periodic=True,
                                            unit_cell_angles=self.unit_cell_angles,
                                            unit_cell_vectors=self.unit_cell_vectors, system_name='test',
                                            number_of_frames=self.number_of_frames)


class HydrogenBondMultiProcessTestCase(HydrogenBondTestCase):
    def setUpMultipliers(self):
        self.atom_multiplier = 24
        self.trajectory_multiplier = 200

    def setUp(self) -> None:
        from yeti.get_features.hydrogen_bonds import HydrogenBonds

        super(HydrogenBondMultiProcessTestCase, self).setUp()

        # setup settings
        self.setUpMultipliers()

        # parameters for quick checks
        # self.atom_multiplier = 3
        # self.trajectory_multiplier = 10

        # replicating xyz and exp masks
        self.unit_cell_vectors = np.repeat(self.unit_cell_vectors, repeats=self.trajectory_multiplier, axis=0)
        self.unit_cell_angles = np.repeat(self.unit_cell_angles, repeats=self.trajectory_multiplier, axis=0)

        self.donor_01.xyz_trajectory = np.repeat(self.donor_01.xyz_trajectory, repeats=self.trajectory_multiplier,
                                                 axis=0)
        self.donor_atom_01.xyz_trajectory = np.repeat(self.donor_atom_01.xyz_trajectory,
                                                      repeats=self.trajectory_multiplier,
                                                      axis=0)
        self.acceptor_01.xyz_trajectory = np.repeat(self.acceptor_01.xyz_trajectory, repeats=self.trajectory_multiplier,
                                                    axis=0)
        self.donor_02.xyz_trajectory = np.repeat(self.donor_02.xyz_trajectory, repeats=self.trajectory_multiplier,
                                                 axis=0)
        self.donor_atom_02.xyz_trajectory = np.repeat(self.donor_atom_02.xyz_trajectory,
                                                      repeats=self.trajectory_multiplier,
                                                      axis=0)
        self.acceptor_02.xyz_trajectory = np.repeat(self.acceptor_02.xyz_trajectory, repeats=self.trajectory_multiplier,
                                                    axis=0)

        for key in self.donor_atom_01.hydrogen_bond_partners.keys():
            self.donor_atom_01.add_system(system_name=key)
            self.acceptor_01.add_system(system_name=key)
            self.donor_atom_02.add_system(system_name=key)
            self.acceptor_02.add_system(system_name=key)

        atoms = []
        for i in range(self.atom_multiplier):
            atoms.extend([copy.deepcopy(self.donor_01), copy.deepcopy(self.donor_atom_01),
                          copy.deepcopy(self.acceptor_01), copy.deepcopy(self.donor_02),
                          copy.deepcopy(self.donor_atom_02), copy.deepcopy(self.acceptor_02)])

        self.atoms = tuple(atoms)
        self.hydrogen_bonds = HydrogenBonds(atoms=self.atoms, periodic=True,
                                            unit_cell_angles=self.unit_cell_angles,
                                            unit_cell_vectors=self.unit_cell_vectors, system_name='test',
                                            number_of_frames=self.number_of_frames)

        del self.donor_01
        del self.donor_atom_01
        del self.acceptor_01
        del self.donor_02
        del self.donor_atom_02
        del self.acceptor_02


class TestStandardMethods(HydrogenBondTestCase):
    def test_init_default(self):
        from multiprocessing import cpu_count

        self.assertTupleEqual(self.atoms, self.hydrogen_bonds.atoms)
        self.assertTrue(self.hydrogen_bonds.periodic)
        npt.assert_array_equal(self.unit_cell_angles, self.hydrogen_bonds.unit_cell_angles)
        npt.assert_array_equal(self.unit_cell_vectors, self.hydrogen_bonds.unit_cell_vectors)
        self.assertEqual('test', self.hydrogen_bonds._system_name)
        self.assertEqual(self.number_of_frames, self.hydrogen_bonds.number_of_frames)
        self.assertEqual(self.hydrogen_bonds.core_units, cpu_count())

        self.assertTupleEqual((self.donor_atom_01, self.donor_atom_02), self.hydrogen_bonds.donor_atoms)
        self.assertTupleEqual((self.acceptor_01, self.acceptor_02), self.hydrogen_bonds.acceptors)

        self.assertIsNone(self.donor_01.hydrogen_bond_partners)
        self.assertIsNone(self.donor_02.hydrogen_bond_partners)
        self.assertDictEqual({'test': [[], [], []]}, self.donor_atom_01.hydrogen_bond_partners)
        self.assertDictEqual({'test': [[], [], []]}, self.donor_atom_02.hydrogen_bond_partners)
        self.assertDictEqual({'test': [[], [], []]}, self.acceptor_01.hydrogen_bond_partners)
        self.assertDictEqual({'test': [[], [], []]}, self.acceptor_02.hydrogen_bond_partners)

    def test_init_with_core_units(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds

        self.hydrogen_bonds = HydrogenBonds(atoms=self.atoms, periodic=True,
                                            unit_cell_angles=self.unit_cell_angles,
                                            unit_cell_vectors=self.unit_cell_vectors, system_name='test',
                                            number_of_frames=self.number_of_frames, core_units=6)

        self.assertEqual(self.hydrogen_bonds.core_units, 6)


class TestBuildMethods(HydrogenBondTestCase):
    def setUp(self) -> None:
        super(TestBuildMethods, self).setUp()

        self.exp = [np.array([True, False, False]), np.array([False, False, False]), np.array([True, True, True]),
                    np.array([True, False, False])]

    def test_build_triplets(self):
        triplets = self.hydrogen_bonds.__build_triplets__(distance_cutoff=0.25, angle_cutoff=2.)

        self.assertEqual(len(self.exp), len(triplets))

        for triplet_exp, triplet_res in zip(self.exp, triplets):
            npt.assert_array_equal(triplet_exp, triplet_res.mask)


class TestBuildMethodsMultiProcessing(HydrogenBondMultiProcessTestCase, TestBuildMethods):
    def setUp(self) -> None:
        super(TestBuildMethodsMultiProcessing, self).setUp()

        exp_donor_01_masks = [np.repeat(np.array([True, False, False]), repeats=self.trajectory_multiplier),
                              np.repeat(np.array([False, False, False]), repeats=self.trajectory_multiplier)]
        exp_donor_02_masks = [np.repeat(np.array([True, True, True]), repeats=self.trajectory_multiplier),
                              np.repeat(np.array([True, False, False]), repeats=self.trajectory_multiplier)]

        self.exp = []
        for i in range(self.atom_multiplier):
            self.exp.extend(exp_donor_01_masks * self.atom_multiplier)
            self.exp.extend(exp_donor_02_masks * self.atom_multiplier)


class HydrogenBondMethodWithCustomClassTestCase(HydrogenBondTestCase):
    def setUpClassMethod(self):
        return None

    def setUp(self) -> None:
        super(HydrogenBondMethodWithCustomClassTestCase, self).setUp()

        hydrogen_bonds_class = self.setUpClassMethod()

        self.hydrogen_bonds = hydrogen_bonds_class(atoms=self.atoms, periodic=True,
                                                   unit_cell_angles=self.unit_cell_angles,
                                                   unit_cell_vectors=self.unit_cell_vectors, system_name='test_system',
                                                   number_of_frames=self.number_of_frames)


class ObtainHydrogenBondsTestCase(HydrogenBondMethodWithCustomClassTestCase):
    def setUpClassMethod(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds

        class TestClass(HydrogenBonds):
            def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
                if frame == 0:
                    triplet = triplets[1]
                    triplet.acceptor.hydrogen_bond_partners[self._system_name][0].append(triplet.donor_atom)
                    triplet.donor_atom.hydrogen_bond_partners[self._system_name][0].append(triplet.acceptor)

                if frame == 1:
                    triplet = triplets[0]
                    triplet.acceptor.hydrogen_bond_partners[self._system_name][1].append(triplet.donor_atom)
                    triplet.donor_atom.hydrogen_bond_partners[self._system_name][1].append(triplet.acceptor)

        return TestClass


class TestObtainHydrogenBondsSimple(ObtainHydrogenBondsTestCase):
    def setUp(self) -> None:
        super(TestObtainHydrogenBondsSimple, self).setUp()

        self.exp_donor_atom_01_hydrogen_bond_partners = [[] for i in range(3)]
        self.exp_acceptor_01_hydrogen_bond_partners = [[] for i in range(3)]
        self.exp_donor_atom_02_hydrogen_bond_partners = [[] for i in range(3)]
        self.exp_acceptor_02_hydrogen_bond_partners = [[] for i in range(3)]

    def test_get_hydrogen_bonds(self):
        self.hydrogen_bonds.__get_hydrogen_bonds__(triplets=self.triplets)

        # prepare expected outcome
        self.exp_donor_atom_01_hydrogen_bond_partners[1].append(self.acceptor_01)
        self.exp_acceptor_01_hydrogen_bond_partners[1].append(self.donor_atom_01)
        self.exp_donor_atom_02_hydrogen_bond_partners[0].append(self.acceptor_02)
        self.exp_acceptor_02_hydrogen_bond_partners[0].append(self.donor_atom_02)

        # test
        self.assertIsNone(self.donor_01.hydrogen_bond_partners)
        self.assertListEqual(self.exp_donor_atom_01_hydrogen_bond_partners,
                             self.donor_atom_01.hydrogen_bond_partners['test_system'])
        self.assertListEqual(self.exp_acceptor_01_hydrogen_bond_partners,
                             self.acceptor_01.hydrogen_bond_partners['test_system'])
        self.assertIsNone(self.donor_02.hydrogen_bond_partners)
        self.assertListEqual(self.exp_donor_atom_02_hydrogen_bond_partners,
                             self.donor_atom_02.hydrogen_bond_partners['test_system'])
        self.assertListEqual(self.exp_acceptor_02_hydrogen_bond_partners,
                             self.acceptor_02.hydrogen_bond_partners['test_system'])

    def test_calculate_hydrogen_bonds(self):
        self.hydrogen_bonds.calculate_hydrogen_bonds(distance_cutoff=0.25, angle_cutoff=2.0)

        # prepare expected outcome
        self.exp_donor_atom_01_hydrogen_bond_partners[0].append(self.acceptor_02)
        self.exp_donor_atom_01_hydrogen_bond_partners[1].append(self.acceptor_01)
        self.exp_acceptor_01_hydrogen_bond_partners[1].append(self.donor_atom_01)
        self.exp_acceptor_02_hydrogen_bond_partners[0].append(self.donor_atom_01)

        # test
        self.assertIsNone(self.donor_01.hydrogen_bond_partners)
        self.assertListEqual(self.exp_donor_atom_01_hydrogen_bond_partners,
                             self.donor_atom_01.hydrogen_bond_partners['test_system'])
        self.assertListEqual(self.exp_acceptor_01_hydrogen_bond_partners,
                             self.acceptor_01.hydrogen_bond_partners['test_system'])
        self.assertIsNone(self.donor_02.hydrogen_bond_partners)
        self.assertListEqual(self.exp_donor_atom_02_hydrogen_bond_partners,
                             self.donor_atom_02.hydrogen_bond_partners['test_system'])
        self.assertListEqual(self.exp_acceptor_02_hydrogen_bond_partners,
                             self.acceptor_02.hydrogen_bond_partners['test_system'])


class TestObtainHydrogenBondsForceMultiThreadQueue(TestObtainHydrogenBondsSimple, HydrogenBondMultiProcessTestCase):
    def setUpMultipliers(self):
        self.atom_multiplier = 1
        self.trajectory_multiplier = 200

    def setUp(self) -> None:
        super(TestObtainHydrogenBondsForceMultiThreadQueue, self).setUp()

        self.donor_01 = self.atoms[0]
        self.donor_atom_01 = self.atoms[1]
        self.acceptor_01 = self.atoms[2]
        self.donor_02 = self.atoms[3]
        self.donor_atom_02 = self.atoms[4]
        self.acceptor_02 = self.atoms[5]

        self.triplets[0].donor = self.donor_01
        self.triplets[0].donor_atom = self.donor_atom_01
        self.triplets[0].acceptor = self.acceptor_01

        self.triplets[1].donor = self.donor_02
        self.triplets[1].donor_atom = self.donor_atom_02
        self.triplets[1].acceptor = self.acceptor_02

        hydrogen_bonds_class = self.setUpClassMethod()

        self.hydrogen_bonds = hydrogen_bonds_class(atoms=self.atoms, periodic=True,
                                                   unit_cell_angles=self.unit_cell_angles,
                                                   unit_cell_vectors=self.unit_cell_vectors, system_name='test_system',
                                                   number_of_frames=self.number_of_frames * self.trajectory_multiplier)

        self.exp_donor_atom_01_hydrogen_bond_partners = [[] for i in range(3 * self.trajectory_multiplier)]
        self.exp_acceptor_01_hydrogen_bond_partners = [[] for i in range(3 * self.trajectory_multiplier)]
        self.exp_donor_atom_02_hydrogen_bond_partners = [[] for i in range(3 * self.trajectory_multiplier)]
        self.exp_acceptor_02_hydrogen_bond_partners = [[] for i in range(3 * self.trajectory_multiplier)]

    def test_get_hydrogen_bonds(self):
        super(TestObtainHydrogenBondsForceMultiThreadQueue, self).test_get_hydrogen_bonds()


class HydrogenBondRepresentationMethodWithCustomClassTestCase(HydrogenBondMethodWithCustomClassTestCase):
    def setUpClassMethod(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds

        class TestClass(HydrogenBonds):
            def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
                for triplet in triplets:
                    if triplet.mask[frame]:
                        triplet.acceptor.hydrogen_bond_partners[self._system_name][frame].append(triplet.donor_atom)
                        triplet.donor_atom.hydrogen_bond_partners[self._system_name][frame].append(triplet.acceptor)

        return TestClass

    def setUp(self) -> None:
        super(HydrogenBondRepresentationMethodWithCustomClassTestCase, self).setUp()

        self.hydrogen_bonds.calculate_hydrogen_bonds(distance_cutoff=0.25, angle_cutoff=2.0)

        first_frame = np.zeros((4, 4), dtype=int)
        first_frame[0, 2] = 1
        first_frame[2, 0] = 1
        first_frame[1, 2] = 1
        first_frame[2, 1] = 1
        first_frame[1, 3] = 1
        first_frame[3, 1] = 1

        second_frame = np.zeros((4, 4), dtype=int)
        second_frame[1, 2] = 1
        second_frame[2, 1] = 1

        third_frame = np.zeros((4, 4), dtype=int)
        third_frame[1, 2] = 1
        third_frame[2, 1] = 1

        self.exp = np.array([first_frame, second_frame, third_frame])

    def test_get_hydrogen_bond_matrix_in_frame(self):
        index_dictionary = {atom.structure_file_index: i for i, atom in
                            enumerate((self.donor_atom_01, self.donor_atom_02, self.acceptor_01, self.acceptor_02))}
        res = self.hydrogen_bonds.__get_hydrogen_bond_matrix_in_frame__(index_dictionary=index_dictionary, frame=0)

        npt.assert_array_equal(self.exp[0], res)

    def test_get_hydrogen_bond_matrix(self):
        res = self.hydrogen_bonds.get_hydrogen_bond_matrix()

        npt.assert_array_equal(self.exp, res)

    def test_get_number_hydrogen_bonds_for_frame(self):
        res = self.hydrogen_bonds.__get_number_hydrogen_bonds_for_frame__(frame=0)

        self.assertEqual(res, 3)

    def test_get_frame_wise_hydrogen_bonds(self):
        res = self.hydrogen_bonds.get_frame_wise_hydrogen_bonds()

        npt.assert_array_equal(res, np.array([3, 1, 1]))


class HydrogenBondExceptionsTestCase(HydrogenBondTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.get_features.hydrogen_bonds import HydrogenBondsException

        super(HydrogenBondExceptionsTestCase, self).setUp()

        self.exception = HydrogenBondsException


class TestStandardMethodExceptions(HydrogenBondExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.get_features.hydrogen_bonds import HydrogenBonds

        super(TestStandardMethodExceptions, self).setUp()
        self.hydrogen_bonds = HydrogenBonds

    def test_init_atoms_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds(atoms=[], periodic=True, unit_cell_angles=self.unit_cell_angles,
                                unit_cell_vectors=self.unit_cell_vectors, system_name='test',
                                number_of_frames=self.number_of_frames)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atoms', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_periodic_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds(atoms=self.atoms, periodic=42, unit_cell_angles=self.unit_cell_angles,
                                unit_cell_vectors=self.unit_cell_vectors, system_name='test',
                                number_of_frames=self.number_of_frames)

        desired_msg = self.create_data_type_exception_messages(parameter_name='periodic', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds(atoms=self.atoms, periodic=True, unit_cell_angles=(),
                                unit_cell_vectors=self.unit_cell_vectors, system_name='test',
                                number_of_frames=self.number_of_frames)

        desired_msg = self.create_data_type_exception_messages(parameter_name='unit_cell_angles',
                                                               data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds(atoms=self.atoms, periodic=True, unit_cell_angles=self.unit_cell_angles,
                                unit_cell_vectors=[], system_name='test',
                                number_of_frames=self.number_of_frames)

        desired_msg = self.create_data_type_exception_messages(parameter_name='unit_cell_vectors',
                                                               data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_dtype(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds(atoms=self.atoms, periodic=True,
                                unit_cell_angles=self.unit_cell_angles.astype(np.float64),
                                unit_cell_vectors=self.unit_cell_vectors, system_name='test',
                                number_of_frames=self.number_of_frames)

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='unit_cell_angles',
                                                                 dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_dtype(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds(atoms=self.atoms, periodic=True, unit_cell_angles=self.unit_cell_angles,
                                unit_cell_vectors=self.unit_cell_vectors.astype(np.int), system_name='test',
                                number_of_frames=self.number_of_frames)

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='unit_cell_vectors',
                                                                 dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_shape(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds(atoms=self.atoms, periodic=True,
                                unit_cell_angles=np.array([[90, 90], [90, 90], [90, 90]], dtype=np.float32),
                                unit_cell_vectors=self.unit_cell_vectors, system_name='test',
                                number_of_frames=self.number_of_frames)

        desired_msg = self.create_array_shape_exception_messages(parameter_name='unit_cell_angles',
                                                                 desired_shape=(None, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_shape(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds(atoms=self.atoms, periodic=True, unit_cell_angles=self.unit_cell_angles,
                                unit_cell_vectors=np.array([[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]], dtype=np.float32),
                                system_name='test', number_of_frames=self.number_of_frames)

        desired_msg = self.create_array_shape_exception_messages(parameter_name='unit_cell_vectors',
                                                                 desired_shape=(None, 3, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_system_name_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds(atoms=self.atoms, periodic=True, unit_cell_angles=self.unit_cell_angles,
                                unit_cell_vectors=self.unit_cell_vectors, system_name=1.2,
                                number_of_frames=self.number_of_frames)

        desired_msg = self.create_data_type_exception_messages(parameter_name='system_name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_number_of_frames_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds(atoms=self.atoms, periodic=True, unit_cell_angles=self.unit_cell_angles,
                                unit_cell_vectors=self.unit_cell_vectors, system_name='test',
                                number_of_frames=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='number_of_frames', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))


class TestBuildMethodExceptions(HydrogenBondExceptionsTestCase):
    def test_build_triplets_distance_cutoff_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds.__build_triplets__(distance_cutoff=True, angle_cutoff=2.)

        desired_msg = self.create_data_type_exception_messages(parameter_name='distance_cutoff', data_type_name='float')
        self.assertEqual(desired_msg, str(context.exception))

    def test_build_triplets_angle_cutoff_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds.__build_triplets__(distance_cutoff=0.25, angle_cutoff=2)

        desired_msg = self.create_data_type_exception_messages(parameter_name='angle_cutoff', data_type_name='float')
        self.assertEqual(desired_msg, str(context.exception))


class TestObtainHydrogenBondMethodExceptions(HydrogenBondExceptionsTestCase):
    def test_get_hydrogen_bonds_triplets_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds.__get_hydrogen_bonds__(triplets=list(self.triplets))

        desired_msg = self.create_data_type_exception_messages(parameter_name='triplets', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_hydrogen_bonds_distance_cutoff_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds.calculate_hydrogen_bonds(distance_cutoff=[], angle_cutoff=2.0)

        desired_msg = self.create_data_type_exception_messages(parameter_name='distance_cutoff', data_type_name='float')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_hydrogen_bonds_angle_cutoff_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds.calculate_hydrogen_bonds(distance_cutoff=0.25, angle_cutoff=2)

        desired_msg = self.create_data_type_exception_messages(parameter_name='angle_cutoff', data_type_name='float')
        self.assertEqual(desired_msg, str(context.exception))


class TestHydrogenBondRepresentationMethodExceptions(HydrogenBondExceptionsTestCase):
    def setUp(self) -> None:
        super(TestHydrogenBondRepresentationMethodExceptions, self).setUp()

        self.index_dictionary = {atom.structure_file_index: atom.subsystem_index for atom in self.atoms}

    def test_get_hydrogen_bond_matrix_in_frame_index_dictionary_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds.__get_hydrogen_bond_matrix_in_frame__(index_dictionary=[], frame=0)

        desired_msg = self.create_data_type_exception_messages(parameter_name='index_dictionary', data_type_name='dict')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_hydrogen_bond_matrix_in_frame_frame_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds.__get_hydrogen_bond_matrix_in_frame__(index_dictionary=self.index_dictionary, frame=0.)

        desired_msg = self.create_data_type_exception_messages(parameter_name='frame', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_number_hydrogen_bonds_for_frame_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds.__get_number_hydrogen_bonds_for_frame__(frame=1.)

        desired_msg = self.create_data_type_exception_messages(parameter_name='frame', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
