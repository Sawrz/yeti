import unittest

import numpy as np
import numpy.testing as npt

from test_utils.test_utils import build_unit_cell_angles_and_vectors, build_atom_triplet, \
    create_data_type_exception_messages, create_array_shape_exception_messages, create_array_dtype_exception_messages, \
    build_triplet, build_multi_atom_triplets


class HydrogenBondsTest(unittest.TestCase):
    def test_init(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds
        atoms = build_atom_triplet()
        number_of_frames = atoms[0].xyz_trajectory.shape[0]

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                       unit_cell_vectors=unit_cell_vectors, system_name='test',
                                       number_of_frames=number_of_frames)

        self.assertTupleEqual(atoms, hydrogen_bonds.atoms)
        self.assertTrue(hydrogen_bonds.periodic)
        npt.assert_array_equal(unit_cell_angles, hydrogen_bonds.unit_cell_angles)
        npt.assert_array_equal(unit_cell_vectors, hydrogen_bonds.unit_cell_vectors)
        self.assertEqual('test', hydrogen_bonds._system_name)
        self.assertEqual(number_of_frames, hydrogen_bonds.number_of_frames)

        self.assertTupleEqual((atoms[1],), hydrogen_bonds.donor_atoms)
        self.assertTupleEqual((atoms[2],), hydrogen_bonds.acceptors)

    def test_init_atoms_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_atom_triplet()
        number_of_frames = atoms[0].xyz_trajectory.shape[0]
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            HydrogenBonds(atoms=[], periodic=True, unit_cell_angles=unit_cell_angles,
                          unit_cell_vectors=unit_cell_vectors, system_name='test',
                          number_of_frames=number_of_frames)

        desired_msg = create_data_type_exception_messages(parameter_name='atoms', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_periodic_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_atom_triplet()
        number_of_frames = atoms[0].xyz_trajectory.shape[0]
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            HydrogenBonds(atoms=atoms, periodic=42, unit_cell_angles=unit_cell_angles,
                          unit_cell_vectors=unit_cell_vectors, system_name='test',
                          number_of_frames=number_of_frames)

        desired_msg = create_data_type_exception_messages(parameter_name='periodic', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_atom_triplet()
        number_of_frames = atoms[0].xyz_trajectory.shape[0]
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=(),
                          unit_cell_vectors=unit_cell_vectors, system_name='test',
                          number_of_frames=number_of_frames)

        desired_msg = create_data_type_exception_messages(parameter_name='unit_cell_angles', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_atom_triplet()
        number_of_frames = atoms[0].xyz_trajectory.shape[0]
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                          unit_cell_vectors=[], system_name='test',
                          number_of_frames=number_of_frames)

        desired_msg = create_data_type_exception_messages(parameter_name='unit_cell_vectors', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_dtype(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_atom_triplet()
        number_of_frames = atoms[0].xyz_trajectory.shape[0]
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)
        unit_cell_angles = unit_cell_angles.astype(np.float64)

        with self.assertRaises(HydrogenBondsException) as context:
            HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                          unit_cell_vectors=unit_cell_vectors, system_name='test',
                          number_of_frames=number_of_frames)

        desired_msg = create_array_dtype_exception_messages(parameter_name='unit_cell_angles', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_dtype(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_atom_triplet()
        number_of_frames = atoms[0].xyz_trajectory.shape[0]
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)
        unit_cell_vectors = unit_cell_vectors.astype(np.int)

        with self.assertRaises(HydrogenBondsException) as context:
            HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                          unit_cell_vectors=unit_cell_vectors, system_name='test',
                          number_of_frames=number_of_frames)

        desired_msg = create_array_dtype_exception_messages(parameter_name='unit_cell_vectors', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_shape(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_atom_triplet()
        number_of_frames = atoms[0].xyz_trajectory.shape[0]
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_vectors,
                          unit_cell_vectors=unit_cell_vectors, system_name='test',
                          number_of_frames=number_of_frames)

        desired_msg = create_array_shape_exception_messages(parameter_name='unit_cell_angles', desired_shape=(None, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_shape(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_atom_triplet()
        number_of_frames = atoms[0].xyz_trajectory.shape[0]
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                          unit_cell_vectors=unit_cell_angles, system_name='test',
                          number_of_frames=number_of_frames)

        desired_msg = create_array_shape_exception_messages(parameter_name='unit_cell_vectors',
                                                            desired_shape=(None, 3, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_system_name_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_atom_triplet()
        number_of_frames = atoms[0].xyz_trajectory.shape[0]
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                          unit_cell_vectors=unit_cell_vectors, system_name=1.2,
                          number_of_frames=number_of_frames)

        desired_msg = create_data_type_exception_messages(parameter_name='system_name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_number_of_frames_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_atom_triplet()
        number_of_frames = atoms[0].xyz_trajectory.shape[0]
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                          unit_cell_vectors=unit_cell_vectors, system_name='test',
                          number_of_frames=True)

        desired_msg = create_data_type_exception_messages(parameter_name='number_of_frames', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_build_triplets(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds

        triplet_01 = build_atom_triplet()
        triplet_02 = build_atom_triplet()
        triplet_02[0].name = 'D'
        triplet_02[0].xyz_trajectory += 0.2
        triplet_02[1].name = 'E'
        triplet_02[1].xyz_trajectory += 0.2
        triplet_02[2].name = 'F'
        triplet_02[2].xyz_trajectory += 0.2

        atoms = (*triplet_01, *triplet_02)
        number_of_frames = atoms[0].xyz_trajectory.shape[0]

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                       unit_cell_vectors=unit_cell_vectors, system_name='test',
                                       number_of_frames=number_of_frames)

        triplets = hydrogen_bonds.__build_triplets__(distance_cutoff=0.25, angle_cutoff=2.)

        exp = [np.array([True, False, False]), np.array([False, False, False]), np.array([True, True, True]),
               np.array([True, False, False])]

        self.assertEqual(len(exp), len(triplets))

        for triplet_exp, triplet_res in zip(exp, triplets):
            npt.assert_array_equal(triplet_exp, triplet_res.mask)

    def test_build_triplets_distance_cutoff_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = (*build_atom_triplet(), *build_atom_triplet())
        number_of_frames = atoms[0].xyz_trajectory.shape[0]

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                       unit_cell_vectors=unit_cell_vectors, system_name='test',
                                       number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            hydrogen_bonds.__build_triplets__(distance_cutoff=True, angle_cutoff=2.)

        desired_msg = create_data_type_exception_messages(parameter_name='distance_cutoff', data_type_name='float')
        self.assertEqual(desired_msg, str(context.exception))

    def test_build_triplets_angle_cutoff_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = (*build_atom_triplet(), *build_atom_triplet())
        number_of_frames = atoms[0].xyz_trajectory.shape[0]

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                       unit_cell_vectors=unit_cell_vectors, system_name='test',
                                       number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            hydrogen_bonds.__build_triplets__(distance_cutoff=0.25, angle_cutoff=2)

        desired_msg = create_data_type_exception_messages(parameter_name='angle_cutoff', data_type_name='float')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_hydrogen_bonds(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds

        class TestHydrogenBonds(HydrogenBonds):
            def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
                if frame == 0:
                    triplet = triplets[1]
                    triplet.acceptor.hydrogen_bond_partners[self._system_name][0].append(triplet.donor_atom)
                    triplet.donor_atom.hydrogen_bond_partners[self._system_name][0].append(triplet.acceptor)

                if frame == 1:
                    triplet = triplets[0]
                    triplet.acceptor.hydrogen_bond_partners[self._system_name][1].append(triplet.donor_atom)
                    triplet.donor_atom.hydrogen_bond_partners[self._system_name][1].append(triplet.acceptor)

        triplets = (build_triplet(), build_triplet())
        atoms = (*triplets[0].triplet, *triplets[1].triplet)
        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = TestHydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                           unit_cell_vectors=unit_cell_vectors, system_name='subsystem',
                                           number_of_frames=number_of_frames)

        hydrogen_bonds.__get_hydrogen_bonds__(triplets=triplets)

        self.assertIsNone(atoms[0].hydrogen_bond_partners)
        self.assertListEqual([[], [atoms[2]], []], atoms[1].hydrogen_bond_partners['subsystem'])
        self.assertListEqual([[], [atoms[1]], []], atoms[2].hydrogen_bond_partners['subsystem'])
        self.assertIsNone(atoms[3].hydrogen_bond_partners)
        self.assertListEqual([[atoms[5]], [], []], atoms[4].hydrogen_bond_partners['subsystem'])
        self.assertListEqual([[atoms[4]], [], []], atoms[5].hydrogen_bond_partners['subsystem'])

    def test_get_hydrogen_bonds_triplets_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        triplets = [build_triplet(), build_triplet()]
        atoms = (*triplets[0].triplet, *triplets[1].triplet)
        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                       unit_cell_vectors=unit_cell_vectors, system_name='subsystem',
                                       number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            hydrogen_bonds.__get_hydrogen_bonds__(triplets=triplets)

        desired_msg = create_data_type_exception_messages(parameter_name='triplets', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_hydrogen_bonds(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds

        class TestHydrogenBonds(HydrogenBonds):
            def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
                if frame == 0:
                    triplet = triplets[1]
                    triplet.acceptor.hydrogen_bond_partners[self._system_name][0].append(triplet.donor_atom)
                    triplet.donor_atom.hydrogen_bond_partners[self._system_name][0].append(triplet.acceptor)

                if frame == 1:
                    triplet = triplets[0]
                    triplet.acceptor.hydrogen_bond_partners[self._system_name][1].append(triplet.donor_atom)
                    triplet.donor_atom.hydrogen_bond_partners[self._system_name][1].append(triplet.acceptor)

        atoms = build_multi_atom_triplets(amount=2)
        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = TestHydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                           unit_cell_vectors=unit_cell_vectors, system_name='subsystem',
                                           number_of_frames=number_of_frames)

        hydrogen_bonds.calculate_hydrogen_bonds(distance_cutoff=0.25, angle_cutoff=2.0)

        self.assertIsNone(atoms[0].hydrogen_bond_partners)
        self.assertListEqual([[atoms[5]], [atoms[2]], []], atoms[1].hydrogen_bond_partners['subsystem'])
        self.assertListEqual([[], [atoms[1]], []], atoms[2].hydrogen_bond_partners['subsystem'])
        self.assertIsNone(atoms[3].hydrogen_bond_partners)
        self.assertListEqual([[], [], []], atoms[4].hydrogen_bond_partners['subsystem'])
        self.assertListEqual([[atoms[1]], [], []], atoms[5].hydrogen_bond_partners['subsystem'])

    def test_calculate_hydrogen_bonds_distance_cutoff_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_multi_atom_triplets(amount=2)
        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                       unit_cell_vectors=unit_cell_vectors, system_name='subsystem',
                                       number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            hydrogen_bonds.calculate_hydrogen_bonds(distance_cutoff=[], angle_cutoff=2.0)

        desired_msg = create_data_type_exception_messages(parameter_name='distance_cutoff', data_type_name='float')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_hydrogen_bonds_angle_cutoff_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_multi_atom_triplets(amount=2)
        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                       unit_cell_vectors=unit_cell_vectors, system_name='subsystem',
                                       number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            hydrogen_bonds.calculate_hydrogen_bonds(distance_cutoff=0.25, angle_cutoff=2)

        desired_msg = create_data_type_exception_messages(parameter_name='angle_cutoff', data_type_name='float')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_hydrogen_bond_matrix_in_frame(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds

        class TestHydrogenBonds(HydrogenBonds):
            def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
                for triplet in triplets:
                    if triplet.mask[frame]:
                        triplet.acceptor.hydrogen_bond_partners[self._system_name][frame].append(triplet.donor_atom)
                        triplet.donor_atom.hydrogen_bond_partners[self._system_name][frame].append(triplet.acceptor)

        atoms = build_multi_atom_triplets(amount=2)
        index_dictionary = {atom.structure_file_index: atom.subsystem_index for atom in atoms}

        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = TestHydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                           unit_cell_vectors=unit_cell_vectors, system_name='subsystem',
                                           number_of_frames=number_of_frames)

        hydrogen_bonds.calculate_hydrogen_bonds(distance_cutoff=0.25, angle_cutoff=2.0)
        res = hydrogen_bonds.__get_hydrogen_bond_matrix_in_frame__(index_dictionary=index_dictionary, frame=0)

        exp = np.zeros((6, 6))
        exp[1, 2] = 1
        exp[1, 5] = 1
        exp[2, 1] = 1
        exp[2, 4] = 1
        exp[4, 2] = 1
        exp[4, 5] = 1
        exp[5, 1] = 1
        exp[5, 4] = 1

        npt.assert_array_equal(exp, res)

    def test_get_hydrogen_bond_matrix_in_frame_index_dictionary_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_multi_atom_triplets(amount=2)
        number_of_frames = 3
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                       unit_cell_vectors=unit_cell_vectors, system_name='subsystem',
                                       number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            hydrogen_bonds.__get_hydrogen_bond_matrix_in_frame__(index_dictionary=[], frame=0)

        desired_msg = create_data_type_exception_messages(parameter_name='index_dictionary', data_type_name='dict')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_hydrogen_bond_matrix_in_frame_frame_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_multi_atom_triplets(amount=2)
        index_dictionary = {atom.structure_file_index: atom.subsystem_index for atom in atoms}

        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                       unit_cell_vectors=unit_cell_vectors, system_name='subsystem',
                                       number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            hydrogen_bonds.__get_hydrogen_bond_matrix_in_frame__(index_dictionary=index_dictionary, frame=0.)

        desired_msg = create_data_type_exception_messages(parameter_name='frame', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_hydrogen_bond_matrix(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds

        class TestHydrogenBonds(HydrogenBonds):
            def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
                for triplet in triplets:
                    if triplet.mask[frame]:
                        triplet.acceptor.hydrogen_bond_partners[self._system_name][frame].append(triplet.donor_atom)
                        triplet.donor_atom.hydrogen_bond_partners[self._system_name][frame].append(triplet.acceptor)

        atoms = build_multi_atom_triplets(amount=2)
        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = TestHydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                           unit_cell_vectors=unit_cell_vectors, system_name='subsystem',
                                           number_of_frames=number_of_frames)

        hydrogen_bonds.calculate_hydrogen_bonds(distance_cutoff=0.25, angle_cutoff=2.0)
        res = hydrogen_bonds.get_hydrogen_bond_matrix()

        first_frame = np.zeros((6, 6))
        first_frame[1, 2] = 1
        first_frame[1, 5] = 1
        first_frame[2, 1] = 1
        first_frame[2, 4] = 1
        first_frame[4, 2] = 1
        first_frame[4, 5] = 1
        first_frame[5, 1] = 1
        first_frame[5, 4] = 1

        second_frame = np.zeros((6, 6))
        third_frame = np.zeros((6, 6))

        npt.assert_array_equal(np.array([first_frame, second_frame, third_frame]), res)

    def test_get_number_hydrogen_bonds_for_frame(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds

        class TestHydrogenBonds(HydrogenBonds):
            def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
                for triplet in triplets:
                    if triplet.mask[frame]:
                        triplet.acceptor.hydrogen_bond_partners[self._system_name][frame].append(triplet.donor_atom)
                        triplet.donor_atom.hydrogen_bond_partners[self._system_name][frame].append(triplet.acceptor)

        atoms = build_multi_atom_triplets(amount=2)
        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = TestHydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                           unit_cell_vectors=unit_cell_vectors, system_name='subsystem',
                                           number_of_frames=number_of_frames)

        hydrogen_bonds.calculate_hydrogen_bonds(distance_cutoff=0.25, angle_cutoff=2.0)
        res = hydrogen_bonds.__get_number_hydrogen_bonds_for_frame__(frame=0)

        self.assertEqual(res, 4)

    def test_get_number_hydrogen_bonds_for_frame_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds, HydrogenBondsException

        atoms = build_multi_atom_triplets(amount=2)
        number_of_frames = 3
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                       unit_cell_vectors=unit_cell_vectors, system_name='subsystem',
                                       number_of_frames=number_of_frames)

        with self.assertRaises(HydrogenBondsException) as context:
            hydrogen_bonds.__get_number_hydrogen_bonds_for_frame__(frame=1.)

        desired_msg = create_data_type_exception_messages(parameter_name='frame', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_frame_wise_hydrogen_bonds(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBonds

        class TestHydrogenBonds(HydrogenBonds):
            def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
                for triplet in triplets:
                    if triplet.mask[frame]:
                        triplet.acceptor.hydrogen_bond_partners[self._system_name][frame].append(triplet.donor_atom)
                        triplet.donor_atom.hydrogen_bond_partners[self._system_name][frame].append(triplet.acceptor)

        atoms = build_multi_atom_triplets(amount=2)
        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = TestHydrogenBonds(atoms=atoms, periodic=True, unit_cell_angles=unit_cell_angles,
                                           unit_cell_vectors=unit_cell_vectors, system_name='subsystem',
                                           number_of_frames=number_of_frames)

        hydrogen_bonds.calculate_hydrogen_bonds(distance_cutoff=0.25, angle_cutoff=2.0)
        res = hydrogen_bonds.get_frame_wise_hydrogen_bonds()

        npt.assert_array_equal(res, np.array([4, 0, 0]))


if __name__ == '__main__':
    unittest.main()
