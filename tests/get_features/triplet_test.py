import unittest

import numpy as np
import numpy.testing as npt

from test_utils.test_utils import build_unit_cell_angles_and_vectors, create_data_type_exception_messages, \
    create_array_shape_exception_messages, create_array_dtype_exception_messages, build_atom_triplet
from yeti.systems.building_blocks import Atom


class TestTriplet(unittest.TestCase):
    def test_init(self):
        from yeti.get_features.hydrogen_bonds import Triplet

        donor, donor_atom, acceptor = build_atom_triplet()
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])
        triplet = Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=True, unit_cell_angles=unit_cell_angles,
                          unit_cell_vectors=unit_cell_vectors)

        self.assertTrue(triplet.periodic)
        npt.assert_array_equal(triplet.unit_cell_vectors, unit_cell_vectors)
        npt.assert_array_equal(triplet.unit_cell_angles, unit_cell_angles)
        self.assertEqual(triplet.donor, donor)
        self.assertEqual(triplet.donor_atom, donor_atom)
        self.assertEqual(triplet.acceptor, acceptor)
        self.assertTupleEqual(triplet.triplet, (donor, donor_atom, acceptor))
        self.assertIsNone(triplet.mask)

    def test_init_donor_atom_too_many_covalent_bonds(self):
        from yeti.get_features.hydrogen_bonds import Triplet, TripletException

        donor, donor_atom, acceptor = build_atom_triplet()
        eve_atom = Atom(structure_file_index=3, subsystem_index=3, name='E', xyz_trajectory=np.array([[0.1, 0.1, 0.2]]))
        eve_atom.add_covalent_bond(atom=donor_atom)

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])

        with self.assertRaises(TripletException) as context:
            Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=True, unit_cell_angles=unit_cell_angles,
                    unit_cell_vectors=unit_cell_vectors)

        desired_msg = 'Donor atom has more than one covalent bond. That violates the assumption of this method. ' \
                      'Please contact the developer.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_donor_atom_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import Triplet, TripletException

        donor, donor_atom, acceptor = build_atom_triplet()
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])

        with self.assertRaises(TripletException) as context:
            Triplet(donor_atom=42, acceptor=acceptor, periodic=True, unit_cell_angles=unit_cell_angles,
                    unit_cell_vectors=unit_cell_vectors)

        desired_msg = create_data_type_exception_messages(parameter_name='donor_atom', data_type_name='Atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_acceptor_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import Triplet, TripletException

        donor, donor_atom, acceptor = build_atom_triplet()
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])

        with self.assertRaises(TripletException) as context:
            Triplet(donor_atom=donor_atom, acceptor=1.2, periodic=True, unit_cell_angles=unit_cell_angles,
                    unit_cell_vectors=unit_cell_vectors)

        desired_msg = create_data_type_exception_messages(parameter_name='acceptor', data_type_name='Atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_periodic_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import Triplet, TripletException

        donor, donor_atom, acceptor = build_atom_triplet()
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])

        with self.assertRaises(TripletException) as context:
            Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=[], unit_cell_angles=unit_cell_angles,
                    unit_cell_vectors=unit_cell_vectors)

        desired_msg = create_data_type_exception_messages(parameter_name='periodic', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import Triplet, TripletException

        donor, donor_atom, acceptor = build_atom_triplet()
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])

        with self.assertRaises(TripletException) as context:
            Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=True, unit_cell_angles=(),
                    unit_cell_vectors=unit_cell_vectors)

        desired_msg = create_data_type_exception_messages(parameter_name='unit_cell_angles', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import Triplet, TripletException

        donor, donor_atom, acceptor = build_atom_triplet()
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])

        with self.assertRaises(TripletException) as context:
            Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=True, unit_cell_angles=unit_cell_angles,
                    unit_cell_vectors=[])

        desired_msg = create_data_type_exception_messages(parameter_name='unit_cell_vectors', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_dtype(self):
        from yeti.get_features.hydrogen_bonds import Triplet, TripletException

        donor, donor_atom, acceptor = build_atom_triplet()
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])
        unit_cell_angles = unit_cell_angles.astype(np.float64)

        with self.assertRaises(TripletException) as context:
            Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=True, unit_cell_angles=unit_cell_angles,
                    unit_cell_vectors=unit_cell_vectors)

        desired_msg = create_array_dtype_exception_messages(parameter_name='unit_cell_angles', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_dtype(self):
        from yeti.get_features.hydrogen_bonds import Triplet, TripletException

        donor, donor_atom, acceptor = build_atom_triplet()
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])
        unit_cell_vectors = unit_cell_vectors.astype(np.float64)

        with self.assertRaises(TripletException) as context:
            Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=True, unit_cell_angles=unit_cell_angles,
                    unit_cell_vectors=unit_cell_vectors)

        desired_msg = create_array_dtype_exception_messages(parameter_name='unit_cell_vectors', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_shape(self):
        from yeti.get_features.hydrogen_bonds import Triplet, TripletException

        donor, donor_atom, acceptor = build_atom_triplet()
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])

        with self.assertRaises(TripletException) as context:
            Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=True, unit_cell_angles=unit_cell_vectors,
                    unit_cell_vectors=unit_cell_vectors)

        desired_msg = create_array_shape_exception_messages(parameter_name='unit_cell_angles', desired_shape=(None, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_shape(self):
        from yeti.get_features.hydrogen_bonds import Triplet, TripletException

        donor, donor_atom, acceptor = build_atom_triplet()
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])

        with self.assertRaises(TripletException) as context:
            Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=True, unit_cell_angles=unit_cell_angles,
                    unit_cell_vectors=unit_cell_angles)

        desired_msg = create_array_shape_exception_messages(parameter_name='unit_cell_vectors',
                                                            desired_shape=(None, 3, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_create_mask(self):
        from yeti.get_features.hydrogen_bonds import Triplet

        donor, donor_atom, acceptor = build_atom_triplet()
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])
        triplet = Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=True, unit_cell_angles=unit_cell_angles,
                          unit_cell_vectors=unit_cell_vectors)
        triplet.create_mask(distance_cutoff=0.25, angle_cutoff=2.)

        npt.assert_array_equal(np.array([True, False, False]), triplet.mask)

    def test_create_mask_distance_cutoff_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import Triplet, TripletException

        donor, donor_atom, acceptor = build_atom_triplet()
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])
        triplet = Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=True, unit_cell_angles=unit_cell_angles,
                          unit_cell_vectors=unit_cell_vectors)

        with self.assertRaises(TripletException) as context:
            triplet.create_mask(distance_cutoff=1, angle_cutoff=2.)

        desired_msg = create_data_type_exception_messages(parameter_name='distance_cutoff', data_type_name='float')
        self.assertEqual(desired_msg, str(context.exception))

    def test_create_mask_angle_cutoff_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import Triplet, TripletException

        donor, donor_atom, acceptor = build_atom_triplet()
        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(
            number_of_frames=donor.xyz_trajectory.shape[0])
        triplet = Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=True, unit_cell_angles=unit_cell_angles,
                          unit_cell_vectors=unit_cell_vectors)

        with self.assertRaises(TripletException) as context:
            triplet.create_mask(distance_cutoff=0.25, angle_cutoff=True)

        desired_msg = create_data_type_exception_messages(parameter_name='angle_cutoff', data_type_name='float')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
