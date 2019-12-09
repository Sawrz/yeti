import unittest

import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintTestCase, BlueprintExceptionsTestCase


class TripletTestCase(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.get_features.hydrogen_bonds import Triplet

        super(TripletTestCase, self).setUp()

        # CREATE ATOMS
        # 1st frame: h-bond, 2nd frame: no h-bond because of distance, 3rd no h-bond because of angle
        self.donor = Atom(structure_file_index=2, subsystem_index=0, name='A',
                          xyz_trajectory=np.array([[0.1, 0.4, 0.3], [0.1, 0.4, 0.3], [0.1, 0.4, 0.3]]))
        self.donor_atom = Atom(structure_file_index=3, subsystem_index=1, name='B',
                               xyz_trajectory=np.array([[0.1, 0.5, 0.2], [0.1, 0.5, 0.2], [0.5, 0.5, 0.2]]))
        self.acceptor = Atom(structure_file_index=4, subsystem_index=2, name='C',
                             xyz_trajectory=np.array([[0.1, 0.6, 0.4], [0.1, 0.7, 0.4], [0.1, 0.6, 0.4]]))

        self.donor.add_covalent_bond(atom=self.donor_atom)
        self.donor_atom.update_donor_state(is_donor_atom=True, donor_slots=1)
        self.acceptor.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        # CREATE UNIT CELL PROPERTIES
        self.unit_cell_angles = np.array([[90, 90, 90], [90, 90, 90], [90, 90, 90]], dtype=np.float32)
        self.unit_cell_vectors = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
            dtype=np.float32)

        self.triplet = Triplet(donor_atom=self.donor_atom, acceptor=self.acceptor, periodic=True,
                               unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)


class TestStandardMethods(TripletTestCase):
    def test_init(self):
        self.assertTrue(self.triplet.periodic)
        npt.assert_array_equal(self.triplet.unit_cell_vectors, np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
            dtype=np.float32))
        npt.assert_array_equal(self.triplet.unit_cell_angles,
                               np.array([[90, 90, 90], [90, 90, 90], [90, 90, 90]], dtype=np.float32))
        self.assertEqual(self.triplet.donor, self.donor)
        self.assertEqual(self.triplet.donor_atom, self.donor_atom)
        self.assertEqual(self.triplet.acceptor, self.acceptor)
        self.assertTupleEqual(self.triplet.triplet, (self.donor, self.donor_atom, self.acceptor))
        self.assertEqual(self.triplet.mask_frame, 0)
        self.assertIsNone(self.triplet.mask)


class TestMaskMethods(TripletTestCase):
    def test_create_mask(self):
        self.triplet.create_mask(distance_cutoff=0.25, angle_cutoff=2.)

        npt.assert_array_equal(np.array([True, False, False]), self.triplet.mask)


class TripletExceptionsTestCase(TripletTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.get_features.hydrogen_bonds import TripletException

        super(TripletExceptionsTestCase, self).setUp()

        self.exception = TripletException


class TestStandardMethodExceptions(TripletExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.get_features.hydrogen_bonds import Triplet

        super(TestStandardMethodExceptions, self).setUp()
        self.triplet = Triplet

    def test_init_donor_atom_too_many_covalent_bonds(self):
        from yeti.systems.building_blocks import Atom

        eve_atom = Atom(structure_file_index=5, subsystem_index=3, name='E', xyz_trajectory=np.array([[0.1, 0.1, 0.2]]))
        eve_atom.add_covalent_bond(atom=self.donor_atom)

        with self.assertRaises(self.exception) as context:
            self.triplet(donor_atom=self.donor_atom, acceptor=self.acceptor, periodic=True,
                         unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)

        desired_msg = 'Donor atom has more than one covalent bond. That violates the assumption of this method. ' \
                      'Please contact the developer.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_donor_atom_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.triplet(donor_atom=42, acceptor=self.acceptor, periodic=True,
                         unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)

        desired_msg = self.create_data_type_exception_messages(parameter_name='donor_atom', data_type_name='Atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_acceptor_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.triplet(donor_atom=self.donor_atom, acceptor=1.2, periodic=True,
                         unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)

        desired_msg = self.create_data_type_exception_messages(parameter_name='acceptor', data_type_name='Atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_periodic_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.triplet(donor_atom=self.donor_atom, acceptor=self.acceptor, periodic=[],
                         unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)

        desired_msg = self.create_data_type_exception_messages(parameter_name='periodic', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.triplet(donor_atom=self.donor_atom, acceptor=self.acceptor, periodic=True, unit_cell_angles=(),
                         unit_cell_vectors=self.unit_cell_vectors)

        desired_msg = self.create_data_type_exception_messages(parameter_name='unit_cell_angles',
                                                               data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.triplet(donor_atom=self.donor_atom, acceptor=self.acceptor, periodic=True,
                         unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=[])

        desired_msg = self.create_data_type_exception_messages(parameter_name='unit_cell_vectors',
                                                               data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_dtype(self):
        with self.assertRaises(self.exception) as context:
            self.triplet(donor_atom=self.donor_atom, acceptor=self.acceptor, periodic=True,
                         unit_cell_angles=self.unit_cell_angles.astype(np.float64),
                         unit_cell_vectors=self.unit_cell_vectors)

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='unit_cell_angles',
                                                                 dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_dtype(self):
        with self.assertRaises(self.exception) as context:
            self.triplet(donor_atom=self.donor_atom, acceptor=self.acceptor, periodic=True,
                         unit_cell_angles=self.unit_cell_angles,
                         unit_cell_vectors=self.unit_cell_vectors.astype(np.float64))

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='unit_cell_vectors',
                                                                 dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_shape(self):
        with self.assertRaises(self.exception) as context:
            self.triplet(donor_atom=self.donor_atom, acceptor=self.acceptor, periodic=True,
                         unit_cell_angles=np.array([[90, 90], [90, 90], [90, 90]], dtype=np.float32),
                         unit_cell_vectors=self.unit_cell_vectors)

        desired_msg = self.create_array_shape_exception_messages(parameter_name='unit_cell_angles',
                                                                 desired_shape=(None, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_shape(self):
        with self.assertRaises(self.exception) as context:
            self.triplet(donor_atom=self.donor_atom, acceptor=self.acceptor, periodic=True,
                         unit_cell_angles=self.unit_cell_angles,
                         unit_cell_vectors=np.array([[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]], dtype=np.float32))

        desired_msg = self.create_array_shape_exception_messages(parameter_name='unit_cell_vectors',
                                                                 desired_shape=(None, 3, 3))
        self.assertEqual(desired_msg, str(context.exception))


class TestMaskMethodExceptions(TripletExceptionsTestCase):
    def test_create_mask_distance_cutoff_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.triplet.create_mask(distance_cutoff=1, angle_cutoff=2.)

        desired_msg = self.create_data_type_exception_messages(parameter_name='distance_cutoff', data_type_name='float')
        self.assertEqual(desired_msg, str(context.exception))

    def test_create_mask_angle_cutoff_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.triplet.create_mask(distance_cutoff=0.25, angle_cutoff=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='angle_cutoff', data_type_name='float')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
