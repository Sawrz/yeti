import unittest

import numpy as np
import numpy.testing as npt

from test_utils.test_utils import create_array_dtype_exception_messages, create_data_type_exception_messages


class TestDisplacement(unittest.TestCase):

    def test_calculate_no_pbc_not_optimized_x(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[[1, 0, 0]]], dtype=np.float32)

        npt.assert_array_equal(res, exp)

    def test_calculate_no_pbc_not_optimized_y(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 1, 0], [0, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[[0, -1, 0]]], dtype=np.float32)

        npt.assert_array_equal(res, exp)

    def test_calculate_no_pbc_not_optimized_z(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 1], [0, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[[0, 0, -1]]], dtype=np.float32)

        npt.assert_array_equal(res, exp)

    def test_calculate_no_pbc_not_optimized_multi_frame(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]], [[0, 0.25, 0], [0, 0.75, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[[1, 0, 0]], [[0, 0.5, 0]]])

        npt.assert_array_equal(res, exp)

    def test_calculate_no_pbc_optimized_x(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[[1, 0, 0]]], dtype=np.float32)

        npt.assert_array_equal(res, exp)

    def test_calculate_no_pbc_optimized_y(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 1, 0], [0, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[[0, -1, 0]]], dtype=np.float32)

        npt.assert_array_equal(res, exp)

    def test_calculate_no_pbc_optimized_z(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 1], [0, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[[0, 0, -1]]], dtype=np.float32)

        npt.assert_array_equal(res, exp)

    def test_calculate_no_pbc_optimized_multi_frame(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=False, unit_cell_angles=np.array([[90, 90, 90], [90, 90, 90]]),
                            unit_cell_vectors=np.array(
                                [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]], [[0, 0.25, 0], [0, 0.75, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[[1, 0, 0]], [[0, 0.5, 0]]])

        npt.assert_array_equal(res, exp)

    def test_calculate_no_pbc_wrong_input(self):
        from yeti.get_features.distances import Displacement, DistanceException

        dist = Displacement(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int32)

        with self.assertRaises(DistanceException) as context:
            dist.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=True)

        desired_msg = create_array_dtype_exception_messages(parameter_name='xyz', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_minimal_image_convention_not_optimized_x_not_boundary(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [0.25, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[[0.25, 0, 0]]], dtype=np.float32)

        npt.assert_array_equal(res, exp)

    def test_calculate_minimal_image_convention_not_optimized_y_not_boundary(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0.5, 0], [0, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[[0, -0.5, 0]]], dtype=np.float32)

        npt.assert_array_equal(res, exp)

    def test_calculate_minimal_image_convention_not_optimized_z_not_boundary(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0.1], [0, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[[0, 0, -0.1]]], dtype=np.float32)

        npt.assert_array_equal(res, exp)

    def test_calculate_minimal_image_convention_not_optimized_x_boundary(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.9, 0, 0], [0.1, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[[0.2, 0, 0]]], dtype=np.float32)

        npt.assert_almost_equal(exp, res, decimal=5)

    def test_calculate_minimal_image_convention_not_optimized_y_boundary(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0.7, 0], [0, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[[0, 0.3, 0]]], dtype=np.float32)

        npt.assert_almost_equal(exp, res, decimal=5)

    def test_calculate_minimal_image_convention_not_optimized_z_boundary(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0.8], [0, 0, 0.1]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[[0, 0, 0.3]]], dtype=np.float32)

        npt.assert_almost_equal(exp, res, decimal=5)

    def test_calculate_minimal_image_convention_not_optimized_multi_frame(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90], [90, 90, 90]]),
                            unit_cell_vectors=np.array(
                                [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]], [[0, 0.25, 0], [0, 0.75, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[[0, 0, 0]], [[0, 0.5, 0]]])

        npt.assert_array_equal(res, exp)

    def test_calculate_minimal_image_convention_optimized_x_no_boundary(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [0.25, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[[0.25, 0, 0]]], dtype=np.float32)

        npt.assert_array_equal(res, exp)

    def test_calculate_minimal_image_convention_optimized_y_no_boundary(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0.5, 0], [0, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[[0, -0.5, 0]]], dtype=np.float32)

        npt.assert_array_equal(res, exp)

    def test_calculate_minimal_image_convention_optimized_z_no_boundary(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0.1], [0, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[[0, 0, -0.1]]], dtype=np.float32)

        npt.assert_array_equal(res, exp)

    def test_calculate_minimal_image_convention_optimized_x_boundary(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.9, 0, 0], [0.1, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[[0.2, 0, 0]]], dtype=np.float32)

        npt.assert_almost_equal(exp, res, decimal=5)

    def test_calculate_minimal_image_convention_optimized_y_boundary(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0.7, 0], [0, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[[0, 0.3, 0]]], dtype=np.float32)

        npt.assert_almost_equal(exp, res, decimal=5)

    def test_calculate_minimal_image_convention_optimized_z_boundary(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0.8], [0, 0, 0.1]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[[0, 0, 0.3]]], dtype=np.float32)

        npt.assert_almost_equal(exp, res, decimal=5)

    def test_calculate_minimal_image_convention_optimized_multi_frame(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90], [90, 90, 90]]),
                            unit_cell_vectors=np.array(
                                [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]], [[0, 0.25, 0], [0, 0.75, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[[0, 0, 0]], [[0, 0.5, 0]]])

        npt.assert_array_equal(res, exp)

    def test_calculate_minimal_image_convention_wrong_input(self):
        from yeti.get_features.distances import Displacement, DistanceException

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int32)

        with self.assertRaises(DistanceException) as context:
            dist.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=True)

        desired_msg = create_array_dtype_exception_messages(parameter_name='xyz', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_compatibility_layer_periodic(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0.8], [0, 0, 0.1]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.get_compatibility_layer(xyz=xyz, indices=indices, opt=True, periodic=True)
        exp = np.array([[[0, 0, 0.3]]], dtype=np.float32)

        npt.assert_almost_equal(exp, res, decimal=5)

    def test_get_compatibility_layer_not_periodic(self):
        from yeti.get_features.distances import Displacement

        dist = Displacement(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0.8], [0, 0, 0.1]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        res = dist.get_compatibility_layer(xyz=xyz, indices=indices, opt=True, periodic=False)
        exp = np.array([[[0, 0, -0.7]]], dtype=np.float32)

        npt.assert_almost_equal(exp, res, decimal=5)

    def test_get_compatibility_layer_wrong_input(self):
        from yeti.get_features.distances import Displacement, DistanceException

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int32)

        with self.assertRaises(DistanceException) as context:
            dist.get_compatibility_layer(xyz=xyz, indices=indices, opt=True, periodic=True)

        desired_msg = create_array_dtype_exception_messages(parameter_name='xyz', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_compatibility_layer_periodic_wrong_data_type(self):
        from yeti.get_features.distances import Displacement, DistanceException

        dist = Displacement(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                            unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)

        with self.assertRaises(DistanceException) as context:
            dist.get_compatibility_layer(xyz=xyz, indices=indices, opt=True, periodic=42)

        desired_msg = create_data_type_exception_messages(parameter_name='periodic', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
