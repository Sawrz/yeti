import unittest

import numpy as np
import numpy.testing as npt

from test_utils.test_utils import create_data_type_exception_messages, create_array_dtype_exception_messages, \
    create_array_shape_exception_messages


class TestAngle(unittest.TestCase):
    def test_init(self):
        from yeti.get_features.angles import Angle, AngleException

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        self.assertEqual(AngleException, angle.ensure_data_type.exception_class)

    def test_mdtraj_paramaeter_compatibility_check(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        angle.__mdtraj_paramaeter_compatibility_check__(
            xyz=np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=np.float32),
            indices=np.array([[0, 1, 2]], dtype=np.int32), opt=True)

    def test_mdtraj_paramaeter_compatibility_check_xyz_not_match(self):
        from yeti.get_features.angles import Angle, AngleException

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        with self.assertRaises(AngleException) as context:
            angle.__mdtraj_paramaeter_compatibility_check__(
                xyz=np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32),
                indices=np.array([[0, 1, 2]], dtype=np.int32), opt=True)

        desired_msg = create_array_shape_exception_messages(parameter_name='xyz', desired_shape=(None, 3, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_angle_not_periodic_simple(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.25, 0, 0], [0, 0, 0], [0, 0.1, 0]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)
        out = np.zeros((xyz.shape[0], indices.shape[0]), dtype=np.float32)

        res = angle.__calculate_angle__(xyz=xyz, indices=indices, periodic=False, out=out)
        exp = np.array([[np.pi / 2]])

        npt.assert_almost_equal(res, exp, decimal=5)

    def test_calculate_angle_not_periodic_multi_dim(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.23, 0, 0.75], [0, 0, 0], [0.1, 0.3, 0.4]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)
        out = np.zeros((xyz.shape[0], indices.shape[0]), dtype=np.float32)

        res = angle.__calculate_angle__(xyz=xyz, indices=indices, periodic=False, out=out)
        exp = np.array([[0.63091194]])

        npt.assert_almost_equal(res, exp, decimal=5)

    def test_calculate_angle_periodic_multi_dim(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)
        out = np.zeros((xyz.shape[0], indices.shape[0]), dtype=np.float32)

        res = angle.__calculate_angle__(xyz=xyz, indices=indices, periodic=True, out=out)
        exp = np.array([[0.37431625336629]])

        npt.assert_almost_equal(res, exp, decimal=5)

    def test_calculate_angle_wrong_data_type_out(self):
        from yeti.get_features.angles import Angle, AngleException

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)
        out = []

        with self.assertRaises(AngleException) as context:
            angle.__calculate_angle__(xyz=xyz, indices=indices, periodic=True, out=out)

        desired_msg = create_data_type_exception_messages(parameter_name='out', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_angle_wrong_dtype_out(self):
        from yeti.get_features.angles import Angle, AngleException

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)
        out = np.zeros((xyz.shape[0], indices.shape[0]), dtype=np.float64)

        with self.assertRaises(AngleException) as context:
            angle.__calculate_angle__(xyz=xyz, indices=indices, periodic=True, out=out)

        desired_msg = create_array_dtype_exception_messages(parameter_name='out', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_angle_wrong_shape_out(self):
        from yeti.get_features.angles import Angle, AngleException

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)
        out = np.zeros((2, indices.shape[0]), dtype=np.float32)

        with self.assertRaises(AngleException) as context:
            angle.__calculate_angle__(xyz=xyz, indices=indices, periodic=True, out=out)

        desired_msg = create_array_shape_exception_messages(parameter_name='out', desired_shape=(1, 1))
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_no_pbc_not_optimized(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        res = angle.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[0.125604342702096]])

        npt.assert_almost_equal(res, exp, decimal=5)

    def test_calculate_no_pbc_not_optimized_multi_frame(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90], [90, 90, 90]]),
                      unit_cell_vectors=np.array(
                          [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array(
            [[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]], [[0.23, 0, 0.75], [0, 0, 0], [0.1, 0.3, 0.4]]],
            dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        res = angle.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[0.125604342702096], [0.63091194]])

        npt.assert_almost_equal(res, exp, decimal=5)

    def test_calculate_no_pbc_optimized(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        res = angle.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[0.125604342702096]])

        npt.assert_almost_equal(res, exp, decimal=5)

    def test_calculate_no_pbc_optimized_multi_frame(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90], [90, 90, 90]]),
                      unit_cell_vectors=np.array(
                          [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array(
            [[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]], [[0.23, 0, 0.75], [0, 0, 0], [0.1, 0.3, 0.4]]],
            dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        res = angle.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[0.125604342702096], [0.63091194]])

        npt.assert_almost_equal(res, exp, decimal=5)

    def test_calculate_no_pbc_wrong_input(self):
        from yeti.get_features.angles import Angle, AngleException

        angle = Angle(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]]], dtype=np.float64)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        with self.assertRaises(AngleException) as context:
            angle.__calculate_no_pbc__(xyz=xyz, indices=indices, opt=True)

        desired_msg = create_array_dtype_exception_messages(parameter_name='xyz', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_minimal_image_convention_not_optimized_not_boundary(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.23, 0.3, 0.25], [0.1, 0.1, 0.1], [0.1, 0.3, 0.4]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        res = angle.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[0.57968202167814]])

        npt.assert_almost_equal(res, exp, decimal=5)

    def test_calculate_minimal_image_convention_not_optimized_boundary(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        res = angle.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[0.37431625336629]], dtype=np.float32)

        npt.assert_almost_equal(res, exp, decimal=5)

    def test_calculate_minimal_image_convention_not_optimized_multi_frame(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=True, unit_cell_angles=np.array([[90, 90, 90], [90, 90, 90]]),
                      unit_cell_vectors=np.array(
                          [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]],
                        [[0.23, 0.3, 0.25], [0.1, 0.1, 0.1], [0.1, 0.3, 0.4]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        res = angle.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=False)
        exp = np.array([[0.37431625336629], [0.57968202167814]])

        npt.assert_almost_equal(res, exp, decimal=5)

    def test_calculate_minimal_image_convention_optimized_no_boundary(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.23, 0.3, 0.25], [0.1, 0.1, 0.1], [0.1, 0.3, 0.4]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        res = angle.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[0.57968202167814]])

        npt.assert_almost_equal(res, exp, decimal=5)

    def test_calculate_minimal_image_convention_optimized_boundary(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        res = angle.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[0.37431625336629]], dtype=np.float32)

        npt.assert_almost_equal(res, exp, decimal=5)

    def test_calculate_minimal_image_convention_optimized_multi_frame(self):
        from yeti.get_features.angles import Angle

        angle = Angle(periodic=True, unit_cell_angles=np.array([[90, 90, 90], [90, 90, 90]]),
                      unit_cell_vectors=np.array(
                          [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]],
                        [[0.23, 0.3, 0.25], [0.1, 0.1, 0.1], [0.1, 0.3, 0.4]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        res = angle.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=True)
        exp = np.array([[0.37431625336629], [0.57968202167814]])

        npt.assert_almost_equal(res, exp, decimal=5)

    def test_calculate_minimal_image_convention_wrong_input(self):
        from yeti.get_features.angles import Angle, AngleException

        angle = Angle(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                      unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0.23, 0.3, 0.25], [0.1, 0.1, 0.1], [0.1, 0.3, 0.4]]], dtype=np.float64)
        indices = np.array([[0, 1, 2]], dtype=np.int32)

        with self.assertRaises(AngleException) as context:
            angle.__calculate_minimal_image_convention__(xyz=xyz, indices=indices, opt=True)

        desired_msg = create_array_dtype_exception_messages(parameter_name='xyz', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
