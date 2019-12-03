import unittest

import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintTestCase, BlueprintExceptionsTestCase


class AngleTestCase(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.get_features.angles import Angle

        super(AngleTestCase, self).setUp()

        self.angle = Angle(unit_cell_angles=np.array([[90, 90, 90]]),
                           unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        self.indices = np.array([[0, 1, 2]], dtype=np.int32)
        self.opt = True


class SingleAngleTestCase(AngleTestCase):
    def setUp(self) -> None:
        super(SingleAngleTestCase, self).setUp()

        self.xyz = np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]]], dtype=np.float32)
        self.out = np.zeros((self.xyz.shape[0], self.indices.shape[0]), dtype=np.float32)


class MultiAngleTestCase(AngleTestCase):
    def setUp(self) -> None:
        super(MultiAngleTestCase, self).setUp()

        self.angle.unit_cell_angles = np.array([[90, 90, 90], [90, 90, 90], [90, 90, 90]])
        self.angle.unit_cell_vectors = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                                 [[10, 0, 0], [0, 10, 0], [0, 0, 10]]])

        self.xyz = np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]],
                             [[0.23, 0, 0.75], [0, 0, 0], [0.1, 0.3, 0.4]],
                             [[2.715, 3.164, 2.63], [2.626, 3.11, 2.624], [2.715, 3.164, 2.63]]], dtype=np.float32)
        self.out = np.zeros((self.xyz.shape[0], self.indices.shape[0]), dtype=np.float32)


class TestStandardMethods(AngleTestCase):
    def test_init(self):
        from yeti.get_features.angles import AngleException

        self.assertEqual(AngleException, self.angle.ensure_data_type.exception_class)


class TestCompatibilityChecks(SingleAngleTestCase):
    def test_mdtraj_paramaeter_compatibility_check(self):
        self.angle.__mdtraj_paramaeter_compatibility_check__(xyz=self.xyz, indices=self.indices, opt=self.opt)


class SingleAngleCalculations(SingleAngleTestCase):
    def setUp(self) -> None:
        super(SingleAngleCalculations, self).setUp()

        self.exp_periodic = np.array([[0.37432076]])
        self.exp_non_periodic = np.array([[0.125604342702096]])

    def test_legacy(self):
        res = self.angle.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True,
                                             out=self.out, legacy=True)

        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)

    def test_non_legacy(self):
        res = self.angle.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True,
                                             out=self.out, legacy=False)

        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)

    def test_periodic(self):
        res = self.angle.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True, out=self.out,
                                             legacy=True)

        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)

    def test_non_periodic(self):
        res = self.angle.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=False, out=self.out,
                                             legacy=True)

        npt.assert_almost_equal(res, self.exp_non_periodic, decimal=5)

    def test_calculate_no_pbc_not_optimized(self):
        res = self.angle.__calculate_no_pbc__(xyz=self.xyz, indices=self.indices, opt=False, legacy=True)

        npt.assert_almost_equal(res, self.exp_non_periodic, decimal=5)

    def test_calculate_no_pbc_optimized(self):
        res = self.angle.__calculate_no_pbc__(xyz=self.xyz, indices=self.indices, opt=True, legacy=True)

        npt.assert_almost_equal(res, self.exp_non_periodic, decimal=5)

    def test_calculate_minimal_image_convention_not_optimized(self):
        res = self.angle.__calculate_minimal_image_convention__(xyz=self.xyz, indices=self.indices, opt=False,
                                                                legacy=True)

        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)

    def test_calculate_minimal_image_convention_optimized(self):
        res = self.angle.__calculate_minimal_image_convention__(xyz=self.xyz, indices=self.indices, opt=True,
                                                                legacy=True)

        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)


class MultiAngleCalculations(MultiAngleTestCase):
    def setUp(self) -> None:
        super(MultiAngleCalculations, self).setUp()

        self.exp_periodic = np.array([[0.37432076], [2.03144108], [np.nan]])
        self.exp_non_periodic = np.array([[0.125604342702096], [0.63091194], [np.nan]])

    def test_legacy(self):
        res = self.angle.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True,
                                             out=self.out, legacy=True)

        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)

    def test_non_legacy(self):
        res = self.angle.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True,
                                             out=self.out, legacy=False)

        self.exp_periodic[2, 0] = 0
        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)

    def test_periodic(self):
        res = self.angle.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True, out=self.out,
                                             legacy=True)

        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)

    def test_non_periodic(self):
        res = self.angle.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=False, out=self.out,
                                             legacy=True)

        npt.assert_almost_equal(res, self.exp_non_periodic, decimal=5)

    def test_calculate_no_pbc_not_optimized(self):
        res = self.angle.__calculate_no_pbc__(xyz=self.xyz, indices=self.indices, opt=False, legacy=True)

        npt.assert_almost_equal(res, self.exp_non_periodic, decimal=5)

    def test_calculate_no_pbc_optimized(self):
        res = self.angle.__calculate_no_pbc__(xyz=self.xyz, indices=self.indices, opt=True, legacy=True)

        # optimization seems nan-safe
        self.exp_non_periodic[2, 0] = 0
        npt.assert_almost_equal(res, self.exp_non_periodic, decimal=5)

    def test_calculate_no_pbc_non_legacy(self):
        res = self.angle.__calculate_no_pbc__(xyz=self.xyz, indices=self.indices, opt=False, legacy=False)

        # optimization seems nan-safe
        self.exp_non_periodic[2, 0] = 0
        npt.assert_almost_equal(res, self.exp_non_periodic, decimal=5)

    def test_calculate_minimal_image_convention_not_optimized(self):
        res = self.angle.__calculate_minimal_image_convention__(xyz=self.xyz, indices=self.indices, opt=False,
                                                                legacy=True)

        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)

    def test_calculate_minimal_image_convention_optimized(self):
        res = self.angle.__calculate_minimal_image_convention__(xyz=self.xyz, indices=self.indices, opt=True,
                                                                legacy=True)

        # optimization seems nan-safe
        self.exp_periodic[2, 0] = 0
        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)

    def test_calculate_minimal_image_convention_non_legacy(self):
        res = self.angle.__calculate_minimal_image_convention__(xyz=self.xyz, indices=self.indices, opt=False,
                                                                legacy=False)

        # optimization seems nan-safe
        self.exp_periodic[2, 0] = 0
        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)


class AngleExceptionsTestCase(SingleAngleTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.get_features.angles import AngleException

        super(AngleExceptionsTestCase, self).setUp()

        self.exception = AngleException


class TestCompatibilityCheckExceptions(AngleExceptionsTestCase):
    def test_mdtraj_paramaeter_compatibility_check_xyz_not_match(self):
        with self.assertRaises(self.exception) as context:
            self.angle.__mdtraj_paramaeter_compatibility_check__(
                xyz=np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32),
                indices=self.indices, opt=self.opt)

        desired_msg = self.create_array_shape_exception_messages(parameter_name='xyz', desired_shape=(None, 3, 3))
        self.assertEqual(desired_msg, str(context.exception))


class TestCalculateAngleMethodExceptions(AngleExceptionsTestCase):
    def test_calculate_angle_wrong_data_type_out(self):
        with self.assertRaises(self.exception) as context:
            self.angle.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True, out=[], legacy=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='out', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_angle_wrong_dtype_out(self):
        with self.assertRaises(self.exception) as context:
            self.angle.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True,
                                           out=np.zeros((self.xyz.shape[0], self.indices.shape[0]),
                                                        dtype=np.float64), legacy=True)

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='out', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_angle_wrong_shape_out(self):
        with self.assertRaises(self.exception) as context:
            self.angle.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True,
                                           out=np.zeros((2, self.indices.shape[0]), dtype=np.float32), legacy=True)

        desired_msg = self.create_array_shape_exception_messages(parameter_name='out', desired_shape=(1, 1))
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_no_pbc_wrong_input(self):
        with self.assertRaises(self.exception) as context:
            self.angle.__calculate_no_pbc__(
                xyz=np.array([[[0.2, 0.1, 0.2], [0.9, 0.9, 0.9], [0.1, 0.2, 0.1]]], dtype=np.float64),
                indices=self.indices, opt=True, legacy=True)

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='xyz', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_minimal_image_convention_wrong_input(self):
        with self.assertRaises(self.exception) as context:
            self.angle.__calculate_minimal_image_convention__(
                xyz=np.array([[[0.23, 0.3, 0.25], [0.1, 0.1, 0.1], [0.1, 0.3, 0.4]]], dtype=np.float64),
                indices=self.indices, opt=True, legacy=True)

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='xyz', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
