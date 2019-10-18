import unittest

import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintTestCase, BlueprintExceptionsTestCase


class DihedralTestCase(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.get_features.angles import Dihedral

        super(DihedralTestCase, self).setUp()

        self.dihedral = Dihedral(unit_cell_angles=np.array([[90, 90, 90]]),
                                 unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        self.indices = np.array([[0, 1, 2, 3]], dtype=np.int32)
        self.opt = True


class SingleDihedralTestCase(DihedralTestCase):
    def setUp(self) -> None:
        super(SingleDihedralTestCase, self).setUp()

        self.xyz = np.array([[[0.23, 0.3, 0.4], [0.1, 0.31, 0.15], [0.2, 0.19, 0.1], [0.4, 0.22, 0.35]]],
                            dtype=np.float32)
        self.out = np.zeros((self.xyz.shape[0], self.indices.shape[0]), dtype=np.float32)


class MultiDihedralTestCase(DihedralTestCase):
    def setUp(self) -> None:
        super(MultiDihedralTestCase, self).setUp()

        self.dihedral.unit_cell_angles = np.array([[90, 90, 90], [90, 90, 90]])
        self.dihedral.unit_cell_vectors = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

        self.xyz = np.array([[[0.8, 0.9, 0.7], [0.9, 0.1, 0.9], [0.1, 0.2, 0.25], [0.25, 0.33, 0.4]],
                             [[0.23, 0.3, 0.4], [0.1, 0.31, 0.15], [0.2, 0.19, 0.1], [0.4, 0.22, 0.35]]],
                            dtype=np.float32)
        self.out = np.zeros((self.xyz.shape[0], self.indices.shape[0]), dtype=np.float32)


class TestStandardMethods(DihedralTestCase):
    def test_init(self):
        from yeti.get_features.angles import DihedralException

        self.assertEqual(DihedralException, self.dihedral.ensure_data_type.exception_class)


class TestCompatibilityChecks(SingleDihedralTestCase):
    def test_mdtraj_paramaeter_compatibility_check(self):
        self.dihedral.__mdtraj_paramaeter_compatibility_check__(xyz=self.xyz, indices=self.indices, opt=self.opt)


class SingleAngleCalculations(SingleDihedralTestCase):
    def setUp(self) -> None:
        super(SingleAngleCalculations, self).setUp()

        self.exp = np.array([[-0.2311416423440]])

    def test_periodic(self):
        res = self.dihedral.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True,
                                                out=self.out)

        npt.assert_almost_equal(res, self.exp, decimal=5)

    def test_non_periodic(self):
        res = self.dihedral.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=False, out=self.out)

        npt.assert_almost_equal(res, self.exp, decimal=5)

    def test_calculate_no_pbc_not_optimized(self):
        res = self.dihedral.__calculate_no_pbc__(xyz=self.xyz, indices=self.indices, opt=False)

        npt.assert_almost_equal(res, self.exp, decimal=5)

    def test_calculate_no_pbc_optimized(self):
        res = self.dihedral.__calculate_no_pbc__(xyz=self.xyz, indices=self.indices, opt=True)

        npt.assert_almost_equal(res, self.exp, decimal=5)

    def test_calculate_minimal_image_convention_not_optimized(self):
        res = self.dihedral.__calculate_minimal_image_convention__(xyz=self.xyz, indices=self.indices, opt=False)

        npt.assert_almost_equal(res, self.exp, decimal=5)

    def test_calculate_minimal_image_convention_optimized(self):
        res = self.dihedral.__calculate_minimal_image_convention__(xyz=self.xyz, indices=self.indices, opt=True)

        npt.assert_almost_equal(res, self.exp, decimal=5)


class MultiAngleCalculations(MultiDihedralTestCase):
    def setUp(self) -> None:
        super(MultiAngleCalculations, self).setUp()

        self.exp_periodic = np.array([[2.42560681293738], [-0.2311416423440]])
        self.exp_non_periodic = np.array([[-0.2628531035206], [-0.2311416423440]])

    def test_periodic(self):
        res = self.dihedral.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True, out=self.out)

        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)

    def test_non_periodic(self):
        res = self.dihedral.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=False, out=self.out)

        npt.assert_almost_equal(res, self.exp_non_periodic, decimal=5)

    def test_calculate_no_pbc_not_optimized(self):
        res = self.dihedral.__calculate_no_pbc__(xyz=self.xyz, indices=self.indices, opt=False)

        npt.assert_almost_equal(res, self.exp_non_periodic, decimal=5)

    def test_calculate_no_pbc_optimized(self):
        res = self.dihedral.__calculate_no_pbc__(xyz=self.xyz, indices=self.indices, opt=True)

        npt.assert_almost_equal(res, self.exp_non_periodic, decimal=5)

    def test_calculate_minimal_image_convention_not_optimized(self):
        res = self.dihedral.__calculate_minimal_image_convention__(xyz=self.xyz, indices=self.indices, opt=False)

        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)

    def test_calculate_minimal_image_convention_optimized(self):
        res = self.dihedral.__calculate_minimal_image_convention__(xyz=self.xyz, indices=self.indices, opt=True)

        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)


class DihedralExceptionsTestCase(SingleDihedralTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.get_features.angles import DihedralException

        super(DihedralExceptionsTestCase, self).setUp()

        self.exception = DihedralException


class TestCompatibilityCheckExceptions(DihedralExceptionsTestCase):
    def test_mdtraj_paramaeter_compatibility_check_xyz_not_match(self):
        with self.assertRaises(self.exception) as context:
            self.dihedral.__mdtraj_paramaeter_compatibility_check__(
                xyz=np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32),
                indices=self.indices, opt=self.opt)

        desired_msg = self.create_array_shape_exception_messages(parameter_name='xyz', desired_shape=(None, 4, 3))
        self.assertEqual(desired_msg, str(context.exception))


class TestCalculateDihedralMethodExceptions(DihedralExceptionsTestCase):
    def test_calculate_angle_wrong_data_type_out(self):
        with self.assertRaises(self.exception) as context:
            self.dihedral.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True, out=[])

        desired_msg = self.create_data_type_exception_messages(parameter_name='out', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_angle_wrong_dtype_out(self):
        with self.assertRaises(self.exception) as context:
            self.dihedral.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True,
                                              out=np.zeros((self.xyz.shape[0], self.indices.shape[0]),
                                                           dtype=np.float64))

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='out', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_angle_wrong_shape_out(self):
        with self.assertRaises(self.exception) as context:
            self.dihedral.__calculate_angle__(xyz=self.xyz, indices=self.indices, periodic=True,
                                              out=np.zeros((2, self.indices.shape[0]), dtype=np.float32))

        desired_msg = self.create_array_shape_exception_messages(parameter_name='out', desired_shape=(1, 1))
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_no_pbc_wrong_input(self):
        with self.assertRaises(self.exception) as context:
            self.dihedral.__calculate_no_pbc__(
                xyz=np.array([[[0.23, 0.3, 0.4], [0.1, 0.31, 0.15], [0.2, 0.19, 0.1], [0.4, 0.22, 0.35]]],
                             dtype=np.float64),
                indices=self.indices, opt=True)

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='xyz', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_minimal_image_convention_wrong_input(self):
        with self.assertRaises(self.exception) as context:
            self.dihedral.__calculate_minimal_image_convention__(
                xyz=np.array([[[0.23, 0.3, 0.4], [0.1, 0.31, 0.15], [0.2, 0.19, 0.1], [0.4, 0.22, 0.35]]],
                             dtype=np.float64), indices=self.indices, opt=True)

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='xyz', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
