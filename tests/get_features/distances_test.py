import unittest

import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintTestCase, BlueprintExceptionsTestCase


class DistanceTestCase(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.get_features.distances import Distance

        super(DistanceTestCase, self).setUp()

        self.distance = Distance(unit_cell_angles=np.array([[90, 90, 90]]),
                                 unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        self.indices = np.array([[0, 1]], dtype=np.int32)


class SingleDistanceTestCase(DistanceTestCase):
    def setUp(self) -> None:
        super(SingleDistanceTestCase, self).setUp()

        self.x = np.array([[[0, 0, 0], [0.75, 0, 0]]], dtype=np.float32)
        self.y = np.array([[[0, 0.75, 0], [0, 0, 0]]], dtype=np.float32)
        self.z = np.array([[[0, 0, 0.75], [0, 0, 0]]], dtype=np.float32)

        self.xyz = np.array([[[0, 0.1, 0], [0.75, 0.8, 0.9]]], dtype=np.float32)


class MultiDistanceTestCase(DistanceTestCase):
    def setUp(self) -> None:
        super(MultiDistanceTestCase, self).setUp()

        self.distance.unit_cell_angles = np.array([[90, 90, 90], [90, 90, 90]])
        self.distance.unit_cell_vectors = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

        self.xyz = np.array([[[0, 0, 0], [0.75, 0.8, 0.9]], [[0.3, 0.25, 0.8], [0.4, 0.75, 0]]], dtype=np.float32)


class SingleDisplacementCalculationsNonOptimized(SingleDistanceTestCase):
    def setUp(self) -> None:
        super(SingleDisplacementCalculationsNonOptimized, self).setUp()

        self.opt = False

    def test_calculate_no_pbc_x(self):
        res = self.distance.__calculate_no_pbc__(xyz=self.x, indices=self.indices, opt=self.opt)

        npt.assert_almost_equal(res, np.array([[0.75]], dtype=np.float32), decimal=5)

    def test_calculate_no_pbc_y(self):
        res = self.distance.__calculate_no_pbc__(xyz=self.y, indices=self.indices, opt=self.opt)

        npt.assert_almost_equal(res, np.array([[0.75]], dtype=np.float32), decimal=5)

    def test_calculate_no_pbc_z(self):
        res = self.distance.__calculate_no_pbc__(xyz=self.z, indices=self.indices, opt=self.opt)

        npt.assert_almost_equal(res, np.array([[0.75]], dtype=np.float32), decimal=5)

    def test_calculate_minimal_image_convention_x(self):
        res = self.distance.__calculate_minimal_image_convention__(xyz=self.x, indices=self.indices, opt=self.opt)

        npt.assert_almost_equal(res, np.array([[0.25]]), decimal=5)

    def test_calculate_minimal_image_convention_y(self):
        res = self.distance.__calculate_minimal_image_convention__(xyz=self.y, indices=self.indices, opt=self.opt)

        npt.assert_almost_equal(res, np.array([[0.25]]), decimal=5)

    def test_calculate_minimal_image_convention_z(self):
        res = self.distance.__calculate_minimal_image_convention__(xyz=self.z, indices=self.indices, opt=self.opt)

        npt.assert_almost_equal(res, np.array([[0.25]]), decimal=5)


class SingleDisplacementCalculationsOptimized(SingleDisplacementCalculationsNonOptimized):
    def setUp(self) -> None:
        super(SingleDisplacementCalculationsOptimized, self).setUp()

        self.opt = True


class MultiDisplacementCalculationsNonOptimized(MultiDistanceTestCase):
    def setUp(self) -> None:
        super(MultiDisplacementCalculationsNonOptimized, self).setUp()

        self.opt = False

        self.exp_periodic = np.array([[0.33541019662496846], [0.5477225575051662]])
        self.exp_non_periodic = np.array([[1.4186260959111108], [0.9486832980505139]])

    def test_calculate_no_pbc(self):
        res = self.distance.__calculate_no_pbc__(xyz=self.xyz, indices=self.indices, opt=self.opt)

        npt.assert_almost_equal(res, self.exp_non_periodic, decimal=5)

    def test_calculate_minimal_image_convention(self):
        res = self.distance.__calculate_minimal_image_convention__(xyz=self.xyz, indices=self.indices, opt=self.opt)

        npt.assert_almost_equal(res, self.exp_periodic, decimal=5)


class MultiDisplacementCalculationsOptimized(MultiDisplacementCalculationsNonOptimized):
    def setUp(self) -> None:
        super(MultiDisplacementCalculationsOptimized, self).setUp()

        self.opt = True


class DisplacementExceptionsTestCase(SingleDistanceTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.get_features.distances import DistanceException

        super(DisplacementExceptionsTestCase, self).setUp()

        self.exception = DistanceException

        self.wrong_xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float64)


class TestCalculateDisplacementMethodExceptions(DisplacementExceptionsTestCase):
    def test_calculate_no_pbc_wrong_input(self):
        with self.assertRaises(self.exception) as context:
            self.distance.__calculate_no_pbc__(xyz=self.wrong_xyz, indices=self.indices, opt=True)

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='xyz', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_minimal_image_convention_wrong_input(self):
        with self.assertRaises(self.exception) as context:
            self.distance.__calculate_minimal_image_convention__(xyz=self.wrong_xyz, indices=self.indices, opt=True)

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='xyz', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
