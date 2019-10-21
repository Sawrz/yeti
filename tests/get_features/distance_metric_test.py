import unittest

import numpy as np

from tests.blueprints_test import BlueprintTestCase, BlueprintExceptionsTestCase


class DistanceMetricTestCase(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.get_features.distances import DistanceMetric

        super(DistanceMetricTestCase, self).setUp()

        self.dist = DistanceMetric(unit_cell_angles=np.array([[90, 90, 90]]),
                                   unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        self.xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        self.indices = np.array([[0, 1]], dtype=np.int32)
        self.opt = False


class TestStandardMethods(DistanceMetricTestCase):
    def test_init(self):
        from yeti.get_features.distances import DistanceException

        self.assertEqual(DistanceException, self.dist.ensure_data_type.exception_class)


class TestCompatibilityChecks(DistanceMetricTestCase):
    def test_mdtraj_paramaeter_compatibility_check(self):
        self.dist.__mdtraj_paramaeter_compatibility_check__(xyz=self.xyz, indices=self.indices, opt=self.opt)


class DihedralExceptionsTestCase(DistanceMetricTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.get_features.distances import DistanceException

        super(DihedralExceptionsTestCase, self).setUp()

        self.exception = DistanceException


class TestCompatibilityCheckExceptions(DihedralExceptionsTestCase):
    def test_mdtraj_paramaeter_compatibility_check_xyz_not_match(self):
        with self.assertRaises(self.exception) as context:
            self.dist.__mdtraj_paramaeter_compatibility_check__(
                xyz=np.array([[[0, 0, 0], [1, 0, 0], [1, 0, 0]]], dtype=np.float32),
                indices=self.indices, opt=self.opt)

        desired_msg = self.create_array_shape_exception_messages(parameter_name='xyz', desired_shape=(None, 2, 3))
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
