import unittest

import numpy as np

from test_utils.test_utils import create_array_shape_exception_messages


class TestDistanceMetric(unittest.TestCase):
    def test_init(self):
        from yeti.get_features.distances import DistanceMetric, DistanceException

        dist = DistanceMetric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                              unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        self.assertEqual(DistanceException, dist.ensure_data_type.exception_class)

    def test_mdtraj_paramaeter_compatibility_check(self):
        from yeti.get_features.distances import DistanceMetric

        dist = DistanceMetric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                              unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        dist.__mdtraj_paramaeter_compatibility_check__(xyz=np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32),
                                                       indices=np.array([[0, 1]], dtype=np.int32), opt=True)

    def test_mdtraj_paramaeter_compatibility_check_xyz_not_match(self):
        from yeti.get_features.distances import DistanceMetric, DistanceException

        dist = DistanceMetric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                              unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))



        with self.assertRaises(DistanceException) as context:
            dist.__mdtraj_paramaeter_compatibility_check__(
                xyz=np.array([[[0, 0, 0], [1, 0, 0], [1, 0, 0]]], dtype=np.float32),
                indices=np.array([[0, 1]], dtype=np.int32), opt=True)

        desired_msg = create_array_shape_exception_messages(parameter_name='xyz', desired_shape=(None, 2, 3))
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
