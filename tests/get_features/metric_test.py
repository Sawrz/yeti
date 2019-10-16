import unittest

import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintTestCase, BlueprintExceptionsTestCase


class MetricTestCase(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.get_features.metric import Metric

        self.metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                             unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))


class TestStandardMethods(MetricTestCase):
    def test_init(self):
        from yeti.get_features.metric import MetricException

        self.assertEqual(self.metric.ensure_data_type.exception_class, MetricException)
        self.assertTrue(self.metric.periodic)
        npt.assert_array_equal(self.metric.unit_cell_angles, np.array([[90, 90, 90]]))
        npt.assert_array_equal(self.metric.unit_cell_vectors, np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))


class TestPreparationMethods(MetricTestCase):
    def test_prepare_xyz_data(self):
        from yeti.systems.building_blocks import Atom

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A',
                       xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        res_xyz = self.metric.__prepare_xyz_data__((atom_01, atom_02))
        exp_xyz = np.array([[[0, 1, 2], [6, 7, 8]], [[3, 4, 5], [9, 10, 11]]])

        npt.assert_array_equal(res_xyz, exp_xyz)

    def test_prepare_atom_indices_amount_two(self):
        res_amount = self.metric.__prepare_atom_indices__(2)
        npt.assert_array_equal(np.array([[0, 1]]), res_amount)

    def test_prepare_atom_indices_amount_three(self):
        res_amount = self.metric.__prepare_atom_indices__(3)
        npt.assert_array_equal(np.array([[0, 1, 2]]), res_amount)

    def test_prepare_atom_indices_amount_four(self):
        res_amount = self.metric.__prepare_atom_indices__(4)
        npt.assert_array_equal(np.array([[0, 1, 2, 3]]), res_amount)


class CompatibilityChecksTestCase(MetricTestCase):
    def setUp(self) -> None:
        super(CompatibilityChecksTestCase, self).setUp()

        self.xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        self.indices = np.array([[0, 1]], dtype=np.int32)
        self.opt = False


class TestCompatibilityChecks(CompatibilityChecksTestCase):
    def test_mdtraj_paramaeter_compatibility_check_amount_two(self):
        self.metric.__mdtraj_paramaeter_compatibility_check__(xyz=self.xyz, indices=self.indices, opt=self.opt,
                                                              atom_amount=2)

    def test_mdtraj_paramaeter_compatibility_check_amount_three(self):
        self.metric.__mdtraj_paramaeter_compatibility_check__(xyz=np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]],
                                                                           dtype=np.float32),
                                                              indices=np.array([[0, 1, 2]], dtype=np.int32),
                                                              opt=self.opt, atom_amount=3)


class CalculationMethodsTestCase(MetricTestCase):
    def setUp(self) -> None:
        from yeti.get_features.metric import Metric
        from yeti.systems.building_blocks import Atom

        super(CalculationMethodsTestCase, self).setUp()

        class TestCalculate(Metric):
            def __calculate_no_pbc__(self, xyz, indices, opt):
                return xyz

            def __calculate_minimal_image_convention__(self, xyz, indices, opt):
                return xyz

        self.metric = TestCalculate(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                                    unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        self.atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A',
                            xyz_trajectory=np.arange(6).reshape((2, 3)))
        self.atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                            xyz_trajectory=np.arange(6, 12).reshape((2, 3)))


class TestCalculationMethods(CalculationMethodsTestCase):
    def test_calculate_opt_false(self):
        res_feat = self.metric.calculate(atoms=(self.atom_01, self.atom_02), opt=False)
        exp_feat = np.array([0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11], dtype=float)

        npt.assert_array_equal(exp_feat, res_feat)

    def test_calculate_opt_true(self):
        res_feat = self.metric.calculate(atoms=(self.atom_01, self.atom_02), opt=True)
        exp_feat = np.array([0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11], dtype=float)

        npt.assert_array_equal(exp_feat, res_feat)


class MetricExceptionsTestCase(MetricTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.get_features.metric import MetricException

        super(MetricExceptionsTestCase, self).setUp()

        self.exception = MetricException


class TestStandardMethodExceptions(MetricExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.get_features.metric import Metric

        super(TestStandardMethodExceptions, self).setUp()
        self.metric = Metric

    def test_init_periodic_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.metric(periodic=42, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        desired_msg = self.create_data_type_exception_messages(parameter_name='periodic', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.metric(periodic=True, unit_cell_angles=[[90, 90, 90]],
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        desired_msg = self.create_data_type_exception_messages(parameter_name='unit_cell_angles',
                                                               data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

        desired_msg = self.create_data_type_exception_messages(parameter_name='unit_cell_vectors',
                                                               data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_shape(self):
        with self.assertRaises(self.exception) as context:
            self.metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        desired_msg = self.create_array_shape_exception_messages(parameter_name='unit_cell_angles',
                                                                 desired_shape=(None, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_unit_cell_vectors_wrong_shape(self):
        with self.assertRaises(self.exception) as context:
            self.metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]]))

        desired_msg = self.create_array_shape_exception_messages(parameter_name='unit_cell_vectors',
                                                                 desired_shape=(None, 3, 3))
        self.assertEqual(desired_msg, str(context.exception))


class TestPreparationMethodExceptions(MetricExceptionsTestCase):
    def test_prepare_xyz_wrong_parameter_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__prepare_xyz_data__(atoms=[1, 2, 3])

        desired_msg = self.create_data_type_exception_messages(parameter_name='atoms', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_prepare_xyz_wrong_data_type_atom(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__prepare_xyz_data__(atoms=(1, 2))

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom in tuple atoms',
                                                               data_type_name='Atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_prepare_atom_indices_unknown_amount(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__prepare_atom_indices__(5)

        desired_msg = 'Invalid amount.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_prepare_atom_indices_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__prepare_atom_indices__('2')

        desired_msg = self.create_data_type_exception_messages(parameter_name='amount', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))


class TestCompatibilityCheckExceptions(CompatibilityChecksTestCase, MetricExceptionsTestCase):
    def test_mdtraj_paramaeter_compatibility_check_xyz_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__mdtraj_paramaeter_compatibility_check__(xyz=[[[0, 0, 0], [1, 0, 0]]],
                                                                  indices=self.indices, opt=self.opt, atom_amount=2)

        desired_msg = self.create_data_type_exception_messages(parameter_name='xyz', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_xyz_wrong_dtype(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__mdtraj_paramaeter_compatibility_check__(xyz=np.array([[[0, 0, 0], [1, 0, 0]]],
                                                                               dtype=np.float64),
                                                                  indices=self.indices, opt=self.opt, atom_amount=2)

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='xyz', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_xyz_wrong_shape_atom_dim(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__mdtraj_paramaeter_compatibility_check__(xyz=self.xyz,
                                                                  indices=np.array([[0, 1], [2, 3]], dtype=np.int32),
                                                                  opt=self.opt, atom_amount=2)

        desired_msg = self.create_array_shape_exception_messages(parameter_name='xyz', desired_shape=(None, 4, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_xyz_wrong_shape_xyz_dim(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__mdtraj_paramaeter_compatibility_check__(xyz=np.array([[[0, 0, 0, 0], [1, 0, 0, 0]]],
                                                                               dtype=np.float32),
                                                                  indices=self.indices, opt=self.opt, atom_amount=2)

        desired_msg = self.create_array_shape_exception_messages(parameter_name='xyz', desired_shape=(None, 2, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_indices_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__mdtraj_paramaeter_compatibility_check__(xyz=self.xyz, indices=[[0, 1]],
                                                                  opt=self.opt, atom_amount=2)

        desired_msg = self.create_data_type_exception_messages(parameter_name='indices', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_indices_wrong_dtype(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__mdtraj_paramaeter_compatibility_check__(xyz=self.xyz,
                                                                  indices=np.array([[0, 1]], dtype=np.int64),
                                                                  opt=self.opt, atom_amount=2)

        desired_msg = self.create_array_dtype_exception_messages(parameter_name='indices', dtype_name='int32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_indices_wrong_shape(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__mdtraj_paramaeter_compatibility_check__(xyz=self.xyz,
                                                                  indices=np.array([[0, 1, 1]], dtype=np.int32),
                                                                  opt=self.opt, atom_amount=2)

        desired_msg = self.create_array_shape_exception_messages(parameter_name='indices', desired_shape=(None, 2))
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_opt_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__mdtraj_paramaeter_compatibility_check__(xyz=self.xyz, indices=self.indices,
                                                                  opt=42, atom_amount=2)

        desired_msg = self.create_data_type_exception_messages(parameter_name='opt', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_incompatible_amount_xyz(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__mdtraj_paramaeter_compatibility_check__(xyz=np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]],
                                                                               dtype=np.float32),
                                                             indices=self.indices, opt=self.opt, atom_amount=2)

        desired_msg = self.create_array_shape_exception_messages(parameter_name='xyz', desired_shape=(None, 2, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_incompatible_amount_indices(self):
        with self.assertRaises(self.exception) as context:
            self.metric.__mdtraj_paramaeter_compatibility_check__(xyz=self.xyz, indices=np.array([[0, 1, 2]]),
                                                                  opt=self.opt, atom_amount=2)

        desired_msg = self.create_array_shape_exception_messages(parameter_name='indices', desired_shape=(None, 2))
        self.assertEqual(desired_msg, str(context.exception))


class TestCalculationMethodExceptions(CalculationMethodsTestCase, MetricExceptionsTestCase):
    def test_calculate_atoms_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.metric.calculate(atoms=41, opt=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atoms', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_opt_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.metric.calculate(atoms=(self.atom_01, self.atom_02), opt=1)

        desired_msg = self.create_data_type_exception_messages(parameter_name='opt', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
