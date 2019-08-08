import unittest

import numpy as np
import numpy.testing as npt

from test_utils.test_utils import create_data_type_exception_messages, create_array_shape_exception_messages, \
    create_array_dtype_exception_messages


class TestMetric(unittest.TestCase):
    def test_init(self):
        from yeti.get_features.metric import Metric, MetricException

        unit_cell_angles = np.array([[90, 90, 90]])
        unit_cell_vectors = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

        metric = Metric(periodic=True, unit_cell_angles=unit_cell_angles, unit_cell_vectors=unit_cell_vectors)

        self.assertEqual(metric.ensure_data_type.exception_class, MetricException)
        self.assertTrue(metric.periodic)
        npt.assert_array_equal(metric.unit_cell_angles, unit_cell_angles)
        npt.assert_array_equal(metric.unit_cell_vectors, unit_cell_vectors)

    def test_init_periodic_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException

        with self.assertRaises(MetricException) as context:
            Metric(periodic=42, unit_cell_angles=np.array([[90, 90, 90]]),
                   unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        desired_msg = create_data_type_exception_messages(parameter_name='periodic', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException

        with self.assertRaises(MetricException) as context:
            Metric(periodic=True, unit_cell_angles=[[90, 90, 90]],
                   unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        desired_msg = create_data_type_exception_messages(parameter_name='unit_cell_angles', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_vectors_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException

        with self.assertRaises(MetricException) as context:
            Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                   unit_cell_vectors=[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

        desired_msg = create_data_type_exception_messages(parameter_name='unit_cell_vectors', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_cell_angles_wrong_shape(self):
        from yeti.get_features.metric import Metric, MetricException

        with self.assertRaises(MetricException) as context:
            Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90, 90]]),
                   unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        desired_msg = create_array_shape_exception_messages(parameter_name='unit_cell_angles', desired_shape=(None, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_unit_unit_cell_vectors_wrong_shape(self):
        from yeti.get_features.metric import Metric, MetricException

        with self.assertRaises(MetricException) as context:
            Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                   unit_cell_vectors=np.array([[[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]]]))

        desired_msg = create_array_shape_exception_messages(parameter_name='unit_cell_vectors',
                                                            desired_shape=(None, 3, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_finalized(self):
        from yeti.get_features.metric import Metric
        from yeti.systems.building_blocks import Atom, Residue

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')
        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_03 = Atom(structure_file_index=3, subsystem_index=2, name='C',
                       xyz_trajectory=np.arange(12, 18).reshape((2, 3)))

        residue.add_atom(atom=atom_01)
        residue.add_atom(atom=atom_02)
        residue.add_atom(atom=atom_03)
        residue.finalize()

        res_atom = metric.__get_atom__(name='B', residue=residue)
        self.assertEqual(res_atom, atom_02)

    def test_get_atom_not_finalized(self):
        from yeti.get_features.metric import Metric
        from yeti.systems.building_blocks import Atom, Residue

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')
        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_03 = Atom(structure_file_index=3, subsystem_index=2, name='C',
                       xyz_trajectory=np.arange(12, 18).reshape((2, 3)))

        residue.add_atom(atom=atom_01)
        residue.add_atom(atom=atom_02)
        residue.add_atom(atom=atom_03)

        res_atom = metric.__get_atom__(name='C', residue=residue)
        self.assertEqual(res_atom, atom_03)

    def test_get_atom_name_not_exist(self):
        from yeti.get_features.metric import Metric, MetricException
        from yeti.systems.building_blocks import Atom, Residue

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')
        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_03 = Atom(structure_file_index=3, subsystem_index=2, name='C',
                       xyz_trajectory=np.arange(12, 18).reshape((2, 3)))

        residue.add_atom(atom=atom_01)
        residue.add_atom(atom=atom_02)
        residue.add_atom(atom=atom_03)
        residue.finalize()

        with self.assertRaises(MetricException) as context:
            metric.__get_atom__(name='Z', residue=residue)

        desired_msg = 'Atom does not exist.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_duplicates(self):
        from yeti.get_features.metric import Metric, MetricException
        from yeti.systems.building_blocks import Atom, Residue

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')
        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_03 = Atom(structure_file_index=3, subsystem_index=2, name='A',
                       xyz_trajectory=np.arange(12, 18).reshape((2, 3)))

        residue.add_atom(atom=atom_01)
        residue.add_atom(atom=atom_02)
        residue.add_atom(atom=atom_03)
        residue.finalize()

        with self.assertRaises(MetricException) as context:
            metric.__get_atom__(name='A', residue=residue)

        desired_msg = 'Atom names are not distinguishable. Check your naming or contact the developer.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_name_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException
        from yeti.systems.building_blocks import Atom, Residue

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')
        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_03 = Atom(structure_file_index=3, subsystem_index=2, name='C',
                       xyz_trajectory=np.arange(12, 18).reshape((2, 3)))

        residue.add_atom(atom=atom_01)
        residue.add_atom(atom=atom_02)
        residue.add_atom(atom=atom_03)
        residue.finalize()

        with self.assertRaises(MetricException) as context:
            metric.__get_atom__(name=42, residue=residue)

        desired_msg = create_data_type_exception_messages(parameter_name='name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_residue_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        with self.assertRaises(MetricException) as context:
            metric.__get_atom__(name='A', residue=[])

        desired_msg = create_data_type_exception_messages(parameter_name='residue', data_type_name='Residue')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atoms_same_residue(self):
        from yeti.get_features.metric import Metric
        from yeti.systems.building_blocks import Atom, Residue

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')
        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A',
                       xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_03 = Atom(structure_file_index=3, subsystem_index=2, name='C',
                       xyz_trajectory=np.arange(12, 18).reshape((2, 3)))

        residue.add_atom(atom=atom_01)
        residue.add_atom(atom=atom_02)
        residue.add_atom(atom=atom_03)
        residue.finalize()

        res_atoms = metric.__get_atoms__(atom_name_residue_pairs=(('A', residue), ('C', residue)))

        self.assertTupleEqual(res_atoms, (atom_01, atom_03))

    def test_get_atoms_different_residues(self):
        from yeti.get_features.metric import Metric
        from yeti.systems.building_blocks import Atom, Residue

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        residue_01 = Residue(subsystem_index=0, structure_file_index=1, name='TEST 01')
        residue_02 = Residue(subsystem_index=1, structure_file_index=2, name='TEST 02')
        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A',
                       xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))
        atom_03 = Atom(structure_file_index=3, subsystem_index=2, name='C',
                       xyz_trajectory=np.arange(12, 18).reshape((2, 3)))

        residue_02.add_atom(atom=atom_01)
        residue_02.add_atom(atom=atom_02)
        residue_02.finalize()

        residue_01.add_atom(atom=atom_03)
        residue_01.finalize()

        res_atoms = metric.__get_atoms__(atom_name_residue_pairs=(('A', residue_02), ('C', residue_01)))

        self.assertTupleEqual(res_atoms, (atom_01, atom_03))

    def test_get_atoms_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        with self.assertRaises(MetricException) as context:
            metric.__get_atoms__(atom_name_residue_pairs=42)

        desired_msg = create_data_type_exception_messages(parameter_name='atom_name_residue_pairs',
                                                          data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_prepare_xyz_data(self):
        from yeti.get_features.metric import Metric
        from yeti.systems.building_blocks import Atom

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A',
                       xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        res_xyz = metric.__prepare_xyz_data__((atom_01, atom_02))
        exp_xyz = np.array([[[0, 1, 2], [6, 7, 8]], [[3, 4, 5], [9, 10, 11]]])

        npt.assert_array_equal(res_xyz, exp_xyz)

    def test_prepare_xyz_wrong_parameter_data_type(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        with self.assertRaises(MetricException) as context:
            metric.__prepare_xyz_data__(atoms=[1, 2, 3])

        desired_msg = create_data_type_exception_messages(parameter_name='atoms', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_prepare_xyz_wrong_data_type_atom(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        with self.assertRaises(MetricException) as context:
            metric.__prepare_xyz_data__(atoms=(1, 2))

        desired_msg = create_data_type_exception_messages(parameter_name='atom in tuple atoms', data_type_name='Atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_prepare_atom_indices_amount_two(self):
        from yeti.get_features.metric import Metric

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        res_amount = metric.__prepare_atom_indices__(2)

        npt.assert_array_equal(np.array([[0, 1]]), res_amount)

    def test_prepare_atom_indices_amount_three(self):
        from yeti.get_features.metric import Metric

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        res_amount = metric.__prepare_atom_indices__(3)

        npt.assert_array_equal(np.array([[0, 1, 2]]), res_amount)

    def test_prepare_atom_indices_amount_four(self):
        from yeti.get_features.metric import Metric

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        res_amount = metric.__prepare_atom_indices__(4)

        npt.assert_array_equal(np.array([[0, 1, 2, 3]]), res_amount)

    def test_prepare_atom_indices_unknwon_amount(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        with self.assertRaises(MetricException) as context:
            metric.__prepare_atom_indices__(5)

        desired_msg = 'Invalid amount.'
        self.assertEqual(desired_msg, str(context.exception))

    def test_prepare_atom_indices_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        with self.assertRaises(MetricException) as context:
            metric.__prepare_atom_indices__('2')

        desired_msg = create_data_type_exception_messages(parameter_name='amount', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_amount_two(self):
        from yeti.get_features.metric import Metric

        metric = Metric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)
        opt = False

        metric.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt, atom_amount=2)

    def test_mdtraj_paramaeter_compatibility_check_amount_three(self):
        from yeti.get_features.metric import Metric

        metric = Metric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)
        opt = False

        metric.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt, atom_amount=3)

    def test_mdtraj_paramaeter_compatibility_check_xyz_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = [[[0, 0, 0], [1, 0, 0]]]
        indices = np.array([[0, 1]], dtype=np.int32)
        opt = False

        with self.assertRaises(MetricException) as context:
            metric.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt, atom_amount=2)

        desired_msg = create_data_type_exception_messages(parameter_name='xyz', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_xyz_wrong_dtype(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float64)
        indices = np.array([[0, 1]], dtype=np.int32)
        opt = False

        with self.assertRaises(MetricException) as context:
            metric.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt, atom_amount=2)

        desired_msg = create_array_dtype_exception_messages(parameter_name='xyz', dtype_name='float32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_xyz_wrong_shape_atom_dim(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1], [2, 3]], dtype=np.int32)
        opt = False

        with self.assertRaises(MetricException) as context:
            metric.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt, atom_amount=2)

        desired_msg = create_array_shape_exception_messages(parameter_name='xyz', desired_shape=(None, 4, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_xyz_wrong_shape_xyz_dim(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0, 0], [1, 0, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)
        opt = False

        with self.assertRaises(MetricException) as context:
            metric.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt, atom_amount=2)

        desired_msg = create_array_shape_exception_messages(parameter_name='xyz', desired_shape=(None, 2, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_indices_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        indices = [[0, 1]]
        opt = False

        with self.assertRaises(MetricException) as context:
            metric.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt, atom_amount=2)

        desired_msg = create_data_type_exception_messages(parameter_name='indices', data_type_name='ndarray')
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_indices_wrong_dtype(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int64)
        opt = False

        with self.assertRaises(MetricException) as context:
            metric.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt, atom_amount=2)

        desired_msg = create_array_dtype_exception_messages(parameter_name='indices', dtype_name='int32')
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_indices_wrong_shape(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1, 1]], dtype=np.int32)
        opt = False

        with self.assertRaises(MetricException) as context:
            metric.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt, atom_amount=2)

        desired_msg = create_array_shape_exception_messages(parameter_name='indices', desired_shape=(None, 2))
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_opt_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)
        opt = 42

        with self.assertRaises(MetricException) as context:
            metric.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt, atom_amount=2)

        desired_msg = create_data_type_exception_messages(parameter_name='opt', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_incompatible_amount_xyz(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int32)
        opt = False

        with self.assertRaises(MetricException) as context:
            metric.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt, atom_amount=3)

        desired_msg = create_array_shape_exception_messages(parameter_name='xyz', desired_shape=(None, 3, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_mdtraj_paramaeter_compatibility_check_incompatible_amount_indices(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=False, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))

        xyz = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)
        indices = np.array([[0, 1]], dtype=np.int32)
        opt = False

        with self.assertRaises(MetricException) as context:
            metric.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt, atom_amount=3)

        desired_msg = create_array_shape_exception_messages(parameter_name='indices', desired_shape=(None, 3))
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_opt_false(self):
        from yeti.get_features.metric import Metric
        from yeti.systems.building_blocks import Atom

        class TestCalculate(Metric):
            def __calculate_no_pbc__(self, xyz, indices, opt):
                return xyz

            def __calculate_minimal_image_convention__(self, xyz, indices, opt):
                return xyz

        metric = TestCalculate(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                               unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A',
                       xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        res_feat = metric.__calculate__(atoms=(atom_01, atom_02), opt=False)
        exp_feat = np.array([0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11], dtype=float)

        npt.assert_array_equal(exp_feat, res_feat)

    def test_calculate_opt_true(self):
        from yeti.get_features.metric import Metric
        from yeti.systems.building_blocks import Atom

        class TestCalculate(Metric):
            def __calculate_no_pbc__(self, xyz, indices, opt):
                return xyz

            def __calculate_minimal_image_convention__(self, xyz, indices, opt):
                return xyz

        metric = TestCalculate(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                               unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A',
                       xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        res_feat = metric.__calculate__(atoms=(atom_01, atom_02), opt=True)
        exp_feat = np.array([0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11], dtype=float)

        npt.assert_array_equal(exp_feat, res_feat)

    def test_calculate_atoms_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException
        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        with self.assertRaises(MetricException) as context:
            metric.__calculate__(atoms=41, opt=True)

        desired_msg = create_data_type_exception_messages(parameter_name='atoms', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_calculate_opt_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException
        from yeti.systems.building_blocks import Atom

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A',
                       xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        with self.assertRaises(MetricException) as context:
            metric.__calculate__(atoms=(atom_01, atom_02), opt=1)

        desired_msg = create_data_type_exception_messages(parameter_name='opt', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get(self):
        from yeti.get_features.metric import Metric
        from yeti.systems.building_blocks import Atom, Residue

        class TestGet(Metric):
            def __calculate_no_pbc__(self, xyz, indices, opt):
                return xyz

            def __calculate_minimal_image_convention__(self, xyz, indices, opt):
                return xyz

        metric = TestGet(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                         unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')
        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        residue.add_atom(atom=atom_01)
        residue.add_atom(atom=atom_02)
        residue.finalize()

        res_feat = metric.get(atom_name_residue_pairs=(('A', residue), ('B', residue)), opt=False)
        exp_feat = np.array([0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11], dtype=float)

        npt.assert_array_equal(exp_feat, res_feat)

    def test_get_opt_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException
        from yeti.systems.building_blocks import Atom, Residue

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        residue = Residue(subsystem_index=0, structure_file_index=1, name='test')
        atom_01 = Atom(structure_file_index=1, subsystem_index=0, name='A', xyz_trajectory=np.arange(6).reshape((2, 3)))
        atom_02 = Atom(structure_file_index=2, subsystem_index=1, name='B',
                       xyz_trajectory=np.arange(6, 12).reshape((2, 3)))

        residue.add_atom(atom=atom_01)
        residue.add_atom(atom=atom_02)
        residue.finalize()

        with self.assertRaises(MetricException) as context:
            metric.get(atom_name_residue_pairs=(('A', residue), ('B', residue)), opt=2)

        desired_msg = create_data_type_exception_messages(parameter_name='opt', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atom_name_residue_pairs_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        with self.assertRaises(MetricException) as context:
            metric.get(atom_name_residue_pairs=2.6, opt=True)

        desired_msg = create_data_type_exception_messages(parameter_name='atom_name_residue_pairs',
                                                          data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
