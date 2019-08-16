import unittest


class MyTestCase(unittest.TestCase):
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

        res_atoms = metric.get_atoms(atom_name_residue_pairs=(('A', residue), ('C', residue)))

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

        res_atoms = metric.get_atoms(atom_name_residue_pairs=(('A', residue_02), ('C', residue_01)))

        self.assertTupleEqual(res_atoms, (atom_01, atom_03))

    def test_get_atoms_wrong_data_type(self):
        from yeti.get_features.metric import Metric, MetricException

        metric = Metric(periodic=True, unit_cell_angles=np.array([[90, 90, 90]]),
                        unit_cell_vectors=np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]))

        with self.assertRaises(MetricException) as context:
            metric.get_atoms(atom_name_residue_pairs=42)

        desired_msg = create_data_type_exception_messages(parameter_name='atom_name_residue_pairs',
                                                          data_type_name='tuple')
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


if __name__ == '__main__':
    unittest.main()
