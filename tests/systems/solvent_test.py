import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintTestCase


class SolventTestCase(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.systems.building_blocks import Residue
        from yeti.systems.molecules.solvents import Water, Solvent

        oxygen_01 = Atom(structure_file_index=3, subsystem_index=0, name='OW',
                         xyz_trajectory=np.array([[0.23, 0.3, 0.4], [0.23, 0.3, 0.4]]))
        hydrogen_01 = Atom(structure_file_index=4, subsystem_index=1, name='HW1',
                           xyz_trajectory=np.array([[0.1, 0.31, 0.15], [0.1, 0.31, 0.15]]))
        hydrogen_02 = Atom(structure_file_index=5, subsystem_index=2, name='HW2',
                           xyz_trajectory=np.array([[0.2, 0.1, 0.2], [0.23, 0, 0.75]]))

        residue_01 = Residue(subsystem_index=0, structure_file_index=1, name='SOL002')
        oxygen_01.set_residue(residue_01)
        hydrogen_01.set_residue(residue_01)
        hydrogen_02.set_residue(residue_01)
        residue_01.add_atom(oxygen_01)
        residue_01.add_atom(hydrogen_01)
        residue_01.add_atom(hydrogen_02)
        residue_01.finalize()

        oxygen_01 = Atom(structure_file_index=6, subsystem_index=3, name='OW',
                         xyz_trajectory=np.array([[0.4, 0.22, 0.35], [0.4, 0.22, 0.35]]))
        hydrogen_01 = Atom(structure_file_index=7, subsystem_index=4, name='HW1',
                           xyz_trajectory=np.array([[0.2, 0.19, 0.1], [0.2, 0.19, 0.1]]))
        hydrogen_02 = Atom(structure_file_index=8, subsystem_index=5, name='HW2',
                           xyz_trajectory=np.array([[0.1, 0.2, 0.1], [0.1, 0.3, 0.4]]))

        residue_02 = Residue(subsystem_index=1, structure_file_index=2, name='SOL003')
        oxygen_01.set_residue(residue_02)
        hydrogen_01.set_residue(residue_02)
        hydrogen_02.set_residue(residue_02)
        residue_02.add_atom(oxygen_01)
        residue_02.add_atom(hydrogen_01)
        residue_02.add_atom(hydrogen_02)
        residue_02.finalize()

        # create box information dictionary
        box_information = dict(unit_cell_angles=np.array([[90, 90, 90], [90, 90, 90]], dtype=np.float32),
                               unit_cell_vectors=np.array(
                                   [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                                   dtype=np.float32))

        self.residues = (residue_01, residue_02)
        self.water_01 = Water(residue=residue_01, periodic=True, box_information=box_information)
        self.water_02 = Water(residue=residue_02, periodic=True, box_information=box_information)

        self.solvent = Solvent(solvent_molecules=(self.water_01, self.water_02), solvent_name='H2O',
                               periodic=True, box_information=box_information)


class TestSolventStandardMethods(SolventTestCase):
    def test_init(self):
        # TODO: check for other parameters based on inheritance
        self.assertTupleEqual(self.solvent.residues, self.residues)
        self.assertTupleEqual(self.solvent.molecules, (self.water_01, self.water_02))
        self.assertEqual(self.solvent.solvent_name, 'H2O')


class TestDistanceMethod(SolventTestCase):
    def test_get_distance(self):
        self.solvent.get_distance(atom_name_01='HW2', residue_id_01=0, atom_name_02='HW2', residue_id_02=1)

        exp_dict = {'SOL002_0000:HW2_0002-SOL003_0001:HW2_0005': np.array([0.17320508075, 0.47895720059])}
        self.assertEqual(self.solvent.distances.keys(), exp_dict.keys())

        for key in exp_dict.keys():
            npt.assert_array_almost_equal(self.solvent.distances[key], exp_dict[key], decimal=5)


class TestAngleMethod(SolventTestCase):
    def test_get_angle(self):
        self.solvent.get_angle(atom_name_01='OW', residue_id_01=0, atom_name_02='HW2', residue_id_02=0,
                               atom_name_03='HW2', residue_id_03=1)

        exp_dict = {'SOL002_0000:OW_0000-SOL002_0000:HW2_0002-SOL003_0001:HW2_0005': np.array([1.63172969, 0.27487123])}
        self.assertEqual(self.solvent.angles.keys(), exp_dict.keys())

        for key in exp_dict.keys():
            npt.assert_array_almost_equal(self.solvent.angles[key], exp_dict[key], decimal=5)


class TestDihedralMethod(SolventTestCase):
    def test_get_dihedral(self):
        self.solvent.get_dihedral(atom_name_01='OW', residue_id_01=0, atom_name_02='HW1', residue_id_02=0,
                                  atom_name_03='HW1', residue_id_03=1, atom_name_04='OW', residue_id_04=1)

        exp_dict = {'SOL002_0000:OW_0000-SOL002_0000:HW1_0001-SOL003_0001:HW1_0004-SOL003_0001:OW_0003': np.array(
            [-0.2311416423440, -0.2311416423440])}
        self.assertEqual(self.solvent.dihedral_angles.keys(), exp_dict.keys())

        for key in exp_dict.keys():
            npt.assert_array_almost_equal(self.solvent.dihedral_angles[key], exp_dict[key], decimal=5)
