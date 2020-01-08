import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintTestCase
from tests.systems.two_atoms_molecule_test import TwoAtomsMoleculeExceptionsTestCase


class WaterTestCase(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.systems.building_blocks import Residue
        from yeti.systems.molecules.solvents import Water

        oxygen_01 = Atom(structure_file_index=3, subsystem_index=0, name='OW',
                         xyz_trajectory=np.array([[0.9, 0.9, 0.9], [0, 0, 0]]))
        hydrogen_01 = Atom(structure_file_index=4, subsystem_index=1, name='HW1',
                           xyz_trajectory=np.array([[0.2, 0.1, 0.2], [0.23, 0, 0.75]]))
        hydrogen_02 = Atom(structure_file_index=5, subsystem_index=2, name='HW2',
                           xyz_trajectory=np.array([[0.1, 0.2, 0.1], [0.1, 0.3, 0.4]]))

        self.residue = Residue(subsystem_index=0, structure_file_index=1, name='SOL002')
        self.residue.add_atom(oxygen_01)
        self.residue.add_atom(hydrogen_01)
        self.residue.add_atom(hydrogen_02)
        self.residue.finalize()

        # create box information dictionary
        box_information = dict(unit_cell_angles=np.array([[90, 90, 90], [90, 90, 90]], dtype=np.float32),
                               unit_cell_vectors=np.array(
                                   [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                                   dtype=np.float32))

        self.water = Water(residue=self.residue, periodic=True, box_information=box_information)


class TestWaterStandardMethods(WaterTestCase):
    def test_init(self):
        self.assertTrue(self.water.residue, self.residue)
        self.assertEqual(self.water._internal_id, 0)
        self.assertEqual(self.water._structure_file_id, 1)
        self.assertEqual(self.water.molecule_name, 'SOL002')


class TestDistanceMethod(WaterTestCase):
    def test_get_distance(self):
        self.water.get_distance()

        npt.assert_array_almost_equal(self.water.distances, np.array([0.17320508075, 0.47895720059]), decimal=5)


class TestAngleMethod(WaterTestCase):
    def test_get_angle(self):
        self.water.get_angle()

        npt.assert_array_almost_equal(self.water.angles, np.array([0.37432076, 2.03144108]), decimal=5)


class WaterExceptionsTestCase(TwoAtomsMoleculeExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.systems.molecules.solvents import Water

        super(WaterExceptionsTestCase, self).setUp()

        self.molecule = Water


class TestWaterStandardMethodsExceptions(WaterExceptionsTestCase):
    def test_init_residue_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residue=4.2)

        desired_msg = self.create_data_type_exception_messages(parameter_name='residue', data_type_name='Residue')
        self.assertEqual(desired_msg, str(context.exception))
