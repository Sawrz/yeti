import unittest

import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintExceptionsTestCase, BlueprintTestCase
from tests.get_features.triplet_test import TripletTestCase


class TwoAtomsMoleculeTestCase(TripletTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Residue
        from yeti.systems.molecules.molecules import TwoAtomsMolecule

        super(TwoAtomsMoleculeTestCase, self).setUp()

        # add systems
        self.donor_atom.add_system(system_name='test_system')
        self.acceptor.add_system(system_name='test_system')

        # create residues
        self.residue_01 = Residue(subsystem_index=0, structure_file_index=4, name='RESA')
        self.residue_02 = Residue(subsystem_index=1, structure_file_index=5, name='RESB')

        self.residue_01.add_atom(atom=self.donor)
        self.donor.set_residue(residue=self.residue_01)

        self.residue_01.add_atom(atom=self.donor_atom)
        self.donor_atom.set_residue(residue=self.residue_01)

        self.residue_02.add_atom(atom=self.acceptor)
        self.acceptor.set_residue(residue=self.residue_02)

        self.residue_01.finalize()
        self.residue_02.finalize()

        self.residues = (self.residue_01, self.residue_02)

        self.molecule_name = 'test'

        # create box information dictionary
        self.box_information = dict(unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)

        # initialize object
        self.atoms = (self.donor, self.donor_atom, self.acceptor)
        self.molecule = TwoAtomsMolecule(residues=self.residues, molecule_name=self.molecule_name,
                                         box_information=self.box_information, periodic=True)


class TestTwoAtomsStandardMethods(TwoAtomsMoleculeTestCase):
    def test_init(self):
        from yeti.systems.molecules.molecules import MoleculeException
        from yeti.get_features.distances import Distance

        self.assertEqual(MoleculeException, self.molecule.ensure_data_type.exception_class)
        self.assertEqual(self.molecule_name, self.molecule.molecule_name)
        self.assertTupleEqual(self.residues, self.molecule.residues)
        self.assertEqual(Distance, type(self.molecule._dist))
        self.assertDictEqual({}, self.molecule.distances)


class TestGenerateKeyMethods(TwoAtomsMoleculeTestCase):
    def test_get_atom_key_name(self):
        res = self.molecule.__get_atom_key_name__(atom=self.donor)

        self.assertEqual('RESA_0000:A_0000', res)

    def test_generate_key_two_atoms(self):
        res = self.molecule.__generate_key__(atoms=(self.donor, self.donor_atom))

        self.assertEqual('RESA_0000:A_0000-RESA_0000:B_0001', res)

    def test_generate_key_three_atoms(self):
        res = self.molecule.__generate_key__(atoms=self.atoms)

        self.assertEqual('RESA_0000:A_0000-RESA_0000:B_0001-RESB_0001:C_0002', res)


class TestAtomMethods(TwoAtomsMoleculeTestCase):
    def test_get_atom(self):
        res = self.molecule.__get_atom__(atom_pos=(1, 0))
        self.assertEqual(self.acceptor, res)

    def test_get_atoms(self):
        res = self.molecule.__get_atoms__(atom_positions=((0, 0), (1, 0)))
        self.assertEqual((self.donor, self.acceptor), res)


class TestXyzMethods(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.systems.building_blocks import Residue
        from yeti.systems.molecules.molecules import TwoAtomsMolecule

        super(TestXyzMethods, self).setUp()

        # CREATE ATOMS
        self.atom_01 = Atom(structure_file_index=2, subsystem_index=0, name='A',
                            xyz_trajectory=np.array([[0.1, 0.4, 0.3]]))
        self.atom_02 = Atom(structure_file_index=3, subsystem_index=1, name='B',
                            xyz_trajectory=np.array([[0.1, 0.5, 0.2]]))
        self.atom_03 = Atom(structure_file_index=4, subsystem_index=2, name='C',
                            xyz_trajectory=np.array([[0.1, 0.6, 0.4]]))
        self.atom_04 = Atom(structure_file_index=5, subsystem_index=3, name='D',
                            xyz_trajectory=np.array([[0.1, 0.7, 0.7]]))

        # create residues
        self.residue_01 = Residue(subsystem_index=0, structure_file_index=4, name='RESA')
        self.residue_02 = Residue(subsystem_index=1, structure_file_index=5, name='RESB')

        self.residue_01.add_atom(atom=self.atom_01)
        self.atom_01.set_residue(residue=self.residue_01)

        self.residue_01.add_atom(atom=self.atom_02)
        self.atom_02.set_residue(residue=self.residue_01)

        self.residue_02.add_atom(atom=self.atom_03)
        self.atom_03.set_residue(residue=self.residue_02)

        self.residue_02.add_atom(atom=self.atom_04)
        self.atom_04.set_residue(residue=self.residue_02)

        self.residue_01.finalize()
        self.residue_02.finalize()

        self.residues = (self.residue_01, self.residue_02)

        self.molecule_name = 'test'

        # CREATE UNIT CELL PROPERTIES
        self.unit_cell_angles = np.array([[90, 90, 90], [90, 90, 90], [90, 90, 90]], dtype=np.float32)
        self.unit_cell_vectors = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
            dtype=np.float32)

        self.box_information = dict(unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)

        # initialize object
        self.atoms = (self.atom_01, self.atom_02, self.atom_03, self.atom_04)
        self.molecule = TwoAtomsMolecule(residues=self.residues, molecule_name=self.molecule_name,
                                         box_information=self.box_information, periodic=True)

        # Expected solutions
        self.exp_xyz = np.array([[[0.1, 0.4, 0.3], [0.1, 0.5, 0.2], [0.1, 0.6, 0.4], [0.1, 0.7, 0.7]],
                                 [[0.1, 0.4, 0.3], [0.1, 0.5, 0.2], [0.1, 0.6, 0.4], [0.1, 0.7, 0.7]]])

    def add_new_frame(self, shift):
        for atom in self.atoms:
            atom.add_frame(frame=np.round(atom.xyz_trajectory[0] + shift, decimals=5))

    def test_get_xyz_one_frame(self):
        res_xyz, res_names = self.molecule.get_xyz()

        exp_xyz = np.array([[[0.1, 0.4, 0.3], [0.1, 0.5, 0.2], [0.1, 0.6, 0.4], [0.1, 0.7, 0.7]]])

        self.assertListEqual(['RESA4_A', 'RESA4_B', 'RESB5_C', 'RESB5_D'], res_names)
        npt.assert_array_equal(res_xyz, exp_xyz)

    def test_get_xyz_multi_frames(self):
        self.add_new_frame(shift=np.array([0.1, 0.2, -0.1]))

        res_xyz, res_names = self.molecule.get_xyz()

        exp_xyz = np.array([[[0.1, 0.4, 0.3], [0.1, 0.5, 0.2], [0.1, 0.6, 0.4], [0.1, 0.7, 0.7]],
                            [[0.2, 0.6, 0.2], [0.2, 0.7, 0.1], [0.2, 0.8, 0.3], [0.2, 0.9, 0.6]]])

        self.assertListEqual(['RESA4_A', 'RESA4_B', 'RESB5_C', 'RESB5_D'], res_names)
        npt.assert_array_equal(res_xyz, exp_xyz)

    def test_get_xyz_aligned_non_periodic_x_ref_frame_0(self):
        self.add_new_frame(shift=np.array([0.1, 0, 0]))

        res_xyz = self.molecule.get_aligned_xyz(reference_frame=0)

        npt.assert_array_equal(res_xyz, self.exp_xyz)

    def test_get_xyz_aligned_non_periodic_y_ref_frame_0(self):
        self.add_new_frame(shift=np.array([0, 0.1, 0]))

        res_xyz = self.molecule.get_aligned_xyz(reference_frame=0)

        npt.assert_array_equal(res_xyz, self.exp_xyz)

    def test_get_xyz_aligned_non_periodic_z_ref_frame_0(self):
        self.add_new_frame(shift=np.array([0, 0, 0.1]))

        res_xyz = self.molecule.get_aligned_xyz(reference_frame=0)

        npt.assert_array_equal(res_xyz, self.exp_xyz)

    def test_get_xyz_aligned_non_periodic_x_ref_frame_1(self):
        self.add_new_frame(shift=np.array([0.1, 0, 0]))

        res_xyz = self.molecule.get_aligned_xyz(reference_frame=1)

        exp_xyz = np.array([[[0.2, 0.4, 0.3], [0.2, 0.5, 0.2], [0.2, 0.6, 0.4], [0.2, 0.7, 0.7]],
                            [[0.2, 0.4, 0.3], [0.2, 0.5, 0.2], [0.2, 0.6, 0.4], [0.2, 0.7, 0.7]]])

        npt.assert_array_equal(res_xyz, exp_xyz)

    def test_get_xyz_aligned_non_periodic_y_ref_frame_1(self):
        self.add_new_frame(shift=np.array([0, 0.1, 0]))

        res_xyz = self.molecule.get_aligned_xyz(reference_frame=1)

        exp_xyz = np.array([[[0.1, 0.5, 0.3], [0.1, 0.6, 0.2], [0.1, 0.7, 0.4], [0.1, 0.8, 0.7]],
                            [[0.1, 0.5, 0.3], [0.1, 0.6, 0.2], [0.1, 0.7, 0.4], [0.1, 0.8, 0.7]]])

        npt.assert_array_equal(res_xyz, exp_xyz)

    def test_get_xyz_aligned_non_periodic_z_ref_frame_1(self):
        self.add_new_frame(shift=np.array([0, 0, 0.1]))

        res_xyz = self.molecule.get_aligned_xyz(reference_frame=1)

        exp_xyz = np.array([[[0.1, 0.4, 0.4], [0.1, 0.5, 0.3], [0.1, 0.6, 0.5], [0.1, 0.7, 0.8]],
                            [[0.1, 0.4, 0.4], [0.1, 0.5, 0.3], [0.1, 0.6, 0.5], [0.1, 0.7, 0.8]]])

        npt.assert_array_equal(res_xyz, exp_xyz)

    def test_get_xyz_aligned_non_periodic_xyz_equal_shift(self):
        self.add_new_frame(shift=np.array([0.1, 0.1, 0.1]))

        res_xyz = self.molecule.get_aligned_xyz(reference_frame=0)

        npt.assert_array_equal(res_xyz, self.exp_xyz)

    def test_get_xyz_aligned_non_periodic_xyz_unequal_shift(self):
        self.add_new_frame(shift=np.array([0.5, 0.1, -0.2]))

        res_xyz = self.molecule.get_aligned_xyz(reference_frame=0)

        npt.assert_array_equal(res_xyz, self.exp_xyz)

    def test_get_xyz_aligned_non_periodic_xyz_rotated(self):
        self.atom_01.add_frame(frame=np.array([0.30198, 0.44656, 0.16514]))
        self.atom_02.add_frame(frame=np.array([0.30018, 0.53183, 0.12487]))
        self.atom_03.add_frame(frame=np.array([0.41968, 0.62204, 0.22082]))
        self.atom_04.add_frame(frame=np.array([0.5796, 0.71391, 0.36219]))

        res_xyz = self.molecule.get_aligned_xyz(reference_frame=0)

        npt.assert_array_equal(res_xyz, self.exp_xyz)

    def test_get_xyz_aligned_non_periodic_xyz_rotated_and_translated(self):
        self.atom_01.add_frame(frame=np.array([0.56344, 0.2833, 0.51386]))
        self.atom_02.add_frame(frame=np.array([0.70793, 0.47808, 0.6454]))
        self.atom_03.add_frame(frame=np.array([0.68114, 0.45878, 0.56954]))
        self.atom_04.add_frame(frame=np.array([0.84106, 0.55065, 0.71091]))

        res_xyz = self.molecule.get_aligned_xyz(reference_frame=0)

        npt.assert_array_equal(res_xyz, self.exp_xyz)

    def test_get_xyz_aligned_non_periodic_xyz_translated_and_rotated(self):
        self.atom_01.add_frame(frame=np.array([0.42266, 0.64481, 0.27589]))
        self.atom_02.add_frame(frame=np.array([0.42086, 0.73007, 0.23563]))
        self.atom_03.add_frame(frame=np.array([0.54036, 0.82029, 0.33158]))
        self.atom_04.add_frame(frame=np.array([0.70028, 0.91216, 0.47294]))

        res_xyz = self.molecule.get_aligned_xyz(reference_frame=0)

        npt.assert_array_equal(res_xyz, self.exp_xyz)

class TestDistanceMethods(TwoAtomsMoleculeTestCase):
    def test_store(self):
        self.molecule.get_distance(atom_01_pos=(0, 0), atom_02_pos=(1, 0), store_result=True, opt=True)
        exp_key = 'RESA_0000:A_0000-RESB_0001:C_0002'
        exp_dict = {exp_key: np.array([0.22360681, 0.31622773, 0.22360681])}

        self.assertEqual(self.molecule.distances.keys(), exp_dict.keys())
        npt.assert_array_almost_equal(self.molecule.distances[exp_key], exp_dict[exp_key], decimal=5)

    def test_not_store(self):
        res = self.molecule.get_distance(atom_01_pos=(0, 0), atom_02_pos=(1, 0), store_result=False, opt=True)

        npt.assert_array_almost_equal(np.array([0.22360681, 0.31622773, 0.22360681]), res, decimal=5)


class TwoAtomsMoleculeExceptionsTestCase(TwoAtomsMoleculeTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.systems.molecules.molecules import MoleculeException

        super(TwoAtomsMoleculeExceptionsTestCase, self).setUp()
        self.exception = MoleculeException


class TestTwoAtomsStandardMethodExceptions(TwoAtomsMoleculeExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.systems.molecules.molecules import TwoAtomsMolecule

        super(TestTwoAtomsStandardMethodExceptions, self).setUp()

        self.molecule = TwoAtomsMolecule

    def test_init_residues_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residues=4.2, molecule_name=self.molecule_name, box_information=self.box_information,
                          periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='residues', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_molecule_name_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residues=self.residues, molecule_name=42, box_information=self.box_information, periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='molecule_name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_box_information_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residues=self.residues, molecule_name=self.molecule_name, box_information=[], periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='box_information', data_type_name='dict')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_periodic_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residues=self.residues, molecule_name=self.molecule_name,
                          box_information=self.box_information, periodic='No')

        desired_msg = self.create_data_type_exception_messages(parameter_name='periodic', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))


class TestGenerateKeyMethodExceptions(TwoAtomsMoleculeExceptionsTestCase):
    def test_get_atom_key_name_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.__get_atom_key_name__(atom=42)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom', data_type_name='Atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_generate_key_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.__generate_key__(atoms=[])

        desired_msg = self.create_data_type_exception_messages(parameter_name='atoms', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))


class TestAtomMethodExceptions(TwoAtomsMoleculeExceptionsTestCase):
    def test_get_atom_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.__get_atom__(atom_pos=[1, 0])

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_pos', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atoms_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.__get_atoms__([(0, 0), (1, 0)])

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_positions', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))


class TestDistanceMethodExceptions(TwoAtomsMoleculeExceptionsTestCase):
    def test_atom_01_pos_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_distance(atom_01_pos=[0, 0], atom_02_pos=(1, 0), store_result=False, opt=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_01_pos', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_atom_02_pos_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_distance(atom_01_pos=(0, 0), atom_02_pos=[1, 0], store_result=False, opt=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_02_pos', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_store_result_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_distance(atom_01_pos=(0, 0), atom_02_pos=(1, 0), store_result=42, opt=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='store_result', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_opt_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_distance(atom_01_pos=(0, 0), atom_02_pos=(1, 0), store_result=False, opt=13)

        desired_msg = self.create_data_type_exception_messages(parameter_name='opt', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
