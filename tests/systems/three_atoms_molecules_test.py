import unittest

import numpy as np
import numpy.testing as npt

from tests.systems.two_atoms_molecule_test import TwoAtomsInitTest, MoleculesTest


class ThreeAtomsInitTest(TwoAtomsInitTest):
    def setUp(self) -> None:
        super(ThreeAtomsInitTest, self).setUp()

        from yeti.systems.molecules.molecules import ThreeAtomsMolecule
        self.molecule = ThreeAtomsMolecule

    def test_init(self):
        from yeti.get_features.angles import Angle

        super(ThreeAtomsInitTest, self).test_init()

        self.assertEqual(Angle, type(self.working_molecule._angle))
        self.assertDictEqual({}, self.working_molecule.angles)


class ThreeAtomsMoleculeTest(MoleculesTest):
    def setUp(self) -> None:
        from yeti.systems.molecules.molecules import ThreeAtomsMolecule
        residues, box_information = super(ThreeAtomsMoleculeTest, self).setUp()

        self.molecule = ThreeAtomsMolecule(residues=residues, molecule_name='test',
                                           box_information=box_information)


class GetAngleTest(ThreeAtomsMoleculeTest):
    def test_store(self):
        self.molecule.get_angle(atom_01_pos=(0, 0), atom_02_pos=(0, 1), atom_03_pos=(1, 0), store_result=True, opt=True)
        exp_key = 'RESA_0000:A_0000-RESA_0000:B_0001-RESB_0001:C_0002'
        exp_dict = {exp_key: np.array([1.2490457, 1.5707961, 0.50662816])}

        self.assertEqual(self.molecule.angles.keys(), exp_dict.keys())
        npt.assert_array_almost_equal(self.molecule.angles[exp_key], exp_dict[exp_key], decimal=5)

    def test_not_store(self):
        res = self.molecule.get_angle(atom_01_pos=(0, 0), atom_02_pos=(0, 1), atom_03_pos=(1, 0), store_result=False,
                                      opt=True)

        npt.assert_array_almost_equal(np.array([1.2490457, 1.5707961, 0.50662816]), res, decimal=5)

    def test_atom_01_pos_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_angle(atom_01_pos=[0, 0], atom_02_pos=(0, 1), atom_03_pos=(1, 0), store_result=True,
                                    opt=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_01_pos', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_atom_02_pos_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_angle(atom_01_pos=(0, 0), atom_02_pos=[0, 1], atom_03_pos=(1, 0), store_result=True,
                                    opt=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_02_pos', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_atom_03_pos_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_angle(atom_01_pos=(0, 0), atom_02_pos=(0, 1), atom_03_pos=[1, 0], store_result=True,
                                    opt=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_03_pos', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_store_result_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_angle(atom_01_pos=(0, 0), atom_02_pos=(0, 1), atom_03_pos=(1, 0), store_result=42,
                                    opt=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='store_result', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_opt_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_angle(atom_01_pos=(0, 0), atom_02_pos=(0, 1), atom_03_pos=(1, 0), store_result=True,
                                    opt=12)

        desired_msg = self.create_data_type_exception_messages(parameter_name='opt', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()