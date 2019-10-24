import unittest

import numpy as np
import numpy.testing as npt

from tests.systems.three_atoms_molecules_test import TestThreeAtomsStandardMethods
from tests.systems.two_atoms_molecule_test import TwoAtomsMoleculeTestCase, TwoAtomsMoleculeExceptionsTestCase, \
    TestTwoAtomsStandardMethodExceptions


class FourAtomsPlusMoleculeTestCase(TwoAtomsMoleculeTestCase):
    def setUp(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.systems.molecules.molecules import FourAtomsPlusMolecule

        super(FourAtomsPlusMoleculeTestCase, self).setUp()

        self.simulation_information = dict(number_of_frames=3)
        self.hydrogen_bond_information = dict(distance_cutoff=0.25, angle_cutoff=2.0)

        self.additional_atom = Atom(structure_file_index=5, subsystem_index=3, name='D',
                                    xyz_trajectory=np.array([[0.2, 0.4, 0.8], [0.3, 0.7, 0.5], [0.4, 0.7, 0.3]]))
        self.additional_atom.set_residue(residue=self.residue_02)

        self.residue_02.definalize()
        self.residue_02.add_atom(atom=self.additional_atom)
        self.residue_02.finalize()

        self.molecule = FourAtomsPlusMolecule(residues=self.residues, molecule_name=self.molecule_name,
                                              periodic=True, box_information=self.box_information,
                                              simulation_information=self.simulation_information,
                                              hydrogen_bond_information=self.hydrogen_bond_information)


class TestFourAtomsPlusStandardMethods(FourAtomsPlusMoleculeTestCase, TestThreeAtomsStandardMethods):
    def test_init(self):
        from yeti.get_features.angles import Dihedral
        from yeti.get_features.hydrogen_bonds import HydrogenBondsFirstComesFirstServes

        super(TestFourAtomsPlusStandardMethods, self).test_init()

        self.assertEqual(Dihedral, type(self.molecule._dih))
        self.assertDictEqual({}, self.molecule.dihedral_angles)
        self.assertEqual(HydrogenBondsFirstComesFirstServes, type(self.molecule._hbonds))
        self.assertEqual(0.25, self.molecule.distance_cutoff)
        self.assertEqual(2.0, self.molecule.angle_cutoff)

        self.assertIsNone(self.donor.hydrogen_bond_partners)
        self.assertIsNone(self.additional_atom.hydrogen_bond_partners)
        self.assertDictEqual({self.molecule_name: [[self.acceptor], [], []], 'test_system': [[], [], []]},
                             self.donor_atom.hydrogen_bond_partners)
        self.assertDictEqual({self.molecule_name: [[self.donor_atom], [], []], 'test_system': [[], [], []]},
                             self.acceptor.hydrogen_bond_partners)

    def check_atoms(self) -> None:
        self.assertIsNone(self.donor.hydrogen_bond_partners)
        self.assertDictEqual({self.molecule_name: [[], [], []], 'test_system': [[], [], []]},
                             self.donor_atom.hydrogen_bond_partners)
        self.assertDictEqual({self.molecule_name: [[], [], []], 'test_system': [[], [], []]},
                             self.acceptor.hydrogen_bond_partners)

    def test_init_angle_cutoff_none(self):
        from yeti.systems.molecules.molecules import FourAtomsPlusMolecule

        self.hydrogen_bond_information['angle_cutoff'] = None
        self.molecule = FourAtomsPlusMolecule(residues=self.residues, molecule_name=self.molecule_name,
                                              periodic=True, box_information=self.box_information,
                                              simulation_information=self.simulation_information,
                                              hydrogen_bond_information=self.hydrogen_bond_information)

        self.check_atoms()

    def test_init_distance_cutoff_none(self):
        from yeti.systems.molecules.molecules import FourAtomsPlusMolecule

        self.hydrogen_bond_information['distance_cutoff'] = None
        self.molecule = FourAtomsPlusMolecule(residues=self.residues, molecule_name=self.molecule_name,
                                              periodic=True, box_information=self.box_information,
                                              simulation_information=self.simulation_information,
                                              hydrogen_bond_information=self.hydrogen_bond_information)

        self.check_atoms()


class TestDihedralAngleMethods(FourAtomsPlusMoleculeTestCase):
    def test_store(self):
        self.molecule.get_dihedral(atom_01_pos=(0, 0), atom_02_pos=(0, 1), atom_03_pos=(1, 0), atom_04_pos=(1, 1),
                                   store_result=True, opt=True, periodic=True)
        exp_key = 'RESA_0000:A_0000-RESA_0000:B_0001-RESB_0001:C_0002-RESB_0001:D_0003'
        exp_dict = {exp_key: np.array([0.27255287453, 1.23095941734077, -2.95544781412733])}

        self.assertEqual(self.molecule.dihedral_angles.keys(), exp_dict.keys())
        npt.assert_array_almost_equal(self.molecule.dihedral_angles[exp_key], exp_dict[exp_key], decimal=5)

    def test_not_store(self):
        res = self.molecule.get_dihedral(atom_01_pos=(0, 0), atom_02_pos=(0, 1), atom_03_pos=(1, 0), atom_04_pos=(1, 1),
                                         store_result=False, opt=True, periodic=True)

        npt.assert_array_almost_equal(np.array([0.27255287453, 1.23095941734077, -2.95544781412733]), res, decimal=5)


class TestHydrogenBondMethodsInitialized(FourAtomsPlusMoleculeTestCase):
    def test_purge_hydrogen_bonds(self):
        self.molecule.purge_hydrogen_bonds()

        self.assertIsNone(self.molecule.residues[0].atoms[0].hydrogen_bond_partners)
        self.assertDictEqual({'test': [[], [], []], 'test_system': [[], [], []]},
                             self.molecule.residues[0].atoms[1].hydrogen_bond_partners)
        self.assertDictEqual({'test': [[], [], []], 'test_system': [[], [], []]},
                             self.molecule.residues[1].atoms[0].hydrogen_bond_partners)
        self.assertIsNone(self.molecule.residues[1].atoms[1].hydrogen_bond_partners)

    def test_get_hydrogen_bonds_frame_wise(self):
        res = self.molecule.get_hydrogen_bonds(representation_style='frame_wise')
        npt.assert_array_equal(np.array([1, 0, 0]), res)

    def test_get_hydrogen_bonds_matrix(self):
        res = self.molecule.get_hydrogen_bonds(representation_style='matrix')
        exp_first_frame = np.zeros((4, 4))
        exp_first_frame[1, 2] = 1
        exp_first_frame[2, 1] = 1

        npt.assert_array_equal(
            np.array([exp_first_frame, np.zeros_like(exp_first_frame), np.zeros_like(exp_first_frame)]), res)


class TestHydrogenBondMethodsNotInitialized(FourAtomsPlusMoleculeTestCase):
    def setUp(self) -> None:
        from yeti.systems.molecules.molecules import FourAtomsPlusMolecule

        super(TestHydrogenBondMethodsNotInitialized, self).setUp()

        self.hydrogen_bond_information = dict(distance_cutoff=None, angle_cutoff=None)
        self.molecule = FourAtomsPlusMolecule(residues=self.residues, molecule_name=self.molecule_name,
                                              periodic=True, box_information=self.box_information,
                                              simulation_information=self.simulation_information,
                                              hydrogen_bond_information=self.hydrogen_bond_information)

    def test_recalculate_hydrogen_bonds(self):
        self.molecule.recalculate_hydrogen_bonds(distance_cutoff=0.25, angle_cutoff=2.0)

        self.assertIsNone(self.molecule.residues[0].atoms[0].hydrogen_bond_partners)
        self.assertDictEqual({'test': [[self.molecule.residues[1].atoms[0]], [], []], 'test_system': [[], [], []]},
                             self.molecule.residues[0].atoms[1].hydrogen_bond_partners)
        self.assertDictEqual({'test': [[self.molecule.residues[0].atoms[1]], [], []], 'test_system': [[], [], []]},
                             self.molecule.residues[1].atoms[0].hydrogen_bond_partners)
        self.assertIsNone(self.molecule.residues[1].atoms[1].hydrogen_bond_partners)


class FourAtomsPlusMoleculeExceptionsTestCase(FourAtomsPlusMoleculeTestCase, TwoAtomsMoleculeExceptionsTestCase):
    def setUp(self) -> None:
        super(FourAtomsPlusMoleculeExceptionsTestCase, self).setUp()


class TestFourAtomsPlusStandardMethodExceptions(FourAtomsPlusMoleculeExceptionsTestCase,
                                                TestTwoAtomsStandardMethodExceptions):
    def setUp(self) -> None:
        from yeti.systems.molecules.molecules import FourAtomsPlusMolecule

        super(TestFourAtomsPlusStandardMethodExceptions, self).setUp()

        self.molecule = FourAtomsPlusMolecule

    def test_init_residues_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residues=4.2, molecule_name=self.molecule_name, box_information=self.box_information,
                          simulation_information=self.simulation_information,
                          hydrogen_bond_information=self.hydrogen_bond_information, periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='residues', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_molecule_name_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residues=self.residues, molecule_name=42, box_information=self.box_information,
                          simulation_information=self.simulation_information,
                          hydrogen_bond_information=self.hydrogen_bond_information, periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='molecule_name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_box_information_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residues=self.residues, molecule_name=self.molecule_name, box_information=[],
                          simulation_information=self.simulation_information,
                          hydrogen_bond_information=self.hydrogen_bond_information, periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='box_information', data_type_name='dict')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_simulation_information_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residues=self.residues, molecule_name=self.molecule_name,
                          box_information=self.box_information, simulation_information=(),
                          hydrogen_bond_information=self.hydrogen_bond_information, periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='simulation_information',
                                                               data_type_name='dict')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_hydrogen_bond_information_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residues=self.residues, molecule_name=self.molecule_name,
                          box_information=self.box_information, simulation_information=self.simulation_information,
                          hydrogen_bond_information=(), periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='hydrogen_bond_information',
                                                               data_type_name='dict')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_periodic_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residues=self.residues, molecule_name=self.molecule_name,
                          box_information=self.box_information, simulation_information=self.simulation_information,
                          hydrogen_bond_information=self.hydrogen_bond_information, periodic='No')

        desired_msg = self.create_data_type_exception_messages(parameter_name='periodic',
                                                               data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))


class TestDihedralAngleMethodExceptions(FourAtomsPlusMoleculeExceptionsTestCase):
    def test_atom_01_pos_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_dihedral(atom_01_pos=[0, 0], atom_02_pos=(0, 1), atom_03_pos=(1, 0), atom_04_pos=(1, 1),
                                       store_result=True, opt=True, periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_01_pos', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_atom_02_pos_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_dihedral(atom_01_pos=(0, 0), atom_02_pos=[0, 1], atom_03_pos=(1, 0), atom_04_pos=(1, 1),
                                       store_result=True, opt=True, periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_02_pos', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_atom_03_pos_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_dihedral(atom_01_pos=(0, 0), atom_02_pos=(0, 1), atom_03_pos=[1, 0], atom_04_pos=(1, 1),
                                       store_result=True, opt=True, periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_03_pos', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_atom_04_pos_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_dihedral(atom_01_pos=(0, 0), atom_02_pos=(0, 1), atom_03_pos=(1, 0), atom_04_pos=[1, 1],
                                       store_result=True, opt=True, periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_04_pos', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_store_result_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_dihedral(atom_01_pos=(0, 0), atom_02_pos=(0, 1), atom_03_pos=(1, 0), atom_04_pos=(1, 1),
                                       store_result=42, opt=True, periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='store_result', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_opt_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_dihedral(atom_01_pos=(0, 0), atom_02_pos=(0, 1), atom_03_pos=(1, 0), atom_04_pos=(1, 1),
                                       store_result=True, opt=4.2, periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='opt', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))

    def test_periodic_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_dihedral(atom_01_pos=(0, 0), atom_02_pos=(0, 1), atom_03_pos=(1, 0), atom_04_pos=(1, 1),
                                       store_result=True, opt=True, periodic=1337)

        desired_msg = self.create_data_type_exception_messages(parameter_name='periodic', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))


class TestHydrogenBondMethodExceptions(FourAtomsPlusMoleculeExceptionsTestCase):
    def test_get_hydrogen_bonds_unknown_repr_style(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_hydrogen_bonds(representation_style='unknown')

        desired_msg = 'Unknown representation style!'
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_hydrogen_bonds_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_hydrogen_bonds(representation_style=3)

        desired_msg = self.create_data_type_exception_messages(parameter_name='representation_style',
                                                               data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_hydrogen_bonds_not_calculated_yet(self):
        from yeti.systems.molecules.molecules import FourAtomsPlusMolecule

        self.hydrogen_bond_information = dict(distance_cutoff=None, angle_cutoff=None)
        self.molecule = FourAtomsPlusMolecule(residues=self.residues, molecule_name=self.molecule_name,
                                              periodic=True, box_information=self.box_information,
                                              simulation_information=self.simulation_information,
                                              hydrogen_bond_information=self.hydrogen_bond_information)

        with self.assertRaises(self.exception) as context:
            self.molecule.get_hydrogen_bonds(representation_style='matrix')

        desired_msg = 'You need to calculate the hydrogen bonds first!'
        self.assertEqual(desired_msg, str(context.exception))



if __name__ == '__main__':
    unittest.main()
