import unittest

import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintExceptionsTestCase
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


class TestXyzMethods(TwoAtomsMoleculeTestCase):
    def setUp(self) -> None:
        from yeti.systems.molecules.molecules import TwoAtomsMolecule
        from yeti.systems.building_blocks import Atom

        super(TestXyzMethods, self).setUp()

        # Extend Residue
        self.residue_01.definalize()

        atom = Atom(structure_file_index=42, subsystem_index=18, name='P',
                    xyz_trajectory=np.array([[0.6, 0.3, 0.8], [0.2, 0.9, 0.6]]))

        self.residue_01.add_atom(atom=atom)
        atom.set_residue(residue=self.residue_01)

        self.residue_01.finalize()

        # Add additional frame
        self.donor.delete_frame(2)
        self.donor_atom.delete_frame(2)
        self.acceptor.delete_frame(2)

        # initialize object
        self.atoms = (self.donor, self.donor_atom, self.acceptor, atom)
        self.molecule = TwoAtomsMolecule(residues=self.residues, molecule_name=self.molecule_name,
                                         box_information=self.box_information, periodic=True)

    def test_get(self):
        res_xyz, res_names = self.molecule.get_xyz()

        exp_xyz = np.array([[[0.1, 0.4, 0.3], [0.1, 0.5, 0.2], [0.6, 0.3, 0.8], [0.1, 0.6, 0.4]],
                            [[0.1, 0.4, 0.3], [0.1, 0.5, 0.2], [0.2, 0.9, 0.6], [0.1, 0.7, 0.4]]])

        self.assertListEqual(['RESA4_A', 'RESA4_B', 'RESA4_P', 'RESB5_C'], res_names)
        npt.assert_array_equal(res_xyz, exp_xyz)


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
