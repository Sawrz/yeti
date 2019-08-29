import unittest

import numpy as np
import numpy.testing as npt


class BlueprintTestCase(unittest.TestCase):
    @staticmethod
    def create_data_type_exception_messages(parameter_name, data_type_name):
        return 'Wrong data type for parameter "{name}". Desired type is {data_type}'.format(name=parameter_name,
                                                                                            data_type=data_type_name)

    @staticmethod
    def create_array_shape_exception_messages(parameter_name, desired_shape):
        return 'Wrong shape for parameter "{name}". Desired shape: {des_shape}.'.format(name=parameter_name,
                                                                                        des_shape=desired_shape)

    @staticmethod
    def create_array_dtype_exception_messages(parameter_name, dtype_name):
        return 'Wrong dtype for ndarray "{name}". Desired dtype is {data_type}'.format(name=parameter_name,
                                                                                       data_type=dtype_name)

    @staticmethod
    def build_unit_cell_angles_and_vectors(number_of_frames):
        angles = []
        vectors = []

        for i in range(number_of_frames):
            angles.append([90, 90, 90])
            vectors.append([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        angles = np.array(angles, dtype=np.float32)
        vectors = np.array(vectors, dtype=np.float32)

        return angles, vectors


class PostAtomTest(BlueprintTestCase):
    def create_atoms(self):
        from yeti.systems.building_blocks import Atom

        # first frame is h-bond
        # second frame is not because of distance
        # third frame is not because of angle

        donor = Atom(structure_file_index=2, subsystem_index=0, name='A',
                     xyz_trajectory=np.array([[0.1, 0.4, 0.3], [0.1, 0.4, 0.3], [0.1, 0.4, 0.3]]))
        donor_atom = Atom(structure_file_index=3, subsystem_index=1, name='B',
                          xyz_trajectory=np.array([[0.1, 0.5, 0.2], [0.1, 0.5, 0.2], [0.5, 0.5, 0.2]]))
        acceptor = Atom(structure_file_index=4, subsystem_index=2, name='C',
                        xyz_trajectory=np.array([[0.1, 0.6, 0.4], [0.1, 0.7, 0.4], [0.1, 0.6, 0.4]]))

        donor.add_covalent_bond(atom=donor_atom)

        donor_atom.update_donor_state(is_donor_atom=True, donor_slots=1)
        donor_atom.add_system(system_name='test_system')

        acceptor.update_acceptor_state(is_acceptor=True, acceptor_slots=2)
        acceptor.add_system(system_name='test_system')

        return donor, donor_atom, acceptor

    def create_residues(self):
        from yeti.systems.building_blocks import Residue

        atom_01, atom_02, atom_03 = self.create_atoms()

        residue_01 = Residue(subsystem_index=0, structure_file_index=4, name='RESA')
        residue_02 = Residue(subsystem_index=1, structure_file_index=5, name='RESB')

        residue_01.add_atom(atom=atom_01)
        atom_01.set_residue(residue=residue_01)

        residue_01.add_atom(atom=atom_02)
        atom_02.set_residue(residue=residue_01)

        residue_02.add_atom(atom=atom_03)
        atom_03.set_residue(residue=residue_02)

        residue_01.finalize()
        residue_02.finalize()

        return residue_01, residue_02


class MoleculesTest(PostAtomTest):

    def setUp(self) -> tuple:
        from yeti.systems.molecules.molecules import MoleculeException

        self.exception = MoleculeException

        unit_cell_angles, unit_cell_vectors = self.build_unit_cell_angles_and_vectors(number_of_frames=3)
        residues = self.create_residues()

        box_information = dict(
            dict(periodic=True, unit_cell_angles=unit_cell_angles, unit_cell_vectors=unit_cell_vectors))

        return residues, box_information


class TwoAtomsInitTest(MoleculesTest):
    def setUp(self) -> None:
        from yeti.systems.molecules.molecules import TwoAtomsMolecule

        self.residues, self.box_information = super(TwoAtomsInitTest, self).setUp()
        self.molecule_name = 'test'
        self.molecule = TwoAtomsMolecule

    def setUpWorkingMolecule(self):
        self.working_molecule = self.molecule(residues=self.residues, molecule_name=self.molecule_name,
                                              box_information=self.box_information)

    def test_init(self):
        from yeti.get_features.distances import Distance

        self.setUpWorkingMolecule()

        self.assertEqual(self.exception, self.working_molecule.ensure_data_type.exception_class)
        self.assertEqual(self.molecule_name, self.working_molecule.molecule_name)
        self.assertTupleEqual(self.residues, self.working_molecule.residues)
        self.assertEqual(Distance, type(self.working_molecule._dist))
        self.assertDictEqual({}, self.working_molecule.distances)

    def test_init_residues_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residues=4.2, molecule_name=self.molecule_name, box_information=self.box_information)

        desired_msg = self.create_data_type_exception_messages(parameter_name='residues', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_molecule_name_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residues=self.residues, molecule_name=42, box_information=self.box_information)

        desired_msg = self.create_data_type_exception_messages(parameter_name='molecule_name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_init_box_information_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule(residues=self.residues, molecule_name=self.molecule_name, box_information=[])

        desired_msg = self.create_data_type_exception_messages(parameter_name='box_information', data_type_name='dict')
        self.assertEqual(desired_msg, str(context.exception))


class TwoAtomsMoleculeTest(MoleculesTest):
    def setUp(self) -> None:
        from yeti.systems.molecules.molecules import TwoAtomsMolecule
        residues, box_information = super(TwoAtomsMoleculeTest, self).setUp()

        self.molecule = TwoAtomsMolecule(residues=residues, molecule_name='test',
                                         box_information=box_information)


class AtomsTest(TwoAtomsMoleculeTest):
    def setUp(self) -> None:
        super(AtomsTest, self).setUp()

        self.atoms = list(self.molecule.residues[0].atoms)
        self.atoms.extend(self.molecule.residues[1].atoms)
        self.atoms = tuple(self.atoms)


class GenerateKeysTest(AtomsTest):
    def test_get_atom_key_name(self):
        res = self.molecule.__get_atom_key_name__(atom=self.atoms[0])

        self.assertEqual('RESA_0000:A_0000', res)

    def test_get_atom_key_name_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.__get_atom_key_name__(atom=42)

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom', data_type_name='Atom')
        self.assertEqual(desired_msg, str(context.exception))

    def test_generate_key_two_atoms(self):
        res = self.molecule.__generate_key__(atoms=self.atoms[:2])

        self.assertEqual('RESA_0000:A_0000-RESA_0000:B_0001', res)

    def test_generate_key_three_atoms(self):
        res = self.molecule.__generate_key__(atoms=self.atoms)

        self.assertEqual('RESA_0000:A_0000-RESA_0000:B_0001-RESB_0001:C_0002', res)

    def test_generate_key_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.__generate_key__(atoms=[])

        desired_msg = self.create_data_type_exception_messages(parameter_name='atoms', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))


class GetAtomsTest(AtomsTest):
    def test_get_atom(self):
        res = self.molecule.__get_atom__(atom_pos=(1, 0))
        self.assertEqual(self.atoms[2], res)

    def test_get_atom_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.__get_atom__(atom_pos=[1, 0])

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_pos', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_atoms(self):
        res = self.molecule.__get_atoms__(atom_positions=((0, 0), (1, 0)))
        self.assertEqual((self.atoms[0], self.atoms[2]), res)

    def test_get_atoms_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.__get_atoms__([(0, 0), (1, 0)])

        desired_msg = self.create_data_type_exception_messages(parameter_name='atom_positions', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))


class GetDistance(TwoAtomsMoleculeTest):
    def test_store(self):
        self.molecule.get_distance(atom_01_pos=(0, 0), atom_02_pos=(1, 0), store_result=True, opt=True)
        exp_key = 'RESA_0000:A_0000-RESB_0001:C_0002'
        exp_dict = {exp_key: np.array([0.22360681, 0.31622773, 0.22360681])}

        self.assertEqual(self.molecule.distances.keys(), exp_dict.keys())
        npt.assert_array_almost_equal(self.molecule.distances[exp_key], exp_dict[exp_key], decimal=5)

    def test_not_store(self):
        res = self.molecule.get_distance(atom_01_pos=(0, 0), atom_02_pos=(1, 0), store_result=False, opt=True)

        npt.assert_array_almost_equal(np.array([0.22360681, 0.31622773, 0.22360681]), res, decimal=5)

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
