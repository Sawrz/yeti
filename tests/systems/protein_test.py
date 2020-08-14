import unittest

import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintExceptionsTestCase
from tests.systems.bio_molecule_test import BioMoleculeTestCase, TestBioMoleculeStandardMethods


class ProteinTestCase(BioMoleculeTestCase):
    def setUpResidues(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.systems.building_blocks import Residue

        atom_01 = Atom(structure_file_index=3,
                       subsystem_index=1,
                       name='CA',
                       xyz_trajectory=np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4]]))
        atom_02 = Atom(structure_file_index=4,
                       subsystem_index=2,
                       name='N',
                       xyz_trajectory=np.array([[0.4, 0.4, 0.4], [0.8, 0.8, 0.8]]))
        atom_03 = Atom(structure_file_index=2,
                       subsystem_index=0,
                       name='CA',
                       xyz_trajectory=np.array([[0.25, 0.25, 0.25], [0.85, 0.85, 0.85]]))

        residue_01 = Residue(subsystem_index=0, structure_file_index=4, name='RESA')
        residue_02 = Residue(subsystem_index=1, structure_file_index=5, name='RESB')

        residue_01.add_atom(atom=atom_03)
        atom_03.set_residue(residue=residue_01)

        residue_02.add_atom(atom=atom_01)
        atom_01.set_residue(residue=residue_02)

        residue_01.add_atom(atom=atom_02)
        atom_02.set_residue(residue=residue_01)

        self.residues = (residue_01, residue_02)

    def setUp(self) -> None:
        from yeti.systems.molecules.proteins import Protein

        super(ProteinTestCase, self).setUp()
        self.setUpResidues()

        self.box_information['unit_cell_angles'] = np.array([[90, 90, 90], [90, 90, 90]], dtype=np.float32)
        self.box_information['unit_cell_vectors'] = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float32)

        self.molecule = Protein(residues=self.residues,
                                molecule_name=self.molecule_name,
                                box_information=self.box_information,
                                periodic=True)


class TestProteinStandardMethods(TestBioMoleculeStandardMethods):
    def setUp(self) -> None:
        from yeti.systems.molecules.proteins import Protein
        from yeti.dictionaries.molecules import biomolecules

        super(TestProteinStandardMethods, self).setUp()

        self.dictionary = biomolecules.Protein
        self.molecule = Protein(residues=self.residues,
                                molecule_name=self.molecule_name,
                                box_information=self.box_information,
                                periodic=True)

    def test_init(self):
        from yeti.systems.molecules.proteins import ProteinException

        self.assertEqual(ProteinException, self.molecule.ensure_data_type.exception_class)
        self.assertEqual(self.dictionary, type(self.molecule.dictionary))


class TestProteinDistanceMethods(ProteinTestCase):
    def setUpResidues(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.systems.building_blocks import Residue

        atom_01 = Atom(structure_file_index=3,
                       subsystem_index=1,
                       name='CA',
                       xyz_trajectory=np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4]]))
        atom_02 = Atom(structure_file_index=4,
                       subsystem_index=2,
                       name='CA',
                       xyz_trajectory=np.array([[0.4, 0.4, 0.4], [0.8, 0.8, 0.8]]))
        atom_03 = Atom(structure_file_index=2,
                       subsystem_index=0,
                       name='A',
                       xyz_trajectory=np.array([[0.25, 0.25, 0.25], [0.85, 0.85, 0.85]]))
        atom_04 = Atom(structure_file_index=5,
                       subsystem_index=3,
                       name='CA',
                       xyz_trajectory=np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]))
        atom_05 = Atom(structure_file_index=2,
                       subsystem_index=0,
                       name='B',
                       xyz_trajectory=np.array([[0.85, 0.85, 0.85], [0.75, 0.75, 0.75]]))

        residue_01 = Residue(subsystem_index=0, structure_file_index=4, name='RESA')
        residue_02 = Residue(subsystem_index=1, structure_file_index=5, name='RESB')
        residue_03 = Residue(subsystem_index=2, structure_file_index=6, name='RESC')
        residue_04 = Residue(subsystem_index=3, structure_file_index=7, name='RESD')

        residue_01.add_atom(atom=atom_03)
        atom_03.set_residue(residue=residue_01)

        residue_01.add_atom(atom=atom_04)
        atom_04.set_residue(residue=residue_01)

        residue_02.add_atom(atom=atom_01)
        atom_01.set_residue(residue=residue_02)

        residue_03.add_atom(atom=atom_02)
        atom_02.set_residue(residue=residue_03)

        residue_04.add_atom(atom=atom_05)
        atom_05.set_residue(residue=residue_04)

        self.residues = (residue_01, residue_02, residue_03, residue_04)

    def test_get_distance(self):
        self.molecule.get_distance(distance_name='CA_CA', residue_id_01=0, residue_id_02=2)
        exp_key = 'RESA_0000:CA_0003-RESC_0002:CA_0002'
        exp_distances = np.array([0.519615242, 0.692820323])

        self.assertListEqual([exp_key], list(self.molecule.distances.keys()))
        npt.assert_array_almost_equal(exp_distances, self.molecule.distances[exp_key], decimal=5)


class TestProteinDihedralAngleMethods(ProteinTestCase):
    def setUpResidues(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.systems.building_blocks import Residue

        # phi
        atom_01 = Atom(structure_file_index=2,
                       subsystem_index=0,
                       name="C",
                       xyz_trajectory=np.array([[0.25, 0, 0], [0.4, 0, 0]]))
        atom_02 = Atom(structure_file_index=3,
                       subsystem_index=1,
                       name="N",
                       xyz_trajectory=np.array([[0.3, 0, 0], [0.5, 0, 0]]))
        atom_03 = Atom(structure_file_index=4,
                       subsystem_index=2,
                       name="CA",
                       xyz_trajectory=np.array([[0, 0.1, 0], [0, 0.2, 0]]))
        atom_04 = Atom(structure_file_index=5,
                       subsystem_index=3,
                       name="C",
                       xyz_trajectory=np.array([[0, 0.2, 0], [0, 0.3, 0]]))

        # psi
        atom_05 = Atom(structure_file_index=6,
                       subsystem_index=4,
                       name="CA",
                       xyz_trajectory=np.array([[0.9, 0.1, 0.9], [0.5, 0, 0]]))
        atom_06 = Atom(structure_file_index=7,
                       subsystem_index=5,
                       name="N",
                       xyz_trajectory=np.array([[0.8, 0.9, 0.7], [0.4, 0, 0]]))
        atom_07 = Atom(structure_file_index=8,
                       subsystem_index=6,
                       name="C",
                       xyz_trajectory=np.array([[0.1, 0.2, 0.25], [0, 0.2, 0]]))
        atom_08 = Atom(structure_file_index=9,
                       subsystem_index=7,
                       name="N",
                       xyz_trajectory=np.array([[0.25, 0.33, 0.4], [0, 0.3, 0]]))

        # chi1
        atom_09 = Atom(structure_file_index=10,
                       subsystem_index=8,
                       name="N",
                       xyz_trajectory=np.array([[0.4, 0, 0], [0.8, 0.9, 0.7]]))
        atom_10 = Atom(structure_file_index=11,
                       subsystem_index=9,
                       name="CA",
                       xyz_trajectory=np.array([[0.5, 0, 0], [0.9, 0.1, 0.9]]))
        atom_11 = Atom(structure_file_index=12,
                       subsystem_index=10,
                       name="CB",
                       xyz_trajectory=np.array([[0, 0.2, 0], [0.1, 0.2, 0.25]]))
        atom_12 = Atom(structure_file_index=13,
                       subsystem_index=11,
                       name="CG",
                       xyz_trajectory=np.array([[0, 0.3, 0], [0.25, 0.33, 0.4]]))

        # atom from chi_pu but other residue
        atom_13 = Atom(structure_file_index=14,
                       subsystem_index=12,
                       name="CG",
                       xyz_trajectory=np.array([[0.85, 0.85, 0.85], [0.75, 0.75, 0.75]]))

        residue_01 = Residue(subsystem_index=0, structure_file_index=4, name='C')
        residue_02 = Residue(subsystem_index=1, structure_file_index=5, name='A')
        residue_03 = Residue(subsystem_index=2, structure_file_index=6, name='C')
        residue_04 = Residue(subsystem_index=3, structure_file_index=7, name='C')

        residue_01.add_atom(atom=atom_01)
        atom_01.set_residue(residue=residue_01)

        residue_02.add_atom(atom=atom_02)
        atom_02.set_residue(residue=residue_02)

        residue_02.add_atom(atom=atom_03)
        atom_03.set_residue(residue=residue_02)

        residue_02.add_atom(atom=atom_04)
        atom_04.set_residue(residue=residue_02)

        residue_03.add_atom(atom=atom_05)
        atom_05.set_residue(residue=residue_03)

        residue_03.add_atom(atom=atom_06)
        atom_06.set_residue(residue=residue_03)

        residue_03.add_atom(atom=atom_07)
        atom_07.set_residue(residue=residue_03)

        residue_04.add_atom(atom=atom_08)
        atom_08.set_residue(residue=residue_04)

        residue_01.add_atom(atom=atom_09)
        atom_09.set_residue(residue=residue_01)

        residue_01.add_atom(atom=atom_10)
        atom_10.set_residue(residue=residue_01)

        residue_01.add_atom(atom=atom_11)
        atom_11.set_residue(residue=residue_01)

        residue_01.add_atom(atom=atom_12)
        atom_12.set_residue(residue=residue_01)

        residue_04.add_atom(atom=atom_13)
        atom_13.set_residue(residue=residue_04)
        self.residues = (residue_01, residue_02, residue_03, residue_04)

    def test_get_dihedral_phi(self):
        self.molecule.get_dihedral(dihedral_name='phi', residue_id=1)
        exp = {'phi_001': np.array([np.pi, np.pi])}

        self.assertEqual(exp.keys(), self.molecule.dihedral_angles.keys())

        for key in exp.keys():
            npt.assert_array_almost_equal(exp[key], self.molecule.dihedral_angles[key], decimal=5)

    def test_get_dihedral_psi(self):
        self.molecule.get_dihedral(dihedral_name='psi', residue_id=2)
        exp = {'psi_002': np.array([2.42560681293738, np.pi])}

        self.assertEqual(exp.keys(), self.molecule.dihedral_angles.keys())

        for key in exp.keys():
            npt.assert_array_almost_equal(exp[key], self.molecule.dihedral_angles[key], decimal=5)

    def test_get_dihedral_chi1(self):
        self.molecule.get_dihedral(dihedral_name='chi1', residue_id=0)
        exp = {'chi1_000': np.array([np.pi, 2.42560681293738])}

        self.assertEqual(exp.keys(), self.molecule.dihedral_angles.keys())

        for key in exp.keys():
            npt.assert_array_almost_equal(exp[key], self.molecule.dihedral_angles[key], decimal=5)

    '''
    def test_get_all_dihedral_angles(self):
        self.molecule.get_all_dihedral_angles()

        exp = {
            'alpha_001': np.array([np.pi, np.pi]),
            'chi_001': np.array([2.42560681293738, np.pi]),
            'chi_002': np.array([np.pi, 2.42560681293738])
        }

        self.assertEqual(exp.keys(), self.molecule.dihedral_angles.keys())

        for key in exp.keys():
            npt.assert_array_almost_equal(exp[key], self.molecule.dihedral_angles[key], decimal=5)
    '''

class ProteinExceptionsTestCase(ProteinTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.systems.molecules.proteins import ProteinException

        super(ProteinExceptionsTestCase, self).setUp()

        self.exception = ProteinException


class TestProteinDistanceMethodExceptions(ProteinExceptionsTestCase):
    def test_get_distance_distance_name_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_distance(distance_name=False, residue_id_01=0, residue_id_02=2)

        desired_msg = self.create_data_type_exception_messages(parameter_name='distance_name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_distance_residue_id_01_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_distance(distance_name='CA_CA', residue_id_01=0., residue_id_02=2)

        desired_msg = self.create_data_type_exception_messages(parameter_name='residue_id_01', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_distance_residue_id_02_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_distance(distance_name='CA_CA', residue_id_01=0, residue_id_02=2.)

        desired_msg = self.create_data_type_exception_messages(parameter_name='residue_id_02', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))


class TestProteinDihedralAngleMethodExceptions(ProteinExceptionsTestCase):
    def test_get_dihedral_dihedral_name_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_dihedral(dihedral_name=4, residue_id=1)

        desired_msg = self.create_data_type_exception_messages(parameter_name='dihedral_name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_dihedral_residue_id_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_dihedral(dihedral_name='alpha', residue_id='1')

        desired_msg = self.create_data_type_exception_messages(parameter_name='residue_id', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
