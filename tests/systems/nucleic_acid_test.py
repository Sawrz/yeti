import unittest

import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintExceptionsTestCase
from tests.systems.bio_molecule_test import BioMoleculeTestCase, TestBioMoleculeStandardMethods


class NucleicAcidTestCase(BioMoleculeTestCase):
    def setUpResidues(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.systems.building_blocks import Residue

        atom_01 = Atom(structure_file_index=3, subsystem_index=1, name='P',
                       xyz_trajectory=np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4]]))
        atom_02 = Atom(structure_file_index=4, subsystem_index=2, name='P',
                       xyz_trajectory=np.array([[0.4, 0.4, 0.4], [0.8, 0.8, 0.8]]))
        atom_03 = Atom(structure_file_index=2, subsystem_index=0, name='A',
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
        from yeti.systems.molecules.nucleic_acids import NucleicAcid

        super(NucleicAcidTestCase, self).setUp()
        self.setUpResidues()

        self.box_information['unit_cell_angles'] = np.array([[90, 90, 90], [90, 90, 90]], dtype=np.float32)
        self.box_information['unit_cell_vectors'] = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float32)

        self.molecule = NucleicAcid(residues=self.residues, molecule_name=self.molecule_name,
                                    box_information=self.box_information,
                                    simulation_information=self.simulation_information, periodic=True,
                                    hydrogen_bond_information=self.hydrogen_bond_information)


class TestNucleicAcidStandardMethods(TestBioMoleculeStandardMethods):
    def setUp(self) -> None:
        from yeti.systems.molecules.nucleic_acids import NucleicAcid
        from yeti.dictionaries.molecules import biomolecules

        super(TestNucleicAcidStandardMethods, self).setUp()

        self.dictionary = biomolecules.NucleicAcid
        self.nucleic_acid = NucleicAcid(residues=self.residues, molecule_name=self.molecule_name,
                                        box_information=self.box_information,
                                        simulation_information=self.simulation_information, periodic=True,
                                        hydrogen_bond_information=self.hydrogen_bond_information)

    def test_init(self):
        from yeti.systems.molecules.nucleic_acids import NucleicAcidException


        self.assertEqual(NucleicAcidException, self.nucleic_acid.ensure_data_type.exception_class)
        self.assertEqual(self.dictionary, type(self.nucleic_acid.dictionary))


class TestNucleicAcidDistanceMethods(NucleicAcidTestCase):
    def setUpResidues(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.systems.building_blocks import Residue

        atom_01 = Atom(structure_file_index=3, subsystem_index=1, name='P',
                       xyz_trajectory=np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4]]))
        atom_02 = Atom(structure_file_index=4, subsystem_index=2, name='P',
                       xyz_trajectory=np.array([[0.4, 0.4, 0.4], [0.8, 0.8, 0.8]]))
        atom_03 = Atom(structure_file_index=2, subsystem_index=0, name='A',
                       xyz_trajectory=np.array([[0.25, 0.25, 0.25], [0.85, 0.85, 0.85]]))
        atom_04 = Atom(structure_file_index=5, subsystem_index=3, name='P',
                       xyz_trajectory=np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]))
        atom_05 = Atom(structure_file_index=2, subsystem_index=0, name='B',
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

    def test_get_p_to_p_distance(self):
        self.molecule.get_p_to_p_distance(residue_id_01=0, residue_id_02=2, periodic=True)
        exp_key = 'RESA_0000:P_0003-RESC_0002:P_0002'
        exp_distances = np.array([0.519615242, 0.692820323])

        self.assertListEqual([exp_key], list(self.molecule.distances.keys()))
        npt.assert_array_almost_equal(exp_distances, self.molecule.distances[exp_key], decimal=5)

    def test_get_all_p_to_p_distances(self):
        self.molecule.get_all_p_to_p_distances(periodic=True)

        exp = {'RESA_0000:P_0003-RESB_0001:P_0001': np.array([0.173205081, 0.346410162]),
               'RESB_0001:P_0001-RESC_0002:P_0002': np.array([0.346410162, 0.692820323]),
               'RESA_0000:P_0003-RESC_0002:P_0002': np.array([0.519615242, 0.692820323])}

        self.assertEqual(exp.keys(), self.molecule.distances.keys())

        for key in exp.keys():
            npt.assert_array_almost_equal(exp[key], self.molecule.distances[key], decimal=5)


class TestNucleicAcidDihedralAngleMethods(NucleicAcidTestCase):
    def setUpResidues(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.systems.building_blocks import Residue

        # alpha
        atom_01 = Atom(structure_file_index=2, subsystem_index=0, name="O3\'",
                       xyz_trajectory=np.array([[0.25, 0, 0], [0.4, 0, 0]]))
        atom_02 = Atom(structure_file_index=3, subsystem_index=1, name="P",
                       xyz_trajectory=np.array([[0.3, 0, 0], [0.5, 0, 0]]))
        atom_03 = Atom(structure_file_index=4, subsystem_index=2, name="O5\'",
                       xyz_trajectory=np.array([[0, 0.1, 0], [0, 0.2, 0]]))
        atom_04 = Atom(structure_file_index=5, subsystem_index=3, name="C5\'",
                       xyz_trajectory=np.array([[0, 0.2, 0], [0, 0.3, 0]]))

        # chi_pu
        atom_05 = Atom(structure_file_index=6, subsystem_index=4, name="C1\'",
                       xyz_trajectory=np.array([[0.9, 0.1, 0.9], [0.5, 0, 0]]))
        atom_06 = Atom(structure_file_index=7, subsystem_index=5, name="O4\'",
                       xyz_trajectory=np.array([[0.8, 0.9, 0.7], [0.4, 0, 0]]))
        atom_07 = Atom(structure_file_index=8, subsystem_index=6, name="N9",
                       xyz_trajectory=np.array([[0.1, 0.2, 0.25], [0, 0.2, 0]]))
        atom_08 = Atom(structure_file_index=9, subsystem_index=7, name="C4",
                       xyz_trajectory=np.array([[0.25, 0.33, 0.4], [0, 0.3, 0]]))

        # chi_py
        atom_09 = Atom(structure_file_index=10, subsystem_index=8, name="O4\'",
                       xyz_trajectory=np.array([[0.4, 0, 0], [0.8, 0.9, 0.7]]))
        atom_10 = Atom(structure_file_index=11, subsystem_index=9, name="C1\'",
                       xyz_trajectory=np.array([[0.5, 0, 0], [0.9, 0.1, 0.9]]))
        atom_11 = Atom(structure_file_index=12, subsystem_index=10, name="N1",
                       xyz_trajectory=np.array([[0, 0.2, 0], [0.1, 0.2, 0.25]]))
        atom_12 = Atom(structure_file_index=13, subsystem_index=11, name="C2",
                       xyz_trajectory=np.array([[0, 0.3, 0], [0.25, 0.33, 0.4]]))

        # atom from chi_pu but other residue
        atom_13 = Atom(structure_file_index=14, subsystem_index=12, name="C2",
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

        residue_02.add_atom(atom=atom_05)
        atom_05.set_residue(residue=residue_02)

        residue_02.add_atom(atom=atom_06)
        atom_06.set_residue(residue=residue_02)

        residue_02.add_atom(atom=atom_07)
        atom_07.set_residue(residue=residue_02)

        residue_02.add_atom(atom=atom_08)
        atom_08.set_residue(residue=residue_02)

        residue_03.add_atom(atom=atom_09)
        atom_09.set_residue(residue=residue_03)

        residue_03.add_atom(atom=atom_10)
        atom_10.set_residue(residue=residue_03)

        residue_03.add_atom(atom=atom_11)
        atom_11.set_residue(residue=residue_03)

        residue_03.add_atom(atom=atom_12)
        atom_12.set_residue(residue=residue_03)

        residue_04.add_atom(atom=atom_13)
        atom_13.set_residue(residue=residue_04)

        self.residues = (residue_01, residue_02, residue_03, residue_04)

    def test_get_dihedral_alpha(self):
        self.molecule.get_dihedral(dihedral_name='alpha', residue_id=1, periodic=True)
        exp = {'alpha_001': np.array([np.pi, np.pi])}

        self.assertEqual(exp.keys(), self.molecule.dihedral_angles.keys())

        for key in exp.keys():
            npt.assert_array_almost_equal(exp[key], self.molecule.dihedral_angles[key], decimal=5)

    def test_get_dihedral_chi_pu(self):
        self.molecule.get_dihedral(dihedral_name='chi', residue_id=1, periodic=True)
        exp = {'chi_001': np.array([2.42560681293738, np.pi])}

        self.assertEqual(exp.keys(), self.molecule.dihedral_angles.keys())

        for key in exp.keys():
            npt.assert_array_almost_equal(exp[key], self.molecule.dihedral_angles[key], decimal=5)

    def test_get_dihedral_chi_py(self):
        self.molecule.get_dihedral(dihedral_name='chi', residue_id=2, periodic=True)
        exp = {'chi_002': np.array([np.pi, 2.42560681293738])}

        self.assertEqual(exp.keys(), self.molecule.dihedral_angles.keys())

        for key in exp.keys():
            npt.assert_array_almost_equal(exp[key], self.molecule.dihedral_angles[key], decimal=5)

    def test_get_all_dihedral_angles(self):
        self.molecule.get_all_dihedral_angles(periodic=True)

        exp = {'alpha_001': np.array([np.pi, np.pi]),
               'chi_001': np.array([2.42560681293738, np.pi]),
               'chi_002': np.array([np.pi, 2.42560681293738])}

        self.assertEqual(exp.keys(), self.molecule.dihedral_angles.keys())

        for key in exp.keys():
            npt.assert_array_almost_equal(exp[key], self.molecule.dihedral_angles[key], decimal=5)


class NucleicAcidExceptionsTestCase(NucleicAcidTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.systems.molecules.nucleic_acids import NucleicAcidException

        super(NucleicAcidExceptionsTestCase, self).setUp()

        self.exception = NucleicAcidException


class TestNucleicAcidDistanceMethodExceptions(NucleicAcidExceptionsTestCase):
    def test_get_p_to_p_distance_residue_id_01_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_p_to_p_distance(residue_id_01=0., residue_id_02=2, periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='residue_id_01', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_p_to_p_distance_residue_id_02_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_p_to_p_distance(residue_id_01=0, residue_id_02=2., periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='residue_id_02', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_p_to_p_distance_periodic_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_p_to_p_distance(residue_id_01=0, residue_id_02=2, periodic=42)

        desired_msg = self.create_data_type_exception_messages(parameter_name='periodic', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))


class TestNucleicAcidDihedralAngleMethodExceptions(NucleicAcidExceptionsTestCase):
    def test_get_dihedral_dihedral_name_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_dihedral(dihedral_name=4, residue_id=1, periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='dihedral_name', data_type_name='str')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_dihedral_residue_id_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_dihedral(dihedral_name='alpha', residue_id='1', periodic=True)

        desired_msg = self.create_data_type_exception_messages(parameter_name='residue_id', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))


    def test_get_dihedral_periodic_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.molecule.get_dihedral(dihedral_name='alpha', residue_id=1, periodic='Yes')

        desired_msg = self.create_data_type_exception_messages(parameter_name='periodic', data_type_name='bool')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
