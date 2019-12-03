import os
import unittest

import mdtraj as md
import numpy as np
import numpy.testing as npt

from tests.blueprints_test import BlueprintTestCase


class SystemTestCase(BlueprintTestCase):
    def setUp(self) -> None:
        from yeti.systems.system import System
        from yeti.dictionaries.molecules.biomolecules import RNA

        test_data_path = os.path.dirname(os.path.abspath(__file__))
        test_data_path = os.path.split(test_data_path)[0]
        test_data_path = os.path.join(test_data_path, 'test_data')

        self.xtc_file_path = os.path.join(test_data_path, 'AGCUG.xtc')
        self.gro_file_path = os.path.join(test_data_path, 'AGCUG.gro')

        self.reference_trajectory = md.load(filename_or_filenames=self.xtc_file_path, top=self.gro_file_path)
        self.system = System(name='My System')
        self.molecule_dict = RNA()


class TestStandardMethods(SystemTestCase):
    def test_init(self):
        self.assertEqual(self.system.name, 'My System')

        self.assertIsNone(self.system.trajectory_file_path)
        self.assertIsNone(self.system.topology_file_path)
        self.assertIsNone(self.system.chunk_size)

        self.assertIsNone(self.system.periodic)
        self.assertIsNone(self.system.unitcell_angles)
        self.assertIsNone(self.system.unitcell_vectors)
        self.assertIsNone(self.system.mdtraj_object)
        self.assertIsNone(self.system.trajectory)

        self.assertEqual(self.system.number_of_frames, 0)
        self.assertDictEqual(self.system.molecules, {})


class TestLoadMethods(SystemTestCase):
    def setUp(self) -> None:
        from types import GeneratorType
        super(TestLoadMethods, self).setUp()
        self.exp_generator = GeneratorType
        self.exp_number_of_frames = 6
        self.exp_unit_cell_angles = np.array([[90, 90, 90], [90, 90, 90], [90, 90, 90], [90, 90, 90], [90, 90, 90],
                                              [90, 90, 90]], dtype=np.float32)
        self.exp_unit_cell_vectors = np.array(
            [[[6.45998, 0, 0], [0, 6.45998, 0], [0, 0, 6.45998]], [[6.45998, 0, 0], [0, 6.45998, 0], [0, 0, 6.45998]],
             [[6.45998, 0, 0], [0, 6.45998, 0], [0, 0, 6.45998]], [[6.45998, 0, 0], [0, 6.45998, 0], [0, 0, 6.45998]],
             [[6.45998, 0, 0], [0, 6.45998, 0], [0, 0, 6.45998]], [[6.45998, 0, 0], [0, 6.45998, 0], [0, 0, 6.45998]]],
            dtype=np.float32)

        self.system.trajectory_file_path = self.xtc_file_path
        self.system.topology_file_path = self.gro_file_path
        self.system.chunk_size = None

    def test_create_generator(self):
        self.system.chunk_size = 1
        self.system.__create_generator__()

        self.assertEqual(type(self.system.mdtraj_object), self.exp_generator)

    def test_trajectory_load(self):
        self.system.__trajectory_load__()

        self.assertEqual(self.system.trajectory, self.reference_trajectory)
        self.assertEqual(self.system.number_of_frames, self.exp_number_of_frames)
        npt.assert_almost_equal(self.system.unitcell_angles, self.exp_unit_cell_angles, decimal=5)
        npt.assert_almost_equal(self.system.unitcell_vectors, self.exp_unit_cell_vectors, decimal=5)

    def test_iter_trajectory_load(self):
        self.system.chunk_size = 2

        self.system.__iter_trajectory_load__()

        self.assertIsNone(self.system.trajectory)
        self.assertEqual(type(self.system.mdtraj_object), self.exp_generator)
        self.assertEqual(self.system.number_of_frames, self.exp_number_of_frames)
        npt.assert_almost_equal(self.system.unitcell_angles, self.exp_unit_cell_angles, decimal=5)
        npt.assert_almost_equal(self.system.unitcell_vectors, self.exp_unit_cell_vectors, decimal=5)

    def test_load_trajectory_no_chunksize(self):
        self.system.load_trajectory(trajectory_file_path=self.xtc_file_path, topology_file_path=self.gro_file_path,
                                    chunk_size=None)

        self.assertEqual(self.system.trajectory_file_path, self.xtc_file_path)
        self.assertEqual(self.system.topology_file_path, self.gro_file_path)
        self.assertIsNone(self.system.chunk_size)

        self.assertEqual(self.system.trajectory, self.reference_trajectory)
        self.assertEqual(self.system.number_of_frames, self.exp_number_of_frames)
        npt.assert_almost_equal(self.system.unitcell_angles, self.exp_unit_cell_angles, decimal=5)
        npt.assert_almost_equal(self.system.unitcell_vectors, self.exp_unit_cell_vectors, decimal=5)

        self.assertTrue(self.system.periodic)

    def test_load_trajectory_with_chunksize(self):
        self.system.load_trajectory(trajectory_file_path=self.xtc_file_path, topology_file_path=self.gro_file_path,
                                    chunk_size=1)

        self.assertEqual(self.system.trajectory_file_path, self.xtc_file_path)
        self.assertEqual(self.system.topology_file_path, self.gro_file_path)
        self.assertEqual(self.system.chunk_size, 1)

        self.assertIsNone(self.system.trajectory)
        self.assertEqual(type(self.system.mdtraj_object), self.exp_generator)
        self.assertEqual(self.system.number_of_frames, self.exp_number_of_frames)
        npt.assert_almost_equal(self.system.unitcell_angles, self.exp_unit_cell_angles, decimal=5)
        npt.assert_almost_equal(self.system.unitcell_vectors, self.exp_unit_cell_vectors, decimal=5)

        self.assertTrue(self.system.periodic)


class SimpleTrajectoryLoadTestCase(SystemTestCase):
    def setUp(self) -> None:
        super(SimpleTrajectoryLoadTestCase, self).setUp()

        self.system.load_trajectory(trajectory_file_path=self.xtc_file_path, topology_file_path=self.gro_file_path,
                                    chunk_size=None)


class TestCreateBuildingBlockMethodsSimpleLoad(SimpleTrajectoryLoadTestCase):
    def test_create_atom(self):
        atom = self.system.__create_atom__(reference_atom=self.reference_trajectory.topology.atom(10), shifting_index=5)

        self.assertEqual(atom.name, 'N9')
        self.assertEqual(atom.subsystem_index, 5)
        self.assertEqual(atom.structure_file_index, 10)

        self.assertIsNone(atom.element)
        self.assertIsNone(atom.residue)

        # TODO: add array I expect
        npt.assert_equal(atom.xyz_trajectory, self.reference_trajectory.xyz[:, 10, :])

        self.assertTupleEqual(atom.covalent_bond_partners, ())

        self.assertFalse(atom.is_donor_atom)
        self.assertEqual(atom.donor_slots, 0)

        self.assertFalse(atom.is_acceptor)
        self.assertEqual(atom.acceptor_slots, 0)

        self.assertIsNone(atom.hydrogen_bond_partners)

    def test_create_residue(self):
        reference_residue = self.reference_trajectory.topology.residue(0)
        residue = self.system.__create_residue__(reference_residue=reference_residue, subsystem_index=1,
                                                 atom_shifting_index=2)

        for index, (exp_atom, res_atom) in enumerate(zip(reference_residue.atoms, residue.atoms)):
            self.assertEqual(exp_atom.name, res_atom.name)
            self.assertEqual(exp_atom.index, res_atom.structure_file_index)
            self.assertEqual(index - 2, res_atom.subsystem_index)

        self.assertTupleEqual(residue.sequence, ("O5'", "H5T", "C5'", "H5'1", "H5'2", "C4'", "H4'", "O4'", "C1'", "H1'",
                                                 "N9", "C8", "H8", "N7", "C5", "C6", "N6", "H61", "H62", "N1", "C2",
                                                 "H2", "N3", "C4", "C3'", "H3'", "C2'", "H2'1", "O2'", "HO'2", "O3'"))
        self.assertEqual(residue.name, 'A')
        self.assertEqual(residue.structure_file_index, 0)
        self.assertEqual(residue.subsystem_index, 1)
        self.assertEqual(residue.number_of_atoms, 31)


class TestCreateBondsSimpleLoad(SimpleTrajectoryLoadTestCase):
    def setUp(self) -> None:
        super(TestCreateBondsSimpleLoad, self).setUp()

        self.residue_01 = self.system.__create_residue__(
            reference_residue=self.reference_trajectory.topology.residue(0), subsystem_index=0,
            atom_shifting_index=0)
        self.residue_02 = self.system.__create_residue__(
            reference_residue=self.reference_trajectory.topology.residue(1), subsystem_index=1,
            atom_shifting_index=31)

    def test_create_inner_covalent_bonds(self):
        self.system.__create_inner_covalent_bonds__(residue=self.residue_02,
                                                    bond_dict=self.molecule_dict.backbone_bonds_dictionary["residual"])

        atoms = self.residue_02.atoms

        self.assertTupleEqual(atoms[0].covalent_bond_partners, (atoms[1], atoms[2], atoms[3]))
        self.assertTupleEqual(atoms[1].covalent_bond_partners, (atoms[0],))
        self.assertTupleEqual(atoms[2].covalent_bond_partners, (atoms[0],))
        self.assertTupleEqual(atoms[3].covalent_bond_partners, (atoms[0], atoms[4]))
        self.assertTupleEqual(atoms[4].covalent_bond_partners, (atoms[3], atoms[5], atoms[6], atoms[7]))
        self.assertTupleEqual(atoms[5].covalent_bond_partners, (atoms[4],))
        self.assertTupleEqual(atoms[6].covalent_bond_partners, (atoms[4],))
        self.assertTupleEqual(atoms[7].covalent_bond_partners, (atoms[4], atoms[8], atoms[9], atoms[27]))
        self.assertTupleEqual(atoms[8].covalent_bond_partners, (atoms[7],))
        self.assertTupleEqual(atoms[9].covalent_bond_partners, (atoms[7], atoms[10]))
        self.assertTupleEqual(atoms[10].covalent_bond_partners, (atoms[9], atoms[11], atoms[29]))
        self.assertTupleEqual(atoms[11].covalent_bond_partners, (atoms[10],))

        for i in range(12, 27):
            self.assertTupleEqual(atoms[i].covalent_bond_partners, ())

        self.assertTupleEqual(atoms[27].covalent_bond_partners, (atoms[7], atoms[28], atoms[29], atoms[33]))
        self.assertTupleEqual(atoms[28].covalent_bond_partners, (atoms[27],))
        self.assertTupleEqual(atoms[29].covalent_bond_partners, (atoms[10], atoms[27], atoms[30], atoms[31]))
        self.assertTupleEqual(atoms[30].covalent_bond_partners, (atoms[29],))
        self.assertTupleEqual(atoms[31].covalent_bond_partners, (atoms[29], atoms[32]))
        self.assertTupleEqual(atoms[32].covalent_bond_partners, (atoms[31],))
        self.assertTupleEqual(atoms[33].covalent_bond_partners, (atoms[27],))

    def test_create_inter_covalent_bonds(self):
        self.system.__create_inter_covalent_bonds__(residue_01=self.residue_01, residue_02=self.residue_02,
                                                    molecule_dict=self.molecule_dict)

        self.assertTupleEqual(self.residue_01.atoms[30].covalent_bond_partners, (self.residue_02.atoms[0],))
        self.assertTupleEqual(self.residue_02.atoms[0].covalent_bond_partners, (self.residue_01.atoms[30],))


class TestElectronicStatesSimpleLoad(SimpleTrajectoryLoadTestCase):
    def setUp(self) -> None:
        super(TestElectronicStatesSimpleLoad, self).setUp()

        self.residue = self.system.__create_residue__(reference_residue=self.reference_trajectory.topology.residue(0),
                                                      subsystem_index=0, atom_shifting_index=0)

    def test_update_donors(self):
        self.system.__update_electronic_state__(residue=self.residue, is_donor=True, is_acceptor=False,
                                                electron_dict=self.molecule_dict.donors_dictionary['adenine'])

        donor_ids = (17, 18)

        for atom_id, atom in enumerate(self.residue.atoms):
            if atom_id in donor_ids:
                self.assertTrue(atom.is_donor_atom)
                self.assertEqual(atom.donor_slots, 1)
            else:
                self.assertFalse(atom.is_donor_atom)
                self.assertEqual(atom.donor_slots, 0)

            self.assertFalse(atom.is_acceptor)
            self.assertEqual(atom.acceptor_slots, 0)

    def test_update_acceptors(self):
        self.system.__update_electronic_state__(residue=self.residue, is_donor=False, is_acceptor=True,
                                                electron_dict=self.molecule_dict.acceptors_dictionary['adenine'])

        acceptor_ids = (13, 16, 19, 22)

        for atom_id, atom in enumerate(self.residue.atoms):
            if atom_id in acceptor_ids:
                self.assertTrue(atom.is_acceptor)
                self.assertEqual(atom.acceptor_slots, 1)
            else:
                self.assertFalse(atom.is_acceptor)
                self.assertEqual(atom.acceptor_slots, 0)

            self.assertFalse(atom.is_donor_atom)
            self.assertEqual(atom.donor_slots, 0)


class TestSelectorsSimpleLoad(SimpleTrajectoryLoadTestCase):
    def test_select_residues_with_generator(self):
        residues = self.system.__select_residues__(residue_ids=range(1, 4))

        self.assertEqual(len(residues), 3)

        # guanine
        index = 0

        self.assertEqual(residues[index].name, 'G')
        self.assertEqual(residues[index].structure_file_index, 1)
        self.assertEqual(residues[index].subsystem_index, 0)
        self.assertEqual(residues[index].atoms[0].structure_file_index, 31)
        self.assertEqual(residues[index].atoms[0].subsystem_index, 0)

        # cytosine
        index = 1

        self.assertEqual(residues[index].name, 'C')
        self.assertEqual(residues[index].structure_file_index, 2)
        self.assertEqual(residues[index].subsystem_index, 1)
        self.assertEqual(residues[index].atoms[0].structure_file_index, 65)
        self.assertEqual(residues[index].atoms[0].subsystem_index, 34)

        # uracil
        index = 2

        self.assertEqual(residues[index].name, 'U')
        self.assertEqual(residues[index].structure_file_index, 3)
        self.assertEqual(residues[index].subsystem_index, 2)
        self.assertEqual(residues[index].atoms[0].structure_file_index, 96)
        self.assertEqual(residues[index].atoms[0].subsystem_index, 65)

    def test_select_residues_two_residues(self):
        residues = self.system.__select_residues__(residue_ids=(1, 3))

        self.assertEqual(len(residues), 2)

        # guanine
        index = 0

        self.assertEqual(residues[index].name, 'G')
        self.assertEqual(residues[index].structure_file_index, 1)
        self.assertEqual(residues[index].subsystem_index, 0)
        self.assertEqual(residues[index].atoms[0].structure_file_index, 31)
        self.assertEqual(residues[index].atoms[0].subsystem_index, 0)

        # uracil
        index = 1

        self.assertEqual(residues[index].name, 'U')
        self.assertEqual(residues[index].structure_file_index, 3)
        self.assertEqual(residues[index].subsystem_index, 1)
        self.assertEqual(residues[index].atoms[0].structure_file_index, 96)
        self.assertEqual(residues[index].atoms[0].subsystem_index, 34)

    def test_select_residues_three_residues(self):
        residues = self.system.__select_residues__(residue_ids=(0, 1, 4))

        self.assertEqual(len(residues), 3)

        # adenine residue 00
        index = 0

        self.assertEqual(residues[index].name, 'A')
        self.assertEqual(residues[index].structure_file_index, 0)
        self.assertEqual(residues[index].subsystem_index, 0)
        self.assertEqual(residues[index].atoms[0].structure_file_index, 0)
        self.assertEqual(residues[index].atoms[0].subsystem_index, 0)

        # guanine residue 01
        index = 1

        self.assertEqual(residues[index].name, 'G')
        self.assertEqual(residues[index].structure_file_index, 1)
        self.assertEqual(residues[index].subsystem_index, 1)
        self.assertEqual(residues[index].atoms[0].structure_file_index, 31)
        self.assertEqual(residues[index].atoms[0].subsystem_index, 31)

        # guanine residue 04
        index = 2

        self.assertEqual(residues[index].name, 'G')
        self.assertEqual(residues[index].structure_file_index, 4)
        self.assertEqual(residues[index].subsystem_index, 2)
        self.assertEqual(residues[index].atoms[0].structure_file_index, 126)
        self.assertEqual(residues[index].atoms[0].subsystem_index, 65)


class TestSelectRnaSimpleLoad(SimpleTrajectoryLoadTestCase):
    def setUp(self) -> None:
        super(TestSelectRnaSimpleLoad, self).setUp()

        self.system.select_rna(residue_ids=(0, 1, 4), name='some_virus_rna')

        self.atoms_res_00 = self.system.molecules['some_virus_rna'].residues[0].atoms
        self.atoms_res_01 = self.system.molecules['some_virus_rna'].residues[1].atoms
        self.atoms_res_02 = self.system.molecules['some_virus_rna'].residues[2].atoms

    def test_covalent_bonds(self):
        # adenine residue 00
        # backbone
        self.assertTupleEqual(self.atoms_res_00[0].covalent_bond_partners, (self.atoms_res_00[1], self.atoms_res_00[2]))
        self.assertTupleEqual(self.atoms_res_00[1].covalent_bond_partners, (self.atoms_res_00[0],))
        self.assertTupleEqual(self.atoms_res_00[2].covalent_bond_partners,
                              (self.atoms_res_00[0], self.atoms_res_00[3], self.atoms_res_00[4], self.atoms_res_00[5]))

        self.assertTupleEqual(self.atoms_res_00[3].covalent_bond_partners, (self.atoms_res_00[2],))
        self.assertTupleEqual(self.atoms_res_00[4].covalent_bond_partners, (self.atoms_res_00[2],))
        self.assertTupleEqual(self.atoms_res_00[5].covalent_bond_partners,
                              (self.atoms_res_00[2], self.atoms_res_00[6], self.atoms_res_00[7], self.atoms_res_00[24]))
        self.assertTupleEqual(self.atoms_res_00[6].covalent_bond_partners, (self.atoms_res_00[5],))
        self.assertTupleEqual(self.atoms_res_00[7].covalent_bond_partners, (self.atoms_res_00[5], self.atoms_res_00[8]))
        self.assertTupleEqual(self.atoms_res_00[8].covalent_bond_partners, (
            self.atoms_res_00[7], self.atoms_res_00[9], self.atoms_res_00[26], self.atoms_res_00[10]))
        self.assertTupleEqual(self.atoms_res_00[9].covalent_bond_partners, (self.atoms_res_00[8],))

        # base
        self.assertTupleEqual(self.atoms_res_00[10].covalent_bond_partners,
                              (self.atoms_res_00[8], self.atoms_res_00[11], self.atoms_res_00[23]))
        self.assertTupleEqual(self.atoms_res_00[11].covalent_bond_partners,
                              (self.atoms_res_00[10], self.atoms_res_00[12], self.atoms_res_00[13]))
        self.assertTupleEqual(self.atoms_res_00[12].covalent_bond_partners, (self.atoms_res_00[11],))
        self.assertTupleEqual(self.atoms_res_00[13].covalent_bond_partners,
                              (self.atoms_res_00[11], self.atoms_res_00[14]))
        self.assertTupleEqual(self.atoms_res_00[14].covalent_bond_partners,
                              (self.atoms_res_00[13], self.atoms_res_00[15], self.atoms_res_00[23]))
        self.assertTupleEqual(self.atoms_res_00[15].covalent_bond_partners,
                              (self.atoms_res_00[14], self.atoms_res_00[16], self.atoms_res_00[19]))
        self.assertTupleEqual(self.atoms_res_00[16].covalent_bond_partners,
                              (self.atoms_res_00[15], self.atoms_res_00[17], self.atoms_res_00[18]))
        self.assertTupleEqual(self.atoms_res_00[17].covalent_bond_partners, (self.atoms_res_00[16],))
        self.assertTupleEqual(self.atoms_res_00[18].covalent_bond_partners, (self.atoms_res_00[16],))
        self.assertTupleEqual(self.atoms_res_00[19].covalent_bond_partners,
                              (self.atoms_res_00[15], self.atoms_res_00[20]))
        self.assertTupleEqual(self.atoms_res_00[20].covalent_bond_partners,
                              (self.atoms_res_00[19], self.atoms_res_00[21], self.atoms_res_00[22]))
        self.assertTupleEqual(self.atoms_res_00[21].covalent_bond_partners, (self.atoms_res_00[20],))
        self.assertTupleEqual(self.atoms_res_00[22].covalent_bond_partners,
                              (self.atoms_res_00[20], self.atoms_res_00[23]))
        self.assertTupleEqual(self.atoms_res_00[23].covalent_bond_partners,
                              (self.atoms_res_00[10], self.atoms_res_00[14], self.atoms_res_00[22]))

        # backbone
        self.assertTupleEqual(self.atoms_res_00[24].covalent_bond_partners, (
            self.atoms_res_00[5], self.atoms_res_00[25], self.atoms_res_00[26], self.atoms_res_00[30]))
        self.assertTupleEqual(self.atoms_res_00[25].covalent_bond_partners, (self.atoms_res_00[24],))
        self.assertTupleEqual(self.atoms_res_00[26].covalent_bond_partners, (
            self.atoms_res_00[8], self.atoms_res_00[24], self.atoms_res_00[27], self.atoms_res_00[28]))
        self.assertTupleEqual(self.atoms_res_00[27].covalent_bond_partners, (self.atoms_res_00[26],))
        self.assertTupleEqual(self.atoms_res_00[28].covalent_bond_partners,
                              (self.atoms_res_00[26], self.atoms_res_00[29]))
        self.assertTupleEqual(self.atoms_res_00[29].covalent_bond_partners, (self.atoms_res_00[28],))
        self.assertTupleEqual(self.atoms_res_00[30].covalent_bond_partners,
                              (self.atoms_res_00[24], self.atoms_res_01[0]))

        # GUANINE RESIDUE 01
        # backbone
        self.assertTupleEqual(self.atoms_res_01[0].covalent_bond_partners,
                              (self.atoms_res_00[30], self.atoms_res_01[1], self.atoms_res_01[2], self.atoms_res_01[3]))
        self.assertTupleEqual(self.atoms_res_01[1].covalent_bond_partners, (self.atoms_res_01[0],))
        self.assertTupleEqual(self.atoms_res_01[2].covalent_bond_partners, (self.atoms_res_01[0],))
        self.assertTupleEqual(self.atoms_res_01[3].covalent_bond_partners, (self.atoms_res_01[0], self.atoms_res_01[4]))
        self.assertTupleEqual(self.atoms_res_01[4].covalent_bond_partners,
                              (self.atoms_res_01[3], self.atoms_res_01[5], self.atoms_res_01[6], self.atoms_res_01[7]))
        self.assertTupleEqual(self.atoms_res_01[5].covalent_bond_partners, (self.atoms_res_01[4],))
        self.assertTupleEqual(self.atoms_res_01[6].covalent_bond_partners, (self.atoms_res_01[4],))
        self.assertTupleEqual(self.atoms_res_01[7].covalent_bond_partners,
                              (self.atoms_res_01[4], self.atoms_res_01[8], self.atoms_res_01[9], self.atoms_res_01[27]))
        self.assertTupleEqual(self.atoms_res_01[8].covalent_bond_partners, (self.atoms_res_01[7],))
        self.assertTupleEqual(self.atoms_res_01[9].covalent_bond_partners,
                              (self.atoms_res_01[7], self.atoms_res_01[10]))
        self.assertTupleEqual(self.atoms_res_01[10].covalent_bond_partners, (
            self.atoms_res_01[9], self.atoms_res_01[11], self.atoms_res_01[29], self.atoms_res_01[12]))
        self.assertTupleEqual(self.atoms_res_01[11].covalent_bond_partners, (self.atoms_res_01[10],))

        # base
        self.assertTupleEqual(self.atoms_res_01[12].covalent_bond_partners,
                              (self.atoms_res_01[10], self.atoms_res_01[13], self.atoms_res_01[26]))
        self.assertTupleEqual(self.atoms_res_01[13].covalent_bond_partners,
                              (self.atoms_res_01[12], self.atoms_res_01[14], self.atoms_res_01[15]))
        self.assertTupleEqual(self.atoms_res_01[14].covalent_bond_partners, (self.atoms_res_01[13],))
        self.assertTupleEqual(self.atoms_res_01[15].covalent_bond_partners,
                              (self.atoms_res_01[13], self.atoms_res_01[16]))
        self.assertTupleEqual(self.atoms_res_01[16].covalent_bond_partners,
                              (self.atoms_res_01[15], self.atoms_res_01[17], self.atoms_res_01[26]))
        self.assertTupleEqual(self.atoms_res_01[17].covalent_bond_partners,
                              (self.atoms_res_01[16], self.atoms_res_01[18], self.atoms_res_01[19]))
        self.assertTupleEqual(self.atoms_res_01[18].covalent_bond_partners, (self.atoms_res_01[17],))
        self.assertTupleEqual(self.atoms_res_01[19].covalent_bond_partners,
                              (self.atoms_res_01[17], self.atoms_res_01[20], self.atoms_res_01[21]))
        self.assertTupleEqual(self.atoms_res_01[20].covalent_bond_partners, (self.atoms_res_01[19],))
        self.assertTupleEqual(self.atoms_res_01[21].covalent_bond_partners,
                              (self.atoms_res_01[19], self.atoms_res_01[22], self.atoms_res_01[25]))
        self.assertTupleEqual(self.atoms_res_01[22].covalent_bond_partners,
                              (self.atoms_res_01[21], self.atoms_res_01[23], self.atoms_res_01[24]))
        self.assertTupleEqual(self.atoms_res_01[23].covalent_bond_partners, (self.atoms_res_01[22],))
        self.assertTupleEqual(self.atoms_res_01[24].covalent_bond_partners, (self.atoms_res_01[22],))
        self.assertTupleEqual(self.atoms_res_01[25].covalent_bond_partners,
                              (self.atoms_res_01[21], self.atoms_res_01[26]))
        self.assertTupleEqual(self.atoms_res_01[26].covalent_bond_partners,
                              (self.atoms_res_01[12], self.atoms_res_01[16], self.atoms_res_01[25]))

        # backbone
        self.assertTupleEqual(self.atoms_res_01[27].covalent_bond_partners, (
            self.atoms_res_01[7], self.atoms_res_01[28], self.atoms_res_01[29], self.atoms_res_01[33]))
        self.assertTupleEqual(self.atoms_res_01[28].covalent_bond_partners, (self.atoms_res_01[27],))
        self.assertTupleEqual(self.atoms_res_01[29].covalent_bond_partners, (
            self.atoms_res_01[10], self.atoms_res_01[27], self.atoms_res_01[30], self.atoms_res_01[31]))
        self.assertTupleEqual(self.atoms_res_01[30].covalent_bond_partners, (self.atoms_res_01[29],))
        self.assertTupleEqual(self.atoms_res_01[31].covalent_bond_partners,
                              (self.atoms_res_01[29], self.atoms_res_01[32]))
        self.assertTupleEqual(self.atoms_res_01[32].covalent_bond_partners, (self.atoms_res_01[31],))
        self.assertTupleEqual(self.atoms_res_01[33].covalent_bond_partners,
                              (self.atoms_res_01[27], self.atoms_res_02[0]))

        # GUANINE RESIDUE 04
        # backbone
        self.assertTupleEqual(self.atoms_res_02[0].covalent_bond_partners,
                              (self.atoms_res_01[33], self.atoms_res_02[1], self.atoms_res_02[2], self.atoms_res_02[3]))
        self.assertTupleEqual(self.atoms_res_02[1].covalent_bond_partners, (self.atoms_res_02[0],))
        self.assertTupleEqual(self.atoms_res_02[2].covalent_bond_partners, (self.atoms_res_02[0],))
        self.assertTupleEqual(self.atoms_res_02[3].covalent_bond_partners, (self.atoms_res_02[0], self.atoms_res_02[4]))
        self.assertTupleEqual(self.atoms_res_02[4].covalent_bond_partners,
                              (self.atoms_res_02[3], self.atoms_res_02[5], self.atoms_res_02[6], self.atoms_res_02[7]))
        self.assertTupleEqual(self.atoms_res_02[5].covalent_bond_partners, (self.atoms_res_02[4],))
        self.assertTupleEqual(self.atoms_res_02[6].covalent_bond_partners, (self.atoms_res_02[4],))
        self.assertTupleEqual(self.atoms_res_02[7].covalent_bond_partners,
                              (self.atoms_res_02[4], self.atoms_res_02[8], self.atoms_res_02[9], self.atoms_res_02[27]))
        self.assertTupleEqual(self.atoms_res_02[8].covalent_bond_partners, (self.atoms_res_02[7],))
        self.assertTupleEqual(self.atoms_res_02[9].covalent_bond_partners,
                              (self.atoms_res_02[7], self.atoms_res_02[10]))
        self.assertTupleEqual(self.atoms_res_02[10].covalent_bond_partners, (
            self.atoms_res_02[9], self.atoms_res_02[11], self.atoms_res_02[29], self.atoms_res_02[12]))
        self.assertTupleEqual(self.atoms_res_02[11].covalent_bond_partners, (self.atoms_res_02[10],))

        # base
        self.assertTupleEqual(self.atoms_res_02[12].covalent_bond_partners,
                              (self.atoms_res_02[10], self.atoms_res_02[13], self.atoms_res_02[26]))
        self.assertTupleEqual(self.atoms_res_02[13].covalent_bond_partners,
                              (self.atoms_res_02[12], self.atoms_res_02[14], self.atoms_res_02[15]))
        self.assertTupleEqual(self.atoms_res_02[14].covalent_bond_partners, (self.atoms_res_02[13],))
        self.assertTupleEqual(self.atoms_res_02[15].covalent_bond_partners,
                              (self.atoms_res_02[13], self.atoms_res_02[16]))
        self.assertTupleEqual(self.atoms_res_02[16].covalent_bond_partners,
                              (self.atoms_res_02[15], self.atoms_res_02[17], self.atoms_res_02[26]))
        self.assertTupleEqual(self.atoms_res_02[17].covalent_bond_partners,
                              (self.atoms_res_02[16], self.atoms_res_02[18], self.atoms_res_02[19]))
        self.assertTupleEqual(self.atoms_res_02[18].covalent_bond_partners, (self.atoms_res_02[17],))
        self.assertTupleEqual(self.atoms_res_02[19].covalent_bond_partners,
                              (self.atoms_res_02[17], self.atoms_res_02[20], self.atoms_res_02[21]))
        self.assertTupleEqual(self.atoms_res_02[20].covalent_bond_partners, (self.atoms_res_02[19],))
        self.assertTupleEqual(self.atoms_res_02[21].covalent_bond_partners,
                              (self.atoms_res_02[19], self.atoms_res_02[22], self.atoms_res_02[25]))
        self.assertTupleEqual(self.atoms_res_02[22].covalent_bond_partners,
                              (self.atoms_res_02[21], self.atoms_res_02[23], self.atoms_res_02[24]))
        self.assertTupleEqual(self.atoms_res_02[23].covalent_bond_partners, (self.atoms_res_02[22],))
        self.assertTupleEqual(self.atoms_res_02[24].covalent_bond_partners, (self.atoms_res_02[22],))
        self.assertTupleEqual(self.atoms_res_02[25].covalent_bond_partners,
                              (self.atoms_res_02[21], self.atoms_res_02[26]))
        self.assertTupleEqual(self.atoms_res_02[26].covalent_bond_partners,
                              (self.atoms_res_02[12], self.atoms_res_02[16], self.atoms_res_02[25]))

        # backbone
        self.assertTupleEqual(self.atoms_res_02[27].covalent_bond_partners, (
            self.atoms_res_02[7], self.atoms_res_02[28], self.atoms_res_02[29], self.atoms_res_02[33]))
        self.assertTupleEqual(self.atoms_res_02[28].covalent_bond_partners, (self.atoms_res_02[27],))
        self.assertTupleEqual(self.atoms_res_02[29].covalent_bond_partners, (
            self.atoms_res_02[10], self.atoms_res_02[27], self.atoms_res_02[30], self.atoms_res_02[31]))
        self.assertTupleEqual(self.atoms_res_02[30].covalent_bond_partners, (self.atoms_res_02[29],))
        self.assertTupleEqual(self.atoms_res_02[31].covalent_bond_partners,
                              (self.atoms_res_02[29], self.atoms_res_02[32]))
        self.assertTupleEqual(self.atoms_res_02[32].covalent_bond_partners, (self.atoms_res_02[31],))
        self.assertTupleEqual(self.atoms_res_02[33].covalent_bond_partners,
                              (self.atoms_res_02[34], self.atoms_res_02[27]))
        self.assertTupleEqual(self.atoms_res_02[34].covalent_bond_partners, (self.atoms_res_02[33],))

    def test_electronic_states(self):
        # adenine residue 00
        atoms_with_electronic_state = [0, 7, 13, 16, 17, 18, 19, 22, 28, 29, 30]

        # acceptors
        self.assertTrue(self.atoms_res_00[0].is_acceptor)
        self.assertEqual(self.atoms_res_00[0].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_00[0].is_donor_atom)
        self.assertEqual(self.atoms_res_00[0].donor_slots, 0)

        self.assertTrue(self.atoms_res_00[7].is_acceptor)
        self.assertEqual(self.atoms_res_00[7].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_00[7].is_donor_atom)
        self.assertEqual(self.atoms_res_00[7].donor_slots, 0)

        self.assertTrue(self.atoms_res_00[13].is_acceptor)
        self.assertEqual(self.atoms_res_00[13].acceptor_slots, 1)
        self.assertFalse(self.atoms_res_00[13].is_donor_atom)
        self.assertEqual(self.atoms_res_00[13].donor_slots, 0)

        self.assertTrue(self.atoms_res_00[16].is_acceptor)
        self.assertEqual(self.atoms_res_00[16].acceptor_slots, 1)
        self.assertFalse(self.atoms_res_00[16].is_donor_atom)
        self.assertEqual(self.atoms_res_00[16].donor_slots, 0)

        self.assertTrue(self.atoms_res_00[19].is_acceptor)
        self.assertEqual(self.atoms_res_00[19].acceptor_slots, 1)
        self.assertFalse(self.atoms_res_00[19].is_donor_atom)
        self.assertEqual(self.atoms_res_00[19].donor_slots, 0)

        self.assertTrue(self.atoms_res_00[22].is_acceptor)
        self.assertEqual(self.atoms_res_00[22].acceptor_slots, 1)
        self.assertFalse(self.atoms_res_00[22].is_donor_atom)
        self.assertEqual(self.atoms_res_00[22].donor_slots, 0)

        self.assertTrue(self.atoms_res_00[28].is_acceptor)
        self.assertEqual(self.atoms_res_00[28].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_00[28].is_donor_atom)
        self.assertEqual(self.atoms_res_00[28].donor_slots, 0)

        self.assertTrue(self.atoms_res_00[30].is_acceptor)
        self.assertEqual(self.atoms_res_00[30].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_00[30].is_donor_atom)
        self.assertEqual(self.atoms_res_00[30].donor_slots, 0)

        # donors
        self.assertFalse(self.atoms_res_00[17].is_acceptor)
        self.assertEqual(self.atoms_res_00[17].acceptor_slots, 0)
        self.assertTrue(self.atoms_res_00[17].is_donor_atom)
        self.assertEqual(self.atoms_res_00[17].donor_slots, 1)

        self.assertFalse(self.atoms_res_00[18].is_acceptor)
        self.assertEqual(self.atoms_res_00[18].acceptor_slots, 0)
        self.assertTrue(self.atoms_res_00[18].is_donor_atom)
        self.assertEqual(self.atoms_res_00[18].donor_slots, 1)

        self.assertFalse(self.atoms_res_00[29].is_acceptor)
        self.assertEqual(self.atoms_res_00[29].acceptor_slots, 0)
        self.assertTrue(self.atoms_res_00[29].is_donor_atom)
        self.assertEqual(self.atoms_res_00[29].donor_slots, 1)

        # non electric state atoms
        for i, atom in enumerate(self.atoms_res_00):
            if i in atoms_with_electronic_state:
                continue
            else:
                self.assertFalse(self.atoms_res_00[i].is_acceptor)
                self.assertEqual(self.atoms_res_00[i].acceptor_slots, 0)
                self.assertFalse(self.atoms_res_00[i].is_donor_atom)
                self.assertEqual(self.atoms_res_00[i].donor_slots, 0)

        # guanine residue 01
        atoms_with_electronic_state = [1, 2, 3, 9, 15, 18, 20, 23, 24, 25, 31, 32, 33]

        # acceptors
        self.assertTrue(self.atoms_res_01[1].is_acceptor)
        self.assertEqual(self.atoms_res_01[1].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_01[1].is_donor_atom)
        self.assertEqual(self.atoms_res_01[1].donor_slots, 0)

        self.assertTrue(self.atoms_res_01[2].is_acceptor)
        self.assertEqual(self.atoms_res_01[2].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_01[2].is_donor_atom)
        self.assertEqual(self.atoms_res_01[2].donor_slots, 0)

        self.assertTrue(self.atoms_res_01[3].is_acceptor)
        self.assertEqual(self.atoms_res_01[3].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_01[3].is_donor_atom)
        self.assertEqual(self.atoms_res_01[3].donor_slots, 0)

        self.assertTrue(self.atoms_res_01[9].is_acceptor)
        self.assertEqual(self.atoms_res_01[9].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_01[9].is_donor_atom)
        self.assertEqual(self.atoms_res_01[9].donor_slots, 0)

        self.assertTrue(self.atoms_res_01[15].is_acceptor)
        self.assertEqual(self.atoms_res_01[15].acceptor_slots, 1)
        self.assertFalse(self.atoms_res_01[15].is_donor_atom)
        self.assertEqual(self.atoms_res_01[15].donor_slots, 0)

        self.assertTrue(self.atoms_res_01[18].is_acceptor)
        self.assertEqual(self.atoms_res_01[18].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_01[18].is_donor_atom)
        self.assertEqual(self.atoms_res_01[18].donor_slots, 0)

        self.assertTrue(self.atoms_res_01[25].is_acceptor)
        self.assertEqual(self.atoms_res_01[25].acceptor_slots, 1)
        self.assertFalse(self.atoms_res_01[25].is_donor_atom)
        self.assertEqual(self.atoms_res_01[25].donor_slots, 0)

        self.assertTrue(self.atoms_res_01[31].is_acceptor)
        self.assertEqual(self.atoms_res_01[31].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_01[31].is_donor_atom)
        self.assertEqual(self.atoms_res_01[31].donor_slots, 0)

        self.assertTrue(self.atoms_res_01[33].is_acceptor)
        self.assertEqual(self.atoms_res_01[33].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_01[33].is_donor_atom)
        self.assertEqual(self.atoms_res_01[33].donor_slots, 0)

        # donors
        self.assertFalse(self.atoms_res_01[20].is_acceptor)
        self.assertEqual(self.atoms_res_01[20].acceptor_slots, 0)
        self.assertTrue(self.atoms_res_01[20].is_donor_atom)
        self.assertEqual(self.atoms_res_01[20].donor_slots, 1)

        self.assertFalse(self.atoms_res_01[23].is_acceptor)
        self.assertEqual(self.atoms_res_01[23].acceptor_slots, 0)
        self.assertTrue(self.atoms_res_01[23].is_donor_atom)
        self.assertEqual(self.atoms_res_01[23].donor_slots, 1)

        self.assertFalse(self.atoms_res_01[24].is_acceptor)
        self.assertEqual(self.atoms_res_01[24].acceptor_slots, 0)
        self.assertTrue(self.atoms_res_01[24].is_donor_atom)
        self.assertEqual(self.atoms_res_01[24].donor_slots, 1)

        self.assertFalse(self.atoms_res_01[32].is_acceptor)
        self.assertEqual(self.atoms_res_01[32].acceptor_slots, 0)
        self.assertTrue(self.atoms_res_01[32].is_donor_atom)
        self.assertEqual(self.atoms_res_01[32].donor_slots, 1)

        # non electric state atoms
        for i, atom in enumerate(self.atoms_res_01):
            if i in atoms_with_electronic_state:
                continue
            else:
                self.assertFalse(self.atoms_res_01[i].is_acceptor)
                self.assertEqual(self.atoms_res_01[i].acceptor_slots, 0)
                self.assertFalse(self.atoms_res_01[i].is_donor_atom)
                self.assertEqual(self.atoms_res_01[i].donor_slots, 0)

        # guanine residue 02
        atoms_with_electronic_state = [1, 2, 3, 9, 15, 18, 20, 23, 24, 25, 31, 32, 33]

        # acceptors
        self.assertTrue(self.atoms_res_02[1].is_acceptor)
        self.assertEqual(self.atoms_res_02[1].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_02[1].is_donor_atom)
        self.assertEqual(self.atoms_res_02[1].donor_slots, 0)

        self.assertTrue(self.atoms_res_02[2].is_acceptor)
        self.assertEqual(self.atoms_res_02[2].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_02[2].is_donor_atom)
        self.assertEqual(self.atoms_res_02[2].donor_slots, 0)

        self.assertTrue(self.atoms_res_02[3].is_acceptor)
        self.assertEqual(self.atoms_res_02[3].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_02[3].is_donor_atom)
        self.assertEqual(self.atoms_res_02[3].donor_slots, 0)

        self.assertTrue(self.atoms_res_02[9].is_acceptor)
        self.assertEqual(self.atoms_res_02[9].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_02[9].is_donor_atom)
        self.assertEqual(self.atoms_res_02[9].donor_slots, 0)

        self.assertTrue(self.atoms_res_02[15].is_acceptor)
        self.assertEqual(self.atoms_res_02[15].acceptor_slots, 1)
        self.assertFalse(self.atoms_res_02[15].is_donor_atom)
        self.assertEqual(self.atoms_res_02[15].donor_slots, 0)

        self.assertTrue(self.atoms_res_02[18].is_acceptor)
        self.assertEqual(self.atoms_res_02[18].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_02[18].is_donor_atom)
        self.assertEqual(self.atoms_res_02[18].donor_slots, 0)

        self.assertTrue(self.atoms_res_02[25].is_acceptor)
        self.assertEqual(self.atoms_res_02[25].acceptor_slots, 1)
        self.assertFalse(self.atoms_res_02[25].is_donor_atom)
        self.assertEqual(self.atoms_res_02[25].donor_slots, 0)

        self.assertTrue(self.atoms_res_02[31].is_acceptor)
        self.assertEqual(self.atoms_res_02[31].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_02[31].is_donor_atom)
        self.assertEqual(self.atoms_res_02[31].donor_slots, 0)

        self.assertTrue(self.atoms_res_02[33].is_acceptor)
        self.assertEqual(self.atoms_res_02[33].acceptor_slots, 2)
        self.assertFalse(self.atoms_res_02[33].is_donor_atom)
        self.assertEqual(self.atoms_res_02[33].donor_slots, 0)

        # donors
        self.assertFalse(self.atoms_res_02[20].is_acceptor)
        self.assertEqual(self.atoms_res_02[20].acceptor_slots, 0)
        self.assertTrue(self.atoms_res_02[20].is_donor_atom)
        self.assertEqual(self.atoms_res_02[20].donor_slots, 1)

        self.assertFalse(self.atoms_res_02[23].is_acceptor)
        self.assertEqual(self.atoms_res_02[23].acceptor_slots, 0)
        self.assertTrue(self.atoms_res_02[23].is_donor_atom)
        self.assertEqual(self.atoms_res_02[23].donor_slots, 1)

        self.assertFalse(self.atoms_res_02[24].is_acceptor)
        self.assertEqual(self.atoms_res_02[24].acceptor_slots, 0)
        self.assertTrue(self.atoms_res_02[24].is_donor_atom)
        self.assertEqual(self.atoms_res_02[24].donor_slots, 1)

        self.assertFalse(self.atoms_res_02[32].is_acceptor)
        self.assertEqual(self.atoms_res_02[32].acceptor_slots, 0)
        self.assertTrue(self.atoms_res_02[32].is_donor_atom)
        self.assertEqual(self.atoms_res_02[32].donor_slots, 1)

        # non electric state atoms
        for i, atom in enumerate(self.atoms_res_02):
            if i in atoms_with_electronic_state:
                continue
            else:
                self.assertFalse(self.atoms_res_02[i].is_acceptor)
                self.assertEqual(self.atoms_res_02[i].acceptor_slots, 0)
                self.assertFalse(self.atoms_res_02[i].is_donor_atom)
                self.assertEqual(self.atoms_res_02[i].donor_slots, 0)

    def test_rna_object(self):
        from yeti.systems.molecules.nucleic_acids import RNA

        self.assertListEqual(list(self.system.molecules.keys()), ['some_virus_rna'])
        self.assertEqual(type(self.system.molecules['some_virus_rna']), RNA)


class IterTrajectoryLoadTestCase(SystemTestCase):
    def setUp(self) -> None:
        super(IterTrajectoryLoadTestCase, self).setUp()

        self.system.load_trajectory(trajectory_file_path=self.xtc_file_path, topology_file_path=self.gro_file_path,
                                    chunk_size=2)


class TestSelectorsIterLoad(IterTrajectoryLoadTestCase, TestSelectorsSimpleLoad):
    pass


class TestSelectRnaIterLoad(IterTrajectoryLoadTestCase, TestSelectRnaSimpleLoad):
    pass


if __name__ == '__main__':
    unittest.main()
