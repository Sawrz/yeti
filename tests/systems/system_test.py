import os
import unittest

import numpy.testing as npt

from tests.blueprints_test import BlueprintTestCase


class SystemTestCase(BlueprintTestCase):
    def setUp(self) -> None:
        import mdtraj as md
        from yeti.systems.system import System
        from yeti.dictionaries.molecules.biomolecules import RNA

        test_data_path = '../test_data/'
        self.xtc_file_path = os.path.join(test_data_path, 'AGCUG.xtc')
        self.gro_file_path = os.path.join(test_data_path, 'AGCUG.gro')

        self.reference_trajectory = md.load(filename_or_filenames=self.xtc_file_path, top=self.gro_file_path)
        self.system = System(trajectory_file_path=self.xtc_file_path, topology_file_path=self.gro_file_path,
                             periodic=True)

        self.molecule_dict = RNA()


class TestStandardMethods(SystemTestCase):
    def test_init(self):
        self.assertEqual(self.system.trajectory_file_path, self.xtc_file_path)
        self.assertEqual(self.system.topology_file_path, self.gro_file_path)

        self.assertEqual(self.system.trajectory, self.reference_trajectory)

        # TODO: add arrays instead of references
        npt.assert_equal(self.system.unitcell_angles, self.reference_trajectory.unitcell_angles)
        npt.assert_equal(self.system.unitcell_vectors, self.reference_trajectory.unitcell_vectors)

        self.assertEqual(self.system.number_of_frames, 6)
        self.assertDictEqual(self.system.molecules, {})


class TestCreateBuildingBlockMethods(SystemTestCase):
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


class TestCreateBonds(SystemTestCase):
    def setUp(self) -> None:
        super(TestCreateBonds, self).setUp()

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


class TestElectronicStates(SystemTestCase):
    def setUp(self) -> None:
        super(TestElectronicStates, self).setUp()

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


class TestSelectors(SystemTestCase):
    def test_select_residues_is_range(self):
        residues = self.system.__select_residues__(residue_ids=(1, 3), is_range=True)

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

    def test_select_residues_is_not_range_two(self):
        residues = self.system.__select_residues__(residue_ids=(1, 3), is_range=False)

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

    def test_select_residues_is_not_range_three(self):
        residues = self.system.__select_residues__(residue_ids=(0, 1, 4), is_range=False)

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


class TestSelectRna(SystemTestCase):
    def setUp(self) -> None:
        super(TestSelectRna, self).setUp()

        self.system.select_rna(residue_ids=(0, 1, 4), name='RNA', distance_cutoff=0.1, angle_cutoff=2.1, is_range=False)

        self.atoms_res_00 = self.system.molecules['RNA'].residues[0].atoms
        self.atoms_res_01 = self.system.molecules['RNA'].residues[1].atoms
        self.atoms_res_02 = self.system.molecules['RNA'].residues[2].atoms

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


if __name__ == '__main__':
    unittest.main()
