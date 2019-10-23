import unittest

import numpy as np

from tests.blueprints_test import BlueprintExceptionsTestCase
from tests.get_features.hydrogen_bonds_test import HydrogenBondTestCase


class HydrogenBondFiFoTestCase(HydrogenBondTestCase):
    def setUpAtoms(self) -> None:
        self.atoms = (self.donor_01, self.donor_atom_01, self.acceptor_01,
                      self.donor_02, self.donor_atom_02, self.acceptor_02,
                      self.donor_03, self.donor_atom_03, self.acceptor_03)

    def setUp(self) -> None:
        from yeti.systems.building_blocks import Atom
        from yeti.get_features.hydrogen_bonds import HydrogenBondsFirstComesFirstServes

        super(HydrogenBondFiFoTestCase, self).setUp()

        # create new objects
        self.donor_02.xyz_trajectory -= 0.2
        self.donor_atom_02.xyz_trajectory -= 0.2
        self.acceptor_02.xyz_trajectory -= 0.2

        self.donor_03 = Atom(structure_file_index=8, subsystem_index=10, name='G',
                             xyz_trajectory=np.array([[0.1, 0.4, 0.3], [0.1, 0.4, 0.3], [0.1, 0.4, 0.3]]))
        self.donor_atom_03 = Atom(structure_file_index=9, subsystem_index=11, name='H',
                                  xyz_trajectory=np.array([[0.1, 0.5, 0.2], [0.1, 0.5, 0.2], [0.5, 0.5, 0.2]]))
        self.acceptor_03 = Atom(structure_file_index=10, subsystem_index=11, name='I',
                                xyz_trajectory=np.array([[0.1, 0.6, 0.4], [0.1, 0.7, 0.4], [0.1, 0.6, 0.4]]))

        self.donor_03.add_covalent_bond(atom=self.donor_atom_03)
        self.donor_atom_03.update_donor_state(is_donor_atom=True, donor_slots=1)
        self.acceptor_03.update_acceptor_state(is_acceptor=True, acceptor_slots=2)

        self.setUpAtoms()
        self.hydrogen_bonds = HydrogenBondsFirstComesFirstServes(atoms=self.atoms, periodic=True,
                                                                 unit_cell_angles=self.unit_cell_angles,
                                                                 unit_cell_vectors=self.unit_cell_vectors,
                                                                 system_name='test',
                                                                 number_of_frames=self.number_of_frames)

        self.triplets = self.hydrogen_bonds.__build_triplets__(distance_cutoff=0.25, angle_cutoff=2.0)


class TestHydrogenBondRepresentationMethodsTooManyDonors(HydrogenBondFiFoTestCase):
    def setUpAtoms(self) -> None:
        self.atoms = (self.donor_01, self.donor_atom_01, self.acceptor_01,
                      self.donor_02, self.donor_atom_02,
                      self.donor_03, self.donor_atom_03)

    def test_get_hydrogen_bonds_in_frame_too_many_donors(self):
        self.hydrogen_bonds.__get_hydrogen_bonds_in_frame__(triplets=self.triplets, frame=0)

        self.assertIsNone(self.hydrogen_bonds.atoms[0].hydrogen_bond_partners)
        self.assertIsNone(self.hydrogen_bonds.atoms[3].hydrogen_bond_partners)
        self.assertIsNone(self.hydrogen_bonds.atoms[5].hydrogen_bond_partners)
        self.assertListEqual([[self.acceptor_01], [], []], self.hydrogen_bonds.atoms[1].hydrogen_bond_partners['test'])
        self.assertListEqual([[self.donor_atom_01, self.donor_atom_02], [], []],
                             self.hydrogen_bonds.atoms[2].hydrogen_bond_partners['test'])
        self.assertListEqual([[self.acceptor_01], [], []], self.hydrogen_bonds.atoms[4].hydrogen_bond_partners['test'])
        self.assertListEqual([[], [], []], self.hydrogen_bonds.atoms[6].hydrogen_bond_partners['test'])

class TestHydrogenBondRepresentationMethodsTooManyAcceptors(HydrogenBondFiFoTestCase):
    def setUpAtoms(self) -> None:
        self.atoms = (self.donor_01, self.donor_atom_01, self.acceptor_01, self.acceptor_02, self.acceptor_03)

    def test_get_hydrogen_bonds_in_frame_too_many_donors(self):
        self.hydrogen_bonds.__get_hydrogen_bonds_in_frame__(triplets=self.triplets, frame=0)

        self.assertIsNone(self.donor_01.hydrogen_bond_partners)
        self.assertListEqual([[self.acceptor_01], [], []], self.hydrogen_bonds.atoms[1].hydrogen_bond_partners['test'])
        self.assertListEqual([[self.donor_atom_01], [], []], self.hydrogen_bonds.atoms[2].hydrogen_bond_partners['test'])
        self.assertListEqual([[], [], []], self.hydrogen_bonds.atoms[3].hydrogen_bond_partners['test'])
        self.assertListEqual([[], [], []], self.hydrogen_bonds.atoms[4].hydrogen_bond_partners['test'])


class HydrogenBondExceptionsFiFoTestCase(HydrogenBondFiFoTestCase, BlueprintExceptionsTestCase):
    def setUp(self) -> None:
        from yeti.get_features.hydrogen_bonds import HydrogenBondsException

        super(HydrogenBondExceptionsFiFoTestCase, self).setUp()

        self.exception = HydrogenBondsException


class TestHydrogenBondRepresentationMethodExceptions(HydrogenBondExceptionsFiFoTestCase):
    def test_get_hydrogen_bonds_in_frame_triplets_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds.__get_hydrogen_bonds_in_frame__(triplets=list(self.triplets), frame=0)

        desired_msg = self.create_data_type_exception_messages(parameter_name='triplets', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_hydrogen_bonds_in_frame_wrong_data_type(self):
        with self.assertRaises(self.exception) as context:
            self.hydrogen_bonds.__get_hydrogen_bonds_in_frame__(triplets=self.triplets, frame=0.)

        desired_msg = self.create_data_type_exception_messages(parameter_name='frame', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
