import unittest

from test_utils.test_utils import build_unit_cell_angles_and_vectors, create_data_type_exception_messages, \
    build_multi_atom_triplets


class HydrogenBondsFirstComesFirstServesTest(unittest.TestCase):
    def test_get_hydrogen_bonds_in_frame_too_many_donors(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBondsFirstComesFirstServes

        atoms = list(build_multi_atom_triplets(amount=3))
        atoms.pop(-1)
        atoms.pop(-3)
        atoms = tuple(atoms)
        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBondsFirstComesFirstServes(atoms=atoms, periodic=True,
                                                            unit_cell_angles=unit_cell_angles,
                                                            unit_cell_vectors=unit_cell_vectors,
                                                            system_name='test_system',
                                                            number_of_frames=number_of_frames)

        triplets = hydrogen_bonds.__build_triplets__(distance_cutoff=0.25, angle_cutoff=2.0)
        hydrogen_bonds.__get_hydrogen_bonds_in_frame__(triplets=triplets, frame=0)

        self.assertIsNone(hydrogen_bonds.atoms[0].hydrogen_bond_partners)
        self.assertIsNone(hydrogen_bonds.atoms[3].hydrogen_bond_partners)
        self.assertIsNone(hydrogen_bonds.atoms[5].hydrogen_bond_partners)
        self.assertListEqual([[atoms[2]], [], []], hydrogen_bonds.atoms[1].hydrogen_bond_partners['test_system'])
        self.assertListEqual([[atoms[1], atoms[4]], [], []],
                             hydrogen_bonds.atoms[2].hydrogen_bond_partners['test_system'])
        self.assertListEqual([[atoms[2]], [], []], hydrogen_bonds.atoms[4].hydrogen_bond_partners['test_system'])
        self.assertListEqual([[], [], []], hydrogen_bonds.atoms[6].hydrogen_bond_partners['test_system'])

    def test_get_hydrogen_bonds_in_frame_too_many_acceptors(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBondsFirstComesFirstServes

        atoms = list(build_multi_atom_triplets(amount=3))
        atoms.pop(-2)
        atoms.pop(-2)
        atoms.pop(-3)
        atoms.pop(-3)

        atoms = tuple(atoms)
        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBondsFirstComesFirstServes(atoms=atoms, periodic=True,
                                                            unit_cell_angles=unit_cell_angles,
                                                            unit_cell_vectors=unit_cell_vectors,
                                                            system_name='test_system',
                                                            number_of_frames=number_of_frames)

        triplets = hydrogen_bonds.__build_triplets__(distance_cutoff=0.25, angle_cutoff=2.0)
        hydrogen_bonds.__get_hydrogen_bonds_in_frame__(triplets=triplets, frame=0)

        self.assertIsNone(hydrogen_bonds.atoms[0].hydrogen_bond_partners)
        self.assertListEqual([[atoms[2]], [], []], hydrogen_bonds.atoms[1].hydrogen_bond_partners['test_system'])
        self.assertListEqual([[atoms[1]], [], []], hydrogen_bonds.atoms[2].hydrogen_bond_partners['test_system'])
        self.assertListEqual([[], [], []], hydrogen_bonds.atoms[3].hydrogen_bond_partners['test_system'])
        self.assertListEqual([[], [], []], hydrogen_bonds.atoms[4].hydrogen_bond_partners['test_system'])

    def test_get_hydrogen_bonds_in_frame_triplets_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBondsFirstComesFirstServes, HydrogenBondsException

        atoms = list(build_multi_atom_triplets(amount=3))
        atoms.pop(-2)
        atoms.pop(-2)
        atoms.pop(-3)
        atoms.pop(-3)

        atoms = tuple(atoms)
        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBondsFirstComesFirstServes(atoms=atoms, periodic=True,
                                                            unit_cell_angles=unit_cell_angles,
                                                            unit_cell_vectors=unit_cell_vectors,
                                                            system_name='test_system',
                                                            number_of_frames=number_of_frames)

        triplets = list(hydrogen_bonds.__build_triplets__(distance_cutoff=0.25, angle_cutoff=2.0))

        with self.assertRaises(HydrogenBondsException) as context:
            hydrogen_bonds.__get_hydrogen_bonds_in_frame__(triplets=triplets, frame=0)

        desired_msg = create_data_type_exception_messages(parameter_name='triplets', data_type_name='tuple')
        self.assertEqual(desired_msg, str(context.exception))

    def test_get_hydrogen_bonds_in_frame_wrong_data_type(self):
        from yeti.get_features.hydrogen_bonds import HydrogenBondsFirstComesFirstServes, HydrogenBondsException

        atoms = list(build_multi_atom_triplets(amount=3))
        atoms.pop(-2)
        atoms.pop(-2)
        atoms.pop(-3)
        atoms.pop(-3)

        atoms = tuple(atoms)
        number_of_frames = 3

        unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=number_of_frames)

        hydrogen_bonds = HydrogenBondsFirstComesFirstServes(atoms=atoms, periodic=True,
                                                            unit_cell_angles=unit_cell_angles,
                                                            unit_cell_vectors=unit_cell_vectors,
                                                            system_name='test_system',
                                                            number_of_frames=number_of_frames)

        triplets = hydrogen_bonds.__build_triplets__(distance_cutoff=0.25, angle_cutoff=2.0)

        with self.assertRaises(HydrogenBondsException) as context:
            hydrogen_bonds.__get_hydrogen_bonds_in_frame__(triplets=triplets, frame=0.)

        desired_msg = create_data_type_exception_messages(parameter_name='frame', data_type_name='int')
        self.assertEqual(desired_msg, str(context.exception))


if __name__ == '__main__':
    unittest.main()
