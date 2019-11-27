import mdtraj as md
import numpy as np

from yeti.dictionaries.molecules.biomolecules import RNA as RNADict
from yeti.systems.building_blocks import Atom, Residue
from yeti.systems.molecules.nucleic_acids import RNA


class System(object):
    def __init__(self, trajectory_file_path, topology_file_path, periodic):
        self.trajectory_file_path = trajectory_file_path
        self.topology_file_path = topology_file_path
        self.periodic = periodic

        # TODO store more parameters (see nucleoSim and mdTraj)

        self.trajectory = md.load(filename_or_filenames=self.trajectory_file_path, top=self.topology_file_path)

        self.unitcell_angles = self.trajectory.unitcell_angles
        self.unitcell_vectors = self.trajectory.unitcell_vectors

        self.number_of_frames = self.trajectory.n_frames
        self.molecules = {}

    def __create_atom__(self, reference_atom, shifting_index):
        atom_index = reference_atom.index
        return Atom(subsystem_index=atom_index - shifting_index, structure_file_index=atom_index,
                    name=reference_atom.name, xyz_trajectory=self.trajectory.xyz[:, atom_index, :])

    def __create_residue__(self, reference_residue, subsystem_index, atom_shifting_index):
        residue_index = reference_residue.index

        actual_residue = Residue(subsystem_index=subsystem_index, structure_file_index=residue_index,
                                 name=reference_residue.name)

        for reference_atom in reference_residue.atoms:
            actual_atom = self.__create_atom__(reference_atom=reference_atom, shifting_index=atom_shifting_index)
            actual_atom.set_residue(residue=actual_residue)

            actual_residue.add_atom(atom=actual_atom)

        actual_residue.finalize()

        return actual_residue

    @staticmethod
    def __create_inner_covalent_bonds__(residue, bond_dict):
        atoms = residue.atoms
        sequence = np.array(residue.sequence)

        for bond in bond_dict:
            atom_01_id = np.where(sequence == bond[0])[0]
            atom_02_id = np.where(sequence == bond[1])[0]

            # TODO: add right Exception type
            if len(atom_01_id) == 0 or len(atom_02_id) == 0:
                raise Exception('Atom not found! Something went wrong.')
            elif len(atom_01_id) != 1 or len(atom_02_id) != 1:
                raise Exception('Indistinguishable atom names. Something went wrong.')

            atom_01 = atoms[atom_01_id[0]]
            atom_02 = atoms[atom_02_id[0]]

            atom_01.add_covalent_bond(atom=atom_02)

    @staticmethod
    def __create_inter_covalent_bonds__(residue_01, residue_02, molecule_dict):
        pair = molecule_dict.bonds_between_residues

        sequence_01 = np.array(residue_01.sequence)
        sequence_02 = np.array(residue_02.sequence)

        atom_01_id = np.where(sequence_01 == pair[0])[0]
        atom_02_id = np.where(sequence_02 == pair[1])[0]

        # TODO: add right Exception type
        if len(atom_01_id) == 0 or len(atom_02_id) == 0:
            raise Exception('Atom not found! Something went wrong.')
        elif len(atom_01_id) != 1 or len(atom_02_id) != 1:
            raise Exception('Indistinguishable atom names. Something went wrong.')

        atom_01 = residue_01.atoms[atom_01_id[0]]
        atom_02 = residue_02.atoms[atom_02_id[0]]

        atom_01.add_covalent_bond(atom=atom_02)

    def __update_electronic_state__(self, residue, electron_dict, is_donor=False, is_acceptor=False):
        atoms = residue.atoms
        sequence = np.array(residue.sequence)

        for atom_name, available_slots in zip(electron_dict.keys(), electron_dict.values()):
            atom_id = np.where(sequence == atom_name)[0]

            # TODO: add right Exception type
            if len(atom_id) == 0:
                raise Exception('Atom not found! Something went wrong.')
            elif len(atom_id) != 1:
                raise Exception('Indistinguishable atom names. Something went wrong.')

            atom_id = atom_id[0]

            if is_donor and not is_acceptor:
                atoms[atom_id].update_donor_state(is_donor_atom=True, donor_slots=available_slots)
            elif is_acceptor and not is_donor:
                atoms[atom_id].update_acceptor_state(is_acceptor=True, acceptor_slots=available_slots)
            else:
                # TODO: add right Exception type
                raise Exception('Either, is_donor or is_acceptor must be True.')

    def __select_residues__(self, residue_ids, is_range):
        residues = []
        residue_lengths = []

        if is_range:
            residue_ids = list(np.arange(residue_ids[0], residue_ids[1] + 1))

        for subsystem_index, residue_id in enumerate(residue_ids):
            if subsystem_index == 0:
                atom_shifting_index = int(self.trajectory.topology.residue(residue_ids[0])._atoms[0].index)
            else:
                atom_shifting_index = int(
                    self.trajectory.topology.residue(residue_ids[subsystem_index])._atoms[0].index - sum(
                        residue_lengths))

            residue = self.__create_residue__(reference_residue=self.trajectory.topology.residue(residue_id),
                                              subsystem_index=subsystem_index,
                                              atom_shifting_index=atom_shifting_index)
            residue_lengths.append(len(residue.atoms))
            residues.append(residue)

        return tuple(residues)

    def select_rna(self, residue_ids, name, distance_cutoff, angle_cutoff, is_range=True):
        rna_dict = RNADict()
        residues = self.__select_residues__(residue_ids=residue_ids, is_range=is_range)

        for residue_id, residue in enumerate(residues):
            if not 'P' in residue.sequence and residue_id == 0:
                backbone_bonds_dictionary = rna_dict.backbone_bonds_dictionary['p_capped']
                backbone_acceptor_dictionary = rna_dict.acceptors_dictionary['backbone_p_capped']
                self.__create_inner_covalent_bonds__(residue=residue,
                                                     bond_dict=rna_dict.termini_bonds_dictionary['p_capping'])
            else:
                backbone_bonds_dictionary = rna_dict.backbone_bonds_dictionary['residual']
                backbone_acceptor_dictionary = rna_dict.acceptors_dictionary['backbone']

            if residue_id == len(residues) - 1:
                backbone_bonds_dictionary = rna_dict.backbone_bonds_dictionary['residual']
                self.__create_inner_covalent_bonds__(residue=residue,
                                                     bond_dict=rna_dict.termini_bonds_dictionary['last_residue'])

            residue_name = rna_dict.abbreviation_dictionary[residue.name]

            self.__create_inner_covalent_bonds__(residue=residue, bond_dict=backbone_bonds_dictionary)
            self.__create_inner_covalent_bonds__(residue=residue,
                                                 bond_dict=rna_dict.side_chain_bonds_dictionary[residue_name])

            self.__update_electronic_state__(residue=residue, is_acceptor=True,
                                             electron_dict=backbone_acceptor_dictionary)
            self.__update_electronic_state__(residue=residue, is_donor=True,
                                             electron_dict=rna_dict.donors_dictionary['backbone'])
            self.__update_electronic_state__(residue=residue, is_acceptor=True,
                                             electron_dict=rna_dict.acceptors_dictionary[residue_name])
            self.__update_electronic_state__(residue=residue, is_donor=True,
                                             electron_dict=rna_dict.donors_dictionary[residue_name])

            if residue_id < len(residues) - 1:
                self.__create_inter_covalent_bonds__(residue_01=residue, residue_02=residues[residue_id + 1],
                                                     molecule_dict=rna_dict)

        self.molecules[name] = RNA(residues=residues, molecule_name=name, periodic=self.periodic,
                                   box_information=dict(unit_cell_angles=self.unitcell_angles,
                                                        unit_cell_vectors=self.unitcell_vectors),
                                   simulation_information=dict(number_of_frames=self.number_of_frames),
                                   hydrogen_bond_information=dict(distance_cutoff=distance_cutoff,
                                                                  angle_cutoff=angle_cutoff))
