import gc
from types import GeneratorType

import mdtraj as md
import numpy as np

from yeti.dictionaries.molecules.biomolecules import RNA as RNADict
from yeti.dictionaries.molecules.solvents import Water as WaterDict
from yeti.systems.building_blocks import Atom, Residue
from yeti.systems.molecules.nucleic_acids import RNA
from yeti.systems.molecules.solvents import Water


class System(object):
    def __init__(self, name):
        # TODO store more parameters (see nucleoSim and mdTraj)
        self.name = name

        self.trajectory_file_path = None
        self.topology_file_path = None
        self.chunk_size = None

        self.periodic = None
        self.unitcell_angles = None
        self.unitcell_vectors = None
        self.mdtraj_object = None
        self.trajectory = None

        self.number_of_frames = 0
        self.molecules = {}

    def __trajectory_load__(self):
        self.mdtraj_object = md.load(filename_or_filenames=self.trajectory_file_path, top=self.topology_file_path)
        self.trajectory = self.mdtraj_object

        self.unitcell_angles = self.trajectory.unitcell_angles
        self.unitcell_vectors = self.trajectory.unitcell_vectors
        self.number_of_frames = self.trajectory.n_frames

    def __create_generator__(self):
        del self.mdtraj_object
        self.mdtraj_object = md.iterload(filename=self.trajectory_file_path, top=self.topology_file_path,
                                         chunk=self.chunk_size)

    def __iter_trajectory_load__(self):
        # TODO: find better generator work araound
        # need to recreate generator object again after usage
        self.__create_generator__()

        for index, self.trajectory in enumerate(self.mdtraj_object):
            if index == 0:
                self.number_of_frames = 0
                self.unitcell_angles = self.trajectory.unitcell_angles
                self.unitcell_vectors = self.trajectory.unitcell_vectors

                if self.unitcell_angles is None or self.unitcell_vectors is None:
                    break
            else:
                self.unitcell_angles = np.vstack([self.unitcell_angles, self.trajectory.unitcell_angles])
                self.unitcell_vectors = np.vstack([self.unitcell_vectors, self.trajectory.unitcell_vectors])

            self.number_of_frames += self.trajectory.n_frames

        del self.trajectory
        self.trajectory = None

    def load_trajectory(self, trajectory_file_path, topology_file_path, chunk_size=None):
        # TODO: Test for right data types
        self.trajectory_file_path = trajectory_file_path
        self.topology_file_path = topology_file_path
        self.chunk_size = chunk_size

        if chunk_size is None:
            self.__trajectory_load__()
        else:
            self.__iter_trajectory_load__()

        if self.unitcell_angles is not None and self.unitcell_vectors is not None:
            self.periodic = True
        else:
            self.periodic = False

    def __create_atom__(self, reference_atom, shifting_index):
        atom_index = reference_atom.index
        return Atom(subsystem_index=atom_index - shifting_index, structure_file_index=atom_index,
                    name=reference_atom.name, xyz_trajectory=self.trajectory.xyz[:, atom_index, :])

    def __create_residue__(self, reference_residue, subsystem_index, atom_shifting_index):
        residue_index = reference_residue.index

        actual_residue = Residue(subsystem_index=subsystem_index, structure_file_index=residue_index,
                                 name=reference_residue.name)

        for reference_atom in reference_residue.atoms:
            atom_index = reference_atom.index
            actual_atom = Atom(subsystem_index=atom_index - atom_shifting_index, structure_file_index=atom_index,
                               name=reference_atom.name, xyz_trajectory=self.trajectory.xyz[:, atom_index, :])
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

    def __select_residues_simple__(self, residue_ids):
        residues = []
        residue_lengths = []

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

    def __select_residues_iter__(self, residue_ids):
        self.__create_generator__()

        for index, chunk in enumerate(self.mdtraj_object):
            self.trajectory = chunk
            if index == 0:
                residues = self.__select_residues_simple__(residue_ids=residue_ids)
            else:
                for residue in residues:
                    for atom in residue.atoms:
                        atom.xyz_trajectory = np.vstack(
                            [atom.xyz_trajectory, self.trajectory.xyz[:, atom.structure_file_index, :]])
                pass

        del self.trajectory
        self.trajectory = None
        return residues

    def __select_residues__(self, residue_ids):
        if type(self.mdtraj_object) is GeneratorType:
            return self.__select_residues_iter__(residue_ids=residue_ids)
        else:
            return self.__select_residues_simple__(residue_ids=residue_ids)

    def select_rna(self, residue_ids, name):
        # TODO: add multi thread
        rna_dict = RNADict()
        residues = self.__select_residues__(residue_ids=residue_ids)

        for residue_id, residue in enumerate(residues):
            if not 'P' in residue.sequence and residue_id == 0:
                backbone_bonds_dictionary = rna_dict.backbone_bonds_dictionary['p_capped']
                backbone_acceptor_dictionary = rna_dict.acceptors_dictionary['backbone_p_capped']
                self.__create_inner_covalent_bonds__(residue=residue,
                                                     bond_dict=rna_dict.termini_bonds_dictionary['p_capped'])
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
                                                        unit_cell_vectors=self.unitcell_vectors))

    def select_water(self, residue_ids):
        # TODO: add multi thread
        water_dict = WaterDict()
        residues = self.__select_residues__(residue_ids=residue_ids)

        for residue_id, residue in enumerate(residues):
            self.__create_inner_covalent_bonds__(residue=residue, bond_dict=water_dict.covalent_bonds)
            self.__update_electronic_state__(residue=residue, electron_dict=water_dict.acceptors_dictionary,
                                             is_acceptor=True)
            self.__update_electronic_state__(residue=residue, electron_dict=water_dict.donors_dictionary, is_donor=True)

            name = 'H2O_{water_id:06d}'.format(water_id=residue_id)

            self.molecules[name] = Water(residue=residue, molecule_name=name, periodic=self.periodic,
                                         box_information=dict(unit_cell_angles=self.unitcell_angles,
                                                              unit_cell_vectors=self.unitcell_vectors))

    def purge_molecule(self, name):
        self.molecules.pop(name)
        gc.collect()

    def purge_all_water(self):
        # TODO: add multi thread

        keys_for_deletion = [key for key in self.molecules.keys() if
                             'H2O' in key and type(self.molecules[key]) is Water]

        for key in keys_for_deletion:
            self.purge_molecule(name=key)
