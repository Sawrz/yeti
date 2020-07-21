import gc
import itertools
from multiprocessing.pool import ThreadPool

import numpy as np
import cupy as cp

from yeti.get_features.angles import Angle, Dihedral
from yeti.get_features.distances import Distance
from yeti.get_features.hydrogen_bonds import HydrogenBondsFirstComesFirstServes, HydrogenBondsDistanceCriterion
from yeti.systems.building_blocks import EnsureDataTypes


class MoleculeException(Exception):
    pass


class BioMoleculeException(Exception):
    pass


class TwoAtomsMolecule(object):
    def __init__(self, residues, molecule_name, periodic, box_information):
        self.ensure_data_type = EnsureDataTypes(exception_class=MoleculeException)
        self.ensure_data_type.ensure_tuple(parameter=residues, parameter_name='residues')
        self.ensure_data_type.ensure_string(parameter=molecule_name, parameter_name='molecule_name')
        self.ensure_data_type.ensure_dict(parameter=box_information, parameter_name='box_information')
        self.ensure_data_type.ensure_boolean(parameter=periodic, parameter_name='periodic')

        self.molecule_name = molecule_name
        self.residues = residues
        self.periodic = periodic

        # TODO: find a better way to transfer this information
        self.box_information = box_information

        self._dist = Distance(**self.box_information)
        self.distances = {}

    def __get_atom_key_name__(self, atom):
        self.ensure_data_type.ensure_atom(parameter=atom, parameter_name='atom')

        return '{residue_name}_{residue_id:04d}:{atom_name}_{atom_id:04d}'.format(
            atom_name=atom.name, atom_id=atom.subsystem_index, residue_name=atom.residue.name,
            residue_id=atom.residue.subsystem_index)

    def __generate_key__(self, atoms):
        self.ensure_data_type.ensure_tuple(parameter=atoms, parameter_name='atoms')

        key = ''

        for atom_number, atom in enumerate(atoms):
            key += self.__get_atom_key_name__(atom=atom)

            if atom_number < len(atoms) - 1:
                key += '-'

        return key

    def __get_atom__(self, atom_pos):
        self.ensure_data_type.ensure_tuple(parameter=atom_pos, parameter_name='atom_pos')

        residue_pos, atom_pos_in_residue = atom_pos
        return self.residues[residue_pos].atoms[atom_pos_in_residue]

    def __get_atoms__(self, atom_positions):
        self.ensure_data_type.ensure_tuple(parameter=atom_positions, parameter_name='atom_positions')

        atoms = []

        for atom_pos in atom_positions:
            atoms.append(self.__get_atom__(atom_pos=atom_pos))

        return tuple(atoms)

    def __get_atom_id__(self, name, residue_id):
        self.ensure_data_type.ensure_string(parameter=name, parameter_name='name')
        self.ensure_data_type.ensure_integer(parameter=residue_id, parameter_name='residue_id')

        if residue_id > len(self.residues) - 1:
            raise BioMoleculeException('Atom requested residue id is higher than available residues.')

        atom_index = np.where(np.array(self.residues[residue_id].sequence) == name)[0]

        if len(atom_index) == 0:
            raise BioMoleculeException('Atom does not exist in this residue.')
        elif len(atom_index) > 1:
            raise BioMoleculeException(
                'Atom names are not distinguishable. Check your naming or contact the developer.')
        else:
            return residue_id, atom_index[0]

    def __get_atom_ids__(self, atom_names, residue_ids):
        self.ensure_data_type.ensure_tuple(parameter=atom_names, parameter_name='atom_names')
        self.ensure_data_type.ensure_tuple(parameter=residue_ids, parameter_name='residue_ids')

        atom_list = []

        for atom_name, residue_id in zip(atom_names, residue_ids):
            atom_list.append(self.__get_atom_id__(name=atom_name, residue_id=residue_id))

        return tuple(atom_list)

    def _process_atom_coordinates(self, atom, names, xyz, atom_xyz, visited_atoms):
        if atom in visited_atoms:
            return None

        names[atom.subsystem_index] = f'{atom.residue.name}{atom.residue.structure_file_index}_{atom.name}'
        xyz[:, atom.subsystem_index, :] += atom_xyz.reshape(atom.xyz_trajectory.shape[0], 1, 3)[:, 0, :]
        visited_atoms.append(atom)

    def _eliminate_periodicity(self, atom, cov_atom):
        ref_xyz = atom.xyz_trajectory
        cov_xyz = cov_atom.xyz_trajectory
        displacement = ref_xyz - cov_xyz

        unit_cell_vectors = self.box_information["unit_cell_vectors"]
        upper_box_boundaries = np.sum(unit_cell_vectors, axis=1)
        geometric_center = upper_box_boundaries / 2

        exceed_upper_boundary_condition = np.where(displacement > geometric_center)
        exceed_lower_boundary_condition = np.where(displacement < -geometric_center)

        if len(exceed_upper_boundary_condition[0]) > 0:
            for frame, dimension in zip(exceed_upper_boundary_condition[0], exceed_upper_boundary_condition[1]):
                cov_xyz[frame] += unit_cell_vectors[frame, dimension]

        if len(exceed_lower_boundary_condition[0]) > 0:
            for frame, dimension in zip(exceed_lower_boundary_condition[0], exceed_lower_boundary_condition[1]):
                cov_xyz[frame] -= unit_cell_vectors[frame, dimension]

        return cov_xyz

    def get_xyz(self, eliminate_periodicity=False):
        frames = self.residues[0].atoms[0].xyz_trajectory.shape[0]
        atoms = tuple(sum([residue.atoms for residue in self.residues], ()))
        atoms_amount = len(atoms)

        names = np.empty(atoms_amount, dtype=object)
        xyz = np.zeros(shape=(frames, atoms_amount, 3), dtype=np.float)
        visited_atoms = []

        for atom_index, atom in enumerate(atoms):
            if atom in visited_atoms:
                continue

            if eliminate_periodicity:
                if atom_index == 0:
                    atom_xyz = atom.xyz_trajectory
                else:
                    atom_xyz = self._eliminate_periodicity(atoms[0], atom)

                self._process_atom_coordinates(atom=atom, names=names, xyz=xyz, atom_xyz=atom_xyz,
                                               visited_atoms=visited_atoms)

                for cov_atom in atom.covalent_bond_partners:
                    if cov_atom in visited_atoms:
                        continue

                    cov_xyz = self._eliminate_periodicity(atom, cov_atom)
                    self._process_atom_coordinates(atom=cov_atom, names=names, xyz=xyz, atom_xyz=cov_xyz,
                                                   visited_atoms=visited_atoms)
            else:
                self._process_atom_coordinates(atom=atom, names=names, xyz=xyz, atom_xyz=atom.xyz_trajectory,
                                               visited_atoms=visited_atoms)

        return xyz, names

    def _align_frames(self, xyz, xyz_aligned, reference_frame):
        # TODO: Although using a Kabsch-like algorithm implement it fully for more versatility: https://en.wikipedia.org/wiki/Kabsch_algorithm
        frames_to_align = np.delete(np.arange(xyz.shape[0]), reference_frame)

        # TODO: check if this is true for non-cubic boxes
        # Get Geometric Center of box
        geometric_center_box = np.sum(self.box_information["unit_cell_vectors"][reference_frame], axis=1) / 2
        geometric_center_molecule = np.mean(xyz_aligned[reference_frame], axis=0)

        # Shift reference frame
        xyz_aligned[reference_frame] += geometric_center_box - geometric_center_molecule
        geometric_center_molecule = np.mean(xyz_aligned[reference_frame], axis=0).reshape(1, 3)

        # Iterate to find best possible alignment
        for i in range(1000):
            # Get Translational Vector and translate
            translation_vector = geometric_center_molecule - np.mean(xyz[frames_to_align], axis=1)
            xyz[frames_to_align] += translation_vector.reshape(frames_to_align.shape[0], 1, 3)

            # Get Rotation Matrix and rotate
            u, s, vh = np.linalg.svd(np.einsum('ijk, ijl->ikl', xyz[frames_to_align],
                                               xyz_aligned[reference_frame].reshape(1, xyz_aligned.shape[1], 3)))
            rotation_matrix = np.einsum('ijk, ikl->ijl', u, vh)

            xyz[frames_to_align] = np.einsum('ijk, ikl->ijl', xyz[frames_to_align], rotation_matrix)

        # Get Translational Vector and translate
        translation_vector = geometric_center_molecule - np.mean(xyz[frames_to_align], axis=1)
        xyz[frames_to_align] += translation_vector.reshape(frames_to_align.shape[0], 1, 3)

        xyz_aligned[frames_to_align] += xyz[frames_to_align]

    # TODO: Improve Performance for GPU and CPU
    def get_aligned_xyz(self, reference_frame, periodic=True):
        # TODO: think about GPU support

        if not np.allclose(self.box_information["unit_cell_angles"], 90):
            raise MoleculeException("Box is not orthogonal. This method works on orthogonal boxes only.")

        xyz = self.get_xyz(eliminate_periodicity=periodic)[0]
        xyz_aligned = np.zeros_like(xyz)

        # Set Reference Frame
        xyz_aligned[reference_frame] = xyz[reference_frame]

        # Align
        self._align_frames(xyz=xyz, xyz_aligned=xyz_aligned, reference_frame=reference_frame)

        return np.round(xyz_aligned, decimals=6)

    def get_distance(self, atom_01_pos, atom_02_pos, store_result=True, opt=True):
        # TODO: ensure it's a tuple of integers
        self.ensure_data_type.ensure_tuple(parameter=atom_01_pos, parameter_name='atom_01_pos')
        self.ensure_data_type.ensure_tuple(parameter=atom_02_pos, parameter_name='atom_02_pos')
        self.ensure_data_type.ensure_boolean(parameter=store_result, parameter_name='store_result')
        self.ensure_data_type.ensure_boolean(parameter=opt, parameter_name='opt')

        atoms = self.__get_atoms__(atom_positions=(atom_01_pos, atom_02_pos))
        distances = self._dist.calculate(atoms=atoms, opt=opt, periodic=self.periodic)

        if store_result:
            key = self.__generate_key__(atoms=atoms)
            self.distances[key] = distances
        else:
            return distances


class ThreeAtomsMolecule(TwoAtomsMolecule):
    def __init__(self, *args, **kwargs):
        super(ThreeAtomsMolecule, self).__init__(*args, **kwargs)

        self._angle = Angle(**self.box_information)
        self.angles = {}

    def get_angle(self, atom_01_pos, atom_02_pos, atom_03_pos, store_result=True, opt=True):
        # TODO: ensure it's a tuple of integers
        self.ensure_data_type.ensure_tuple(parameter=atom_01_pos, parameter_name='atom_01_pos')
        self.ensure_data_type.ensure_tuple(parameter=atom_02_pos, parameter_name='atom_02_pos')
        self.ensure_data_type.ensure_tuple(parameter=atom_03_pos, parameter_name='atom_03_pos')
        self.ensure_data_type.ensure_boolean(parameter=store_result, parameter_name='store_result')
        self.ensure_data_type.ensure_boolean(parameter=opt, parameter_name='opt')

        atoms = self.__get_atoms__(atom_positions=(atom_01_pos, atom_02_pos, atom_03_pos))
        angles = self._angle.calculate(atoms=atoms, opt=opt, periodic=self.periodic)

        if store_result:
            key = self.__generate_key__(atoms=atoms)
            self.angles[key] = angles
        else:
            return angles


# TODO: find a more representative name
class FourAtomsPlusMolecule(ThreeAtomsMolecule):
    def __init__(self, *args, **kwargs):
        super(FourAtomsPlusMolecule, self).__init__(*args, **kwargs)

        self._dih = Dihedral(**self.box_information)
        self.dihedral_angles = {}

        self.angle_cutoff = None
        self.distance_cutoff = None
        self._hbonds = None

    def get_dihedral(self, atom_01_pos, atom_02_pos, atom_03_pos, atom_04_pos, store_result=True, opt=True):
        # TODO: ensure it's a tuple of integers
        self.ensure_data_type.ensure_tuple(parameter=atom_01_pos, parameter_name='atom_01_pos')
        self.ensure_data_type.ensure_tuple(parameter=atom_02_pos, parameter_name='atom_02_pos')
        self.ensure_data_type.ensure_tuple(parameter=atom_03_pos, parameter_name='atom_03_pos')
        self.ensure_data_type.ensure_tuple(parameter=atom_04_pos, parameter_name='atom_04_pos')
        self.ensure_data_type.ensure_boolean(parameter=store_result, parameter_name='store_result')
        self.ensure_data_type.ensure_boolean(parameter=opt, parameter_name='opt')

        atoms = self.__get_atoms__(atom_positions=(atom_01_pos, atom_02_pos, atom_03_pos, atom_04_pos))
        dihedral_angles = self._dih.calculate(atoms=atoms, opt=opt, periodic=self.periodic)

        if store_result:
            key = self.__generate_key__(atoms=atoms)
            self.dihedral_angles[key] = dihedral_angles
        else:
            return dihedral_angles

    def calculate_hydrogen_bonds(self, distance_cutoff, angle_cutoff, calc_method='distance'):
        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = angle_cutoff

        # estimate the number of frames from arbitrary atom since data from same simulation
        number_of_frames = self.residues[0].atoms[0].xyz_trajectory.shape[0]

        # TODO: search if there is a more efficient way
        atoms = tuple(itertools.chain.from_iterable(residue.atoms for residue in self.residues))

        # TODO: Think of passing residues instead of atoms (reason againt it: inter system hbonds, but can extract atoms from residues)
        if calc_method == 'distance':
            self._hbonds = HydrogenBondsDistanceCriterion(atoms=atoms, periodic=self.periodic,
                                                          number_of_frames=number_of_frames,
                                                          system_name=self.molecule_name, **self.box_information)
        elif calc_method == 'fifo':
            self._hbonds = HydrogenBondsFirstComesFirstServes(atoms=atoms, periodic=self.periodic,
                                                              number_of_frames=number_of_frames,
                                                              system_name=self.molecule_name, **self.box_information)
        else:
            raise MoleculeException('Please choose a valid calculation method!')

        self._hbonds.calculate_hydrogen_bonds(distance_cutoff=self.distance_cutoff, angle_cutoff=self.angle_cutoff)

    def purge_hydrogen_bonds(self):
        self.angle_cutoff = None
        self.distance_cutoff = None

        # TODO: search if there is a more efficient way
        for atom in itertools.chain.from_iterable(residue.atoms for residue in self.residues):
            if atom.is_donor_atom or atom.is_acceptor:
                atom.purge_hydrogen_bond_partner_history(system_name=self.molecule_name)

    def recalculate_hydrogen_bonds(self, distance_cutoff, angle_cutoff):
        self.purge_hydrogen_bonds()

        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = angle_cutoff

        self._hbonds.calculate_hydrogen_bonds(distance_cutoff=self.distance_cutoff, angle_cutoff=self.angle_cutoff)

    def get_hydrogen_bonds(self, representation_style):
        self.ensure_data_type.ensure_string(parameter=representation_style, parameter_name='representation_style')

        if self.distance_cutoff is None or self.angle_cutoff is None:
            raise MoleculeException('You need to calculate the hydrogen bonds first!')

        if representation_style == 'frame_wise':
            return self._hbonds.get_frame_wise_hydrogen_bonds()
        elif representation_style == 'matrix':
            return self._hbonds.get_hydrogen_bond_matrix()
        else:
            raise MoleculeException('Unknown representation style!')


# TODO: really necessary?
class BioMolecule(FourAtomsPlusMolecule):
    def __init__(self, *args, **kwargs):
        super(BioMolecule, self).__init__(*args, **kwargs)
        self.ensure_data_type.exception_class = MoleculeException
        self.dictionary = None
