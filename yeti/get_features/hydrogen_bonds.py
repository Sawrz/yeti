import itertools
from multiprocessing.pool import ThreadPool
from threading import Thread

import numpy as np

from yeti.get_features.angles import Angle
from yeti.get_features.distances import Distance
from yeti.systems.building_blocks import EnsureDataTypes


class TripletException(Exception):
    pass


class HydrogenBondsException(Exception):
    pass


class Triplet(object):
    def __init__(self, donor_atom, acceptor, periodic, unit_cell_angles, unit_cell_vectors):
        self.ensure_data_type = EnsureDataTypes(exception_class=TripletException)
        self.ensure_data_type.ensure_atom(parameter=donor_atom, parameter_name='donor_atom')
        self.ensure_data_type.ensure_atom(parameter=acceptor, parameter_name='acceptor')
        self.ensure_data_type.ensure_boolean(parameter=periodic, parameter_name='periodic')
        self.ensure_data_type.ensure_numpy_array(parameter=unit_cell_angles, parameter_name='unit_cell_angles',
                                                 shape=(None, 3), desired_dtype=np.float32)
        self.ensure_data_type.ensure_numpy_array(parameter=unit_cell_vectors, parameter_name='unit_cell_vectors',
                                                 shape=(None, 3, 3), desired_dtype=np.float32)

        self.periodic = periodic
        self.unit_cell_angles = unit_cell_angles
        self.unit_cell_vectors = unit_cell_vectors

        if len(donor_atom.covalent_bond_partners) > 1:
            msg = 'Donor atom has more than one covalent bond. That violates the assumption of this method. ' \
                  'Please contact the developer.'
            raise TripletException(msg)

        self.donor_atom = donor_atom
        self.donor = self.donor_atom.covalent_bond_partners[0]
        self.acceptor = acceptor

        self.triplet = (self.donor, self.donor_atom, self.acceptor)
        self.mask = None

    def create_mask(self, distance_cutoff, angle_cutoff):
        self.ensure_data_type.ensure_float(parameter=distance_cutoff, parameter_name='distance_cutoff')
        self.ensure_data_type.ensure_float(parameter=angle_cutoff, parameter_name='angle_cutoff')

        # calculate distances
        dist = Distance(unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)
        distances = dist.calculate((self.donor, self.acceptor), opt=True, periodic=self.periodic)

        angle = Angle(unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)
        angles = angle.calculate(self.triplet, opt=False, periodic=self.periodic, legacy=False)

        # angles should not be negative but for safety and mathematical correctness
        self.mask = np.logical_and(distances < distance_cutoff, np.pi - np.abs(angles) < angle_cutoff)


class HydrogenBonds(object):
    def __init__(self, atoms, periodic, unit_cell_angles, unit_cell_vectors, system_name, number_of_frames):
        self.ensure_data_type = EnsureDataTypes(exception_class=HydrogenBondsException)

        self.ensure_data_type.ensure_tuple(parameter=atoms, parameter_name='atoms')
        self.ensure_data_type.ensure_boolean(parameter=periodic, parameter_name='periodic')
        self.ensure_data_type.ensure_numpy_array(parameter=unit_cell_angles, parameter_name='unit_cell_angles',
                                                 shape=(None, 3), desired_dtype=np.float32)
        self.ensure_data_type.ensure_numpy_array(parameter=unit_cell_vectors, parameter_name='unit_cell_vectors',
                                                 shape=(None, 3, 3), desired_dtype=np.float32)
        self.ensure_data_type.ensure_string(parameter=system_name, parameter_name='system_name')
        self.ensure_data_type.ensure_integer(parameter=number_of_frames, parameter_name='number_of_frames')

        self.periodic = periodic
        self.unit_cell_angles = unit_cell_angles
        self.unit_cell_vectors = unit_cell_vectors

        self.atoms = atoms
        self._system_name = system_name
        self.number_of_frames = number_of_frames

        self.donor_atoms = []
        self.acceptors = []

        for atom in atoms:
            if atom.is_donor_atom:
                self.donor_atoms.append(atom)
                atom.add_system(system_name=self._system_name)
            elif atom.is_acceptor:
                self.acceptors.append(atom)
                atom.add_system(system_name=self._system_name)

        self.donor_atoms = tuple(self.donor_atoms)
        self.acceptors = tuple(self.acceptors)

    def __build_triplet__(self, donor_atom, acceptor):
        return Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=self.periodic,
                       unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)

    def __build_triplets__(self, distance_cutoff, angle_cutoff):
        self.ensure_data_type.ensure_float(parameter=distance_cutoff, parameter_name='distance_cutoff')
        self.ensure_data_type.ensure_float(parameter=angle_cutoff, parameter_name='angle_cutoff')

        # initialize triplets
        donor_acceptor_combinations = itertools.product(*[self.donor_atoms, self.acceptors])

        pool = ThreadPool()
        triplets = pool.starmap(self.__build_triplet__, donor_acceptor_combinations)
        pool.close()

        threads = []
        for triplet in triplets:
            process = Thread(target=triplet.create_mask,
                             kwargs=dict(distance_cutoff=distance_cutoff, angle_cutoff=angle_cutoff))
            process.start()
            threads.append(process)

        for process in threads:
            process.join()

        return tuple(triplets)

    def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
        pass

    def __get_hydrogen_bonds__(self, triplets):
        # TODO: Create Multi Process Unit Test

        self.ensure_data_type.ensure_tuple(parameter=triplets, parameter_name='triplets')

        threads = []
        for frame in range(self.number_of_frames):
            process = Thread(target=self.__get_hydrogen_bonds_in_frame__, kwargs=dict(triplets=triplets, frame=frame))
            process.start()
            threads.append(process)

        for process in threads:
            process.join()

    def calculate_hydrogen_bonds(self, distance_cutoff, angle_cutoff):
        for donor_atom in self.donor_atoms:
            donor_atom.purge_hydrogen_bond_partner_history(system_name=self._system_name)
        for acceptor_atom in self.acceptors:
            acceptor_atom.purge_hydrogen_bond_partner_history(system_name=self._system_name)

        print('Building Triplets...')
        triplets = self.__build_triplets__(distance_cutoff=distance_cutoff, angle_cutoff=angle_cutoff)
        print('Getting Hydrogen Bonds...')
        self.__get_hydrogen_bonds__(triplets=triplets)

    def __get_hydrogen_bond_matrix_in_frame__(self, index_dictionary, frame):
        self.ensure_data_type.ensure_dict(parameter=index_dictionary, parameter_name='index_dictionary')
        self.ensure_data_type.ensure_integer(parameter=frame, parameter_name='frame')

        hydrogen_bond_matrix = np.zeros((len(self.atoms), len(self.atoms)))

        for acceptor in self.acceptors:
            hydrogen_bond_partners = acceptor.hydrogen_bond_partners[self._system_name][frame]

            if len(hydrogen_bond_partners) > 0:
                acceptor_index = index_dictionary[acceptor.structure_file_index]

                for hydrogen_bond_partner in hydrogen_bond_partners:
                    hydrogen_bond_partner_index = index_dictionary[hydrogen_bond_partner.structure_file_index]

                    hydrogen_bond_matrix[acceptor_index][hydrogen_bond_partner_index] += 1
                    hydrogen_bond_matrix[hydrogen_bond_partner_index][acceptor_index] += 1

            if np.any(hydrogen_bond_matrix > 1):
                raise HydrogenBondsException(
                    'An entry in the adjacent matrix is bigger than one. Please check your hydrogen bonds.')

        return hydrogen_bond_matrix

    def get_hydrogen_bond_matrix(self):
        # TODO: make use multi thread

        index_dictionary = {}
        for index, atom in enumerate(self.atoms):
            index_dictionary[atom.structure_file_index] = index

        matrices = []

        for frame in range(self.number_of_frames):
            matrix = self.__get_hydrogen_bond_matrix_in_frame__(index_dictionary=index_dictionary, frame=frame)
            matrices.append(matrix)

        return np.array(matrices)

    def __get_number_hydrogen_bonds_for_frame__(self, frame):
        self.ensure_data_type.ensure_integer(parameter=frame, parameter_name='frame')

        hydrogen_bonds = 0

        for acceptor in self.acceptors:
            acceptor_hydrogen_bonds = len(acceptor.hydrogen_bond_partners[self._system_name][frame])
            hydrogen_bonds += acceptor_hydrogen_bonds

        return hydrogen_bonds

    def get_frame_wise_hydrogen_bonds(self):
        # TODO: make use multi thread

        number_of_hydrogen_bonds = []

        for frame in range(self.number_of_frames):
            number_of_hydrogen_bonds.append(self.__get_number_hydrogen_bonds_for_frame__(frame=frame))

        return np.array(number_of_hydrogen_bonds)


class HydrogenBondsFirstComesFirstServes(HydrogenBonds):
    def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
        self.ensure_data_type.ensure_tuple(parameter=triplets, parameter_name='triplets')
        self.ensure_data_type.ensure_integer(parameter=frame, parameter_name='frame')

        for triplet in triplets:
            if triplet.mask[frame] == 0:
                continue

            donor_slot_free = len(
                triplet.donor_atom.hydrogen_bond_partners[self._system_name][frame]) < triplet.donor_atom.donor_slots
            acceptor_slot_free = len(
                triplet.acceptor.hydrogen_bond_partners[self._system_name][frame]) < triplet.acceptor.acceptor_slots

            if not donor_slot_free or not acceptor_slot_free:
                continue

            triplet.donor_atom.hydrogen_bond_partners[self._system_name][frame].append(triplet.acceptor)
            triplet.acceptor.hydrogen_bond_partners[self._system_name][frame].append(triplet.donor_atom)
