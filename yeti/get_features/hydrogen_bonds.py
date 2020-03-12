import itertools
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from threading import Thread

import numpy as np

from yeti.get_features.angles import Angle
from yeti.get_features.distances import Distance
from yeti.systems.building_blocks import EnsureDataTypes


# TODO: clean up

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
        self.mask_frame = 0
        self.mask = None

    def create_mask(self, distance_cutoff, angle_cutoff):
        self.ensure_data_type.ensure_float(parameter=distance_cutoff, parameter_name='distance_cutoff')
        self.ensure_data_type.ensure_float(parameter=angle_cutoff, parameter_name='angle_cutoff')

        # calculate distances
        dist = Distance(unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)
        distances = dist.calculate((self.donor, self.acceptor), opt=True, periodic=self.periodic)

        angle = Angle(unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)
        angles = angle.calculate(self.triplet, opt=True, periodic=self.periodic, legacy=True)

        if np.any(np.isnan(angles)):
            del angles
            angles = angle.calculate(self.triplet, opt=False, periodic=self.periodic, legacy=False)

        # angles should not be negative but for safety and mathematical correctness
        self.mask = np.logical_and(distances < distance_cutoff, np.pi - np.abs(angles) < angle_cutoff)


class HydrogenBonds(object):
    def __init__(self, atoms, periodic, unit_cell_angles, unit_cell_vectors, system_name, number_of_frames,
                 core_units=None):
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

        self.current_cutoff_distance = None
        self.current_cutoff_angle = None

        if core_units is not None and core_units > cpu_count():
            # TODO: proper excpetion type
            # TODO: Test data type of core units
            raise Exception('More cores assigned than exist.')
        elif core_units is None:
            self.core_units = cpu_count()
        else:
            self.core_units = core_units

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

    # def __build_triplet__(self, donor_atom, acceptor):
    #    return Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=self.periodic,
    #                   unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)

    def __build_triplet__(self, donor_atom, acceptor):
        # return Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=self.periodic,
        #               unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)

        triplet = Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=self.periodic,
                          unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)
        triplet.create_mask(distance_cutoff=self.distance_cutoff, angle_cutoff=self.angle_cutoff)

        return triplet

    def __build_triplets__(self, distance_cutoff, angle_cutoff):
        self.ensure_data_type.ensure_float(parameter=distance_cutoff, parameter_name='distance_cutoff')
        self.ensure_data_type.ensure_float(parameter=angle_cutoff, parameter_name='angle_cutoff')

        # initialize triplets
        donor_acceptor_combinations = itertools.product(*[self.donor_atoms, self.acceptors])

        self.angle_cutoff = angle_cutoff
        self.distance_cutoff = distance_cutoff

        pool = ThreadPool(processes=self.core_units)
        triplets = pool.starmap(self.__build_triplet__, donor_acceptor_combinations)
        # triplets = triplets.get()
        pool.close()

        return tuple(triplets)

    def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
        pass

    def __execute_frame_queue__(self, queue, triplets):
        frame = queue.get()
        self.__get_hydrogen_bonds_in_frame__(triplets=triplets, frame=frame)

        queue.task_done()

    def __get_hydrogen_bonds__(self, triplets):
        # TODO: Create Multi Process Unit Test
        # TODO: add multi-threading support

        self.ensure_data_type.ensure_tuple(parameter=triplets, parameter_name='triplets')

        # queue = mp.JoinableQueue(maxsize=self.core_units)
        # for i in range(self.core_units):
        #    worker = Thread(target=self.__execute_frame_queue__, kwargs=dict(queue=queue, triplets=triplets))
        #    worker.setDaemon(True)
        #    worker.start()

        for frame in range(self.number_of_frames):
            # queue.put(frame)
            self.__get_hydrogen_bonds_in_frame__(triplets=triplets, frame=frame)

        # queue.join()

        # threads = []
        # for frame in range(self.number_of_frames):
        #    process = Thread(target=self.__get_hydrogen_bonds_in_frame__, kwargs=dict(triplets=triplets, frame=frame))
        #    process.start()
        #    threads.append(process)

        # for process in threads:
        #    process.join()

    def calculate_hydrogen_bonds(self, distance_cutoff, angle_cutoff):
        threads = []
        for donor_atom in self.donor_atoms:
            process = Thread(target=donor_atom.purge_hydrogen_bond_partner_history,
                             kwargs=dict(system_name=self._system_name))
            process.start()
            threads.append(process)
        for acceptor_atom in self.acceptors:
            process = Thread(target=acceptor_atom.purge_hydrogen_bond_partner_history,
                             kwargs=dict(system_name=self._system_name))
            process.start()
            threads.append(process)

        for process in threads:
            process.join()

        print('Building Triplets...')
        triplets = self.__build_triplets__(distance_cutoff=distance_cutoff, angle_cutoff=angle_cutoff)
        print('Getting Hydrogen Bonds...')
        self.__get_hydrogen_bonds__(triplets=triplets)

    def __get_hydrogen_bond_matrix_in_frame__(self, index_dictionary_donor_atoms, index_dictionary_acceptors, frame):
        # REMARK: the matrix in frame methods supports onlt atoms which have a max 256 donor or acceptor slots.
        # From a scientific point of view, that should be more than enough!

        # TODO: add type unit tests
        self.ensure_data_type.ensure_dict(parameter=index_dictionary_donor_atoms,
                                          parameter_name='index_dictionary_donor_atoms')
        self.ensure_data_type.ensure_dict(parameter=index_dictionary_acceptors,
                                          parameter_name='index_dictionary_acceptors')
        self.ensure_data_type.ensure_integer(parameter=frame, parameter_name='frame')

        hydrogen_bond_matrix = np.zeros(
            (len(index_dictionary_acceptors.keys()), len(index_dictionary_donor_atoms.keys())), dtype=np.int8)

        for acceptor in self.acceptors:
            hydrogen_bond_partners = acceptor.hydrogen_bond_partners[self._system_name][frame]

            if len(hydrogen_bond_partners) > 0:
                acceptor_index = index_dictionary_acceptors[acceptor.structure_file_index]

                for hydrogen_bond_partner in hydrogen_bond_partners:
                    hydrogen_bond_partner_index = index_dictionary_donor_atoms[
                        hydrogen_bond_partner.structure_file_index]

                    hydrogen_bond_matrix[acceptor_index][hydrogen_bond_partner_index] += 1

            if np.any(hydrogen_bond_matrix > 1):
                raise HydrogenBondsException(
                    'An entry in the adjacent matrix is bigger than one. Please check your hydrogen bonds.')

        return hydrogen_bond_matrix

    def get_hydrogen_bond_matrix(self):
        # TODO: make use multi thread
        # TODO: use only donor atoms and acceptors for matrix

        index_dictionary_donor_atoms = {}
        for index, atom in enumerate(self.donor_atoms):
            index_dictionary_donor_atoms[atom.structure_file_index] = index

        index_dictionary_acceptors = {}
        for index, atom in enumerate(self.acceptors):
            index_dictionary_acceptors[atom.structure_file_index] = index

        pool = ThreadPool(processes=self.core_units)
        matrices = pool.starmap(self.__get_hydrogen_bond_matrix_in_frame__,
                                zip(itertools.repeat(index_dictionary_donor_atoms, times=self.number_of_frames),
                                    itertools.repeat(index_dictionary_acceptors, times=self.number_of_frames),
                                    iter(range(self.number_of_frames))))
        pool.close()

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
            if not triplet.mask[frame]:
                continue

            donor_slot_free = len(
                triplet.donor_atom.hydrogen_bond_partners[self._system_name][frame]) < triplet.donor_atom.donor_slots
            acceptor_slot_free = len(
                triplet.acceptor.hydrogen_bond_partners[self._system_name][frame]) < triplet.acceptor.acceptor_slots

            if not donor_slot_free or not acceptor_slot_free:
                continue

            triplet.acceptor.add_hydrogen_bond_partner(frame=frame, atom=triplet.donor_atom,
                                                       system_name=self._system_name)


class HydrogenBondsDistanceCriterion(HydrogenBonds):
    @staticmethod
    def __get_sorted_distances__(xyz_triplet, hydrogen_bond_partners, frame):
        distances = []

        for partner in hydrogen_bond_partners:
            distances.append(
                np.linalg.norm(xyz_triplet - partner.xyz_trajectory[frame]))

        if len(distances) > 1:
            sorted_args = np.argsort(distances)
        else:
            sorted_args = np.array([0])

        return np.round(distances, 5), sorted_args

    @staticmethod
    def __check__(sorted_args, distances, triplet, frame):
        # calculate new triplet distance
        triplet_distance = np.linalg.norm(triplet.donor.xyz_trajectory[frame] - triplet.acceptor.xyz_trajectory[frame])
        triplet_distance = np.round(triplet_distance, 5)

        # Compare to biggest old distance of possible new bond partner.
        arg = sorted_args[-1]
        distance = distances[arg]

        # If the biggest distance is still smaller than the new one, else the new distance wins.
        if distance <= triplet_distance:
            return None
        else:
            return arg

    def __replace__(self, triplet_atom, new_atom, old_atom, frame):
        triplet_atom.remove_hydrogen_bond_partner(frame=frame, atom=old_atom,
                                                  system_name=self._system_name)
        triplet_atom.add_hydrogen_bond_partner(frame=frame, atom=new_atom,
                                               system_name=self._system_name)

    def __get_index__(self, triplet, frame, free_slot_is_donor):
        if free_slot_is_donor:
            hydrogen_bond_partners = triplet.acceptor.hydrogen_bond_partners[self._system_name][frame]
            distances, sorted_args = self.__get_sorted_distances__(
                xyz_triplet=triplet.acceptor.xyz_trajectory[frame],
                hydrogen_bond_partners=[donor_atom.covalent_bond_partners[0] for donor_atom in hydrogen_bond_partners],
                frame=frame)

        else:
            hydrogen_bond_partners = triplet.donor_atom.hydrogen_bond_partners[self._system_name][frame]
            distances, sorted_args = self.__get_sorted_distances__(xyz_triplet=triplet.donor.xyz_trajectory[frame],
                                                                   hydrogen_bond_partners=hydrogen_bond_partners,
                                                                   frame=frame)

        return self.__check__(sorted_args=sorted_args, distances=distances, triplet=triplet, frame=frame)

    # TODO: think about sanity checks (e.g. donor_atom looses  partner because of last triplet while rejecting others before)
    def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
        self.ensure_data_type.ensure_tuple(parameter=triplets, parameter_name='triplets')
        self.ensure_data_type.ensure_integer(parameter=frame, parameter_name='frame')

        for triplet in triplets:
            if not triplet.mask[frame]:
                continue

            donor_slot_free = len(
                triplet.donor_atom.hydrogen_bond_partners[self._system_name][frame]) < triplet.donor_atom.donor_slots
            acceptor_slot_free = len(
                triplet.acceptor.hydrogen_bond_partners[self._system_name][frame]) < triplet.acceptor.acceptor_slots

            if not donor_slot_free and not acceptor_slot_free:

                # get index where distances are smaller than assigned
                donor_index = self.__get_index__(triplet=triplet, frame=frame, free_slot_is_donor=False)
                acceptor_index = self.__get_index__(triplet=triplet, frame=frame, free_slot_is_donor=True)

                if donor_index is not None and acceptor_index is not None:
                    triplet.donor_atom.remove_hydrogen_bond_partner(frame=frame,
                                                                    atom=triplet.donor_atom.hydrogen_bond_partners[
                                                                        self._system_name][frame][donor_index],
                                                                    system_name=self._system_name)
                    triplet.acceptor.remove_hydrogen_bond_partner(frame=frame,
                                                                  atom=triplet.acceptor.hydrogen_bond_partners[
                                                                      self._system_name][frame][acceptor_index],
                                                                  system_name=self._system_name)

                    triplet.acceptor.add_hydrogen_bond_partner(frame=frame, atom=triplet.donor_atom,
                                                               system_name=self._system_name)

            elif not donor_slot_free and acceptor_slot_free:
                index = self.__get_index__(triplet=triplet, frame=frame, free_slot_is_donor=False)

                if index is not None:
                    self.__replace__(triplet_atom=triplet.donor_atom,
                                     old_atom=triplet.donor_atom.hydrogen_bond_partners[self._system_name][frame][
                                         index],
                                     new_atom=triplet.acceptor, frame=frame)

            elif not acceptor_slot_free and donor_slot_free:
                index = self.__get_index__(triplet=triplet, frame=frame, free_slot_is_donor=True)

                if index is not None:
                    self.__replace__(triplet_atom=triplet.acceptor,
                                     old_atom=triplet.acceptor.hydrogen_bond_partners[self._system_name][frame][index],
                                     new_atom=triplet.donor_atom, frame=frame)

            else:
                triplet.acceptor.add_hydrogen_bond_partner(frame=frame, atom=triplet.donor_atom,
                                                           system_name=self._system_name)
