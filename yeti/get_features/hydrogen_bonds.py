import multiprocessing as mp

import numpy as np

from yeti.get_features.angles import Angle
from yeti.get_features.distances import Distance


class TripletException(Exception):
    pass


class HydrogenBondsException(Exception):
    pass


class Triplet(object):
    def __init__(self, donor_atom, acceptor, periodic, unit_cell_angles, unit_cell_vectors):

        self.periodic = periodic
        self.unit_cell_angles = unit_cell_angles
        self.unit_cell_vectors = unit_cell_vectors

        if len(donor_atom.covalent_bond_partners) > 1:
            msg = 'Donor atom has more than one covalent bond. That violates the assumption of this method. ' \
                  'Please contact the developer.'
            raise TripletException(msg)

        self.donor = self.donor_atom.covalent_bond_partners[0]
        self.donor_atom = donor_atom
        self.acceptor = acceptor

        self.triplet = (self.donor, self.donor_atom, self.acceptor)
        self.mask = None

    def create_mask(self, distance_cutoff, angle_cutoff):
        # calculate distances
        dist = Distance(periodic=self.periodic, unit_cell_angles=self.unit_cell_anglesunit_cell_angles,
                        unit_cell_vectors=self.unit_cell_vectors)
        distances = dist.__calculate__((self.donor, self.acceptor), opt=True, amount=2)

        angle = Angle(periodic=self.periodic, unit_cell_angles=self.unit_cell_anglesunit_cell_angles,
                      unit_cell_vectors=self.unit_cell_vectors)
        angles = angle.__calculate__(self.triplet, opt=True, amount=3)

        # Security check if some angle is nan
        is_there_nan = np.isnan(angles)

        if is_there_nan.any():
            raise TripletException('Angle calculation throws nan. Please contact the developer.')

        self.mask = np.logical_and(distances < distance_cutoff, np.pi - angles < angle_cutoff)


class HydrogenBonds(object):
    def __init__(self, atoms, periodic, unit_cell_angles, unit_cell_vectors, system_name, number_of_frames):
        self.periodic = periodic
        self.unit_cell_angles = unit_cell_angles
        self.unit_cell_vectors = unit_cell_vectors

        self.atoms = atoms
        self._system_name = system_name
        self.number_of_frames = number_of_frames

        self.donor_atoms = tuple([atom for atom in atoms if atom.is_donor_atom])
        self.acceptors = tuple([atom for atom in atoms if atom.is_acceptor])

    def __build_triplets__(self, distance_cutoff, angle_cutoff):
        triplets = []

        # initialize triplets
        for donor_atom in self.donor_atoms:
            for acceptor in self.acceptors:
                triplet = Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=self.periodic,
                                  unit_cell_angles=self.unit_cell_angles, unit_cell_vectors=self.unit_cell_vectors)
                triplets.append(triplet)

        # create_masks
        processes = []

        for triplet in triplets:
            worker = mp.Process(target=triplet.create_mask,
                                kwargs=dict(distance_cutoff=distance_cutoff, angle_cutoff=angle_cutoff))
            processes.append(worker)

        [worker.start() for worker in processes]

        return triplets

    def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
        pass

    def __get_hydrogen_bonds__(self, triplets):
        processes = []
        kwargs = dict(triplets=triplets)

        for frame in range(self.number_of_frames):
            kwargs['frame'] = frame
            worker = mp.Process(target=self.__get_hydrogen_bonds_in_frame__, kwargs=kwargs)
            processes.append(worker)

        [worker.start for worker in processes]

    def calculate_hydrogen_bonds(self, distance_cutoff, angle_cutoff):
        for donor_atom in self.donor_atoms:
            donor_atom.purge_hydrogen_bond_partner_history(system_name=self._system_name)
        for acceptor_atom in self.acceptors:
            acceptor_atom.purge_hydrogen_bond_partner_history(system_name=self._system_name)

        triplets = self.__build_triplets__(distance_cutoff=distance_cutoff, angle_cutoff=angle_cutoff)
        self.__get_hydrogen_bonds__(triplets=triplets)

    def __get_hydrogen_bond_matrix_in_frame__(self, index_dictionary, frame):
        hydrogen_bond_matrix = np.zeros((self.number_of_frames, self.number_of_frames))

        for acceptor in self.acceptors:
            hydrogen_bond_partners = acceptor.hydrogen_bond_partners[frame]

            if len(hydrogen_bond_partners) > 0:
                acceptor_index = index_dictionary[acceptor.structure_file_index]

                for hydrogen_bond_partner in hydrogen_bond_partners:
                    hydrogen_bond_partner_index = index_dictionary[hydrogen_bond_partner.structure_file_index]

                    hydrogen_bond_matrix[acceptor_index][hydrogen_bond_partner_index] += 1
                    hydrogen_bond_matrix[hydrogen_bond_partner_index][acceptor_index] += 1

        return hydrogen_bond_matrix

    def get_hydrogen_bond_matrix(self):
        index_dictionary = {}
        for index, atom in enumerate(self.atoms):
            index_dictionary[atom.structure_file_index] = index

        processes = []
        kwargs = dict(index_dictionary=index_dictionary)

        for frame in range(self.number_of_frames):
            kwargs['frame'] = frame
            worker = mp.Process(target=self.__get_hydrogen_bond_matrix_in_frame__, kwargs=kwargs)
            processes.append(worker)

        return np.array([worker.start for worker in processes])

    def __get_number_hydrogen_bonds_for_frame__(self, frame):
        hydrogen_bonds = 0

        for acceptor in self.acceptors:
            acceptor_hydrogen_bonds = len(acceptor.hydrogen_bond_partners[self._system_name][frame])
            hydrogen_bonds += acceptor_hydrogen_bonds

        return hydrogen_bonds

    def get_frame_wise_hydrogen_bonds(self):
        processes = []
        kwargs = {}

        for frame in range(self.number_of_frames):
            kwargs['frame'] = frame
            worker = mp.Process(target=self.__get_number_hydrogen_bonds_for_frame__, kwargs=kwargs)
            processes.append(worker)

        return np.array([worker.start for worker in processes])


class HydrogenBondsFirstComesFirstServes(HydrogenBonds):
    def __get_hydrogen_bonds_in_frame__(self, triplets, frame):
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