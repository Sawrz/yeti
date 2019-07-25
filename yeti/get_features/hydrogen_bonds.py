import multiprocessing as mp

import numpy as np

from yeti.get_features.angles import Angle
from yeti.get_features.distances import Distance


class TripletException(Exception):
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

        donor = self.donor_atom.covalent_bond_partners[0]

        self.triplet = (donor, donor_atom, acceptor)
        self.mask = None

    def create_mask(self, distance_cutoff, angle_cutoff):
        # calculate distances
        dist = Distance(periodic=self.periodic, unit_cell_angles=self.unit_cell_anglesunit_cell_angles,
                        unit_cell_vectors=self.unit_cell_vectors)
        distances = dist.__calculate__((self.tripletp[0], self.triplet[2]), opt=True, amount=2)

        angle = Angle(periodic=self.periodic, unit_cell_angles=self.unit_cell_anglesunit_cell_angles,
                      unit_cell_vectors=self.unit_cell_vectors)
        angles = angle.__calculate__(self.triplet, opt=True, amount=3)

        # Security check if some angle is nan
        is_there_nan = np.isnan(angles)

        if is_there_nan.any():
            raise TripletException('Angle calculation throws nan. Please contact the developer.')

        self.mask = np.logical_and(distances < distance_cutoff, np.pi - angles < angle_cutoff)


class HydrogenBonds(object):
    def __init__(self, atoms, periodic, unit_cell_angles, unit_cell_vectors, system_name):
        self.periodic = periodic
        self.unit_cell_angles = unit_cell_angles
        self.unit_cell_vectors = unit_cell_vectors

        self.atoms = atoms
        self.number_of_frames = self.atoms.xyz_trajectory.shape[0]
        self._system_name = system_name

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

    def __calculate_hydrogen_bond_fcfs__(self, triplet, distance_cutoff, angle_cutoff, frame):
        pass

    def calculate_hydrogen_bonds_in_frame(self, frame, distance_cutoff, angle_cutoff):
        pass
