import numpy as np


class Metric(object):
    def __init__(self, periodic, box_vectors, unit_cell_angles, unit_cell_vectors):
        self.unit_cell_vectors = unit_cell_vectors
        self.unit_cell_angles = unit_cell_angles
        self.box_vectors = box_vectors
        self.periodic = periodic

    @staticmethod
    def __get_atom__(name, residue):
        atom_index = np.where(residue.sequence == name)[0][0]

        return residue.atoms[atom_index]

    def __get_atoms__(self, *atom_name_residue_pairs):
        atom_list = []

        for atom_name, residue in atom_name_residue_pairs:
            atom_list.append(self.__get_atom__(name=atom_name, residue=residue))

        return atom_list
