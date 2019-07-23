import numpy as np


class Residue(object):
    def __init__(self, subssytem_index, structure_file_index, name):
        self.atoms = []

        self.name = name
        self.sequence = None
        self.structure_file_index = structure_file_index
        self.subssytem_index = subssytem_index

        self.number_of_atoms = len(self.atoms)

    def __str__(self):
        return '{name}{index:d}'.format(name=self.name, index=self.structure_file_index)

    def __add_atom__(self, atom):
        self.atoms.append(atom)

    def __get_atom_sequence__(self):
        self.sequence = np.array([str(atom) for atom in self.atoms])

    def __finalize__(self):
        self.atoms = tuple(self.atoms)
        self.number_of_atoms = len(self.atoms)

        self.__get_atom_sequence__()
