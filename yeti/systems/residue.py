class Residue(object):
    def __init__(self, subsystem_index, structure_file_index, name):
        self.atoms = []
        self.sequence = []

        self.name = name
        self.structure_file_index = structure_file_index
        self.subsystem_index = subsystem_index

        self.number_of_atoms = 0

    def __str__(self):
        return '{name}{index:d}'.format(name=self.name, index=self.subsystem_index)

    def __add_atom__(self, atom):
        self.atoms.append(atom)
        self.sequence.append(str(atom))

        self.number_of_atoms += 1

    def __finalize__(self):
        self.atoms = tuple(self.atoms)
        self.sequence = tuple(self.sequence)
