class Atom(object):
    def __init__(self, structure_file_index, subsystem_index, name, residue, xyz_trajectory):
        self.name = name
        self.subsystem_index = subsystem_index
        self.structure_file_index = structure_file_index

        self.element = None
        self.residue = residue

        self.xyz_trajectory = xyz_trajectory

    def __str__(self):
        return self.name
