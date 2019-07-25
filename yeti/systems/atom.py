class AtomException(Exception):
    pass


class Atom(object):
    def __init__(self, structure_file_index, subsystem_index, name, residue, xyz_trajectory):
        self.name = name
        self.subsystem_index = subsystem_index
        self.structure_file_index = structure_file_index

        self.element = None
        self.residue = residue

        self.xyz_trajectory = xyz_trajectory

        # covalent bonds
        self.covalent_bond_partners = ()

        # hydrogen bonds
        self.is_donor_atom = None
        self.donor_slots = None

        self.is_acceptor = None
        self.acceptor_slots = None

        if self.is_acceptor or self.is_donor_atom:
            self.hydrogen_bond_partners = dict(subsystem=[[]] * self.xyz_trajectory.shape[0])
        else:
            self.hydrogen_bond_partners = None

    def add_covalent_bond(self, atom):
        self.covalent_bond_partners = (*self.covalent_bond_partners, atom)

    def update_donor_state(self, is_donor_atom, donor_slots):
        self.is_donor_atom = is_donor_atom
        self.donor_slots = donor_slots

    def update_acceptor_state(self, is_acceptor, acceptor_slots):
        self.is_acceptor = is_acceptor
        self.acceptor_slots = acceptor_slots

    def update_hydrogen_bond_partner(self, frame, atom):
        if self.hydrogen_bond_partners is not None:
            self.hydrogen_bond_partners[frame].append(atom)
        else:
            raise AtomException('The atom is neither acceptor nor a donor atom. Update its state first!')

    def __str__(self):
        return self.name
