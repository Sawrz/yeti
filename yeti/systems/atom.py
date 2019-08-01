from yeti.utils.ensure_data_type import EnsureDataTypes


class AtomException(Exception):
    pass


class Atom(object):
    def __init__(self, structure_file_index, subsystem_index, name, residue, xyz_trajectory):

        self.ensure_data_type = EnsureDataTypes(exception_class=AtomException)
        self.ensure_data_type.ensure_string(parameter=name, parameter_name='name')
        self.ensure_data_type.ensure_integer(parameter=structure_file_index, parameter_name='structure_file_index')
        self.ensure_data_type.ensure_integer(parameter=subsystem_index, parameter_name='subsystem_index')
        self.ensure_data_type.ensure_residue(parameter=residue, parameter_name='residue')
        self.ensure_data_type.ensure_numpy_array(parameter=xyz_trajectory, parameter_name='xyz_trajectory',
                                                 shape=(3, None))

        self.name = name
        self.subsystem_index = subsystem_index
        self.structure_file_index = structure_file_index

        self.element = None
        self.residue = residue

        self.xyz_trajectory = xyz_trajectory

        # covalent bonds
        self.covalent_bond_partners = ()

        # hydrogen bonds
        self.is_donor_atom = False
        self.donor_slots = None

        self.is_acceptor = False
        self.acceptor_slots = None

        self.hydrogen_bond_partners = None

    def add_covalent_bond(self, atom):
        self.covalent_bond_partners = (*self.covalent_bond_partners, atom)

    def __update_hydrogen_bond_partners__(self, is_hydrogen_bond_active):
        if is_hydrogen_bond_active:
            self.hydrogen_bond_partners = dict(subsystem=[[]] * self.xyz_trajectory.shape[0])
        else:
            self.hydrogen_bond_partners = None

    def update_donor_state(self, is_donor_atom, donor_slots):
        self.is_donor_atom = is_donor_atom
        self.donor_slots = donor_slots
        self.__update_hydrogen_bond_partners__(is_hydrogen_bond_active=is_donor_atom)

    def update_acceptor_state(self, is_acceptor, acceptor_slots):
        self.is_acceptor = is_acceptor
        self.acceptor_slots = acceptor_slots
        self.__update_hydrogen_bond_partners__(is_hydrogen_bond_active=is_acceptor)

    def add_hydrogen_bond_partner(self, frame, atom):
        if self.hydrogen_bond_partners is not None:
            self.hydrogen_bond_partners[frame].append(atom)
        else:
            raise AtomException('The atom is neither acceptor nor a donor atom. Update its state first!')

    def purge_hydrogen_bond_partner_history(self, system_name):
        if self.is_acceptor or self.is_donor_atom:
            del self.hydrogen_bond_partners[system_name]
            self.hydrogen_bond_partners[system_name] = [[]] * self.xyz_trajectory.shape[0]
        else:
            raise AtomException('The given atom is neither donor nor acceptor. Purging does not make sense!')

    def __str__(self):
        return self.name
