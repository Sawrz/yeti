from yeti.utils.ensure_data_type_residue import Ensure_DataTypesWithResidue


class AtomWarning(Warning):
    pass


class AtomException(Exception):
    pass


class Atom(object):
    def __init__(self, structure_file_index, subsystem_index, name, residue, xyz_trajectory):

        self.ensure_data_type = Ensure_DataTypesWithResidue(exception_class=AtomException)
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
        self.donor_slots = 0

        self.is_acceptor = False
        self.acceptor_slots = 0

        self.hydrogen_bond_partners = None

    def __update_covalent_bond__(self, atom):
        if type(atom) is not Atom:
            raise AtomException('Wrong data type for parameter "atom". Desired type is atom')

        if atom.structure_file_index == self.structure_file_index:
            raise AtomException('Atom can not have a covalent bond with itself.')

        if atom in self.covalent_bond_partners:
            raise AtomWarning('Covalent bond already exists. Skipping...')

        self.covalent_bond_partners = (*self.covalent_bond_partners, atom)

    def add_covalent_bond(self, atom):
        self.__update_covalent_bond__(atom=atom)
        atom.__update_covalent_bond__(atom=self)

    def __update_hydrogen_bond_partners__(self, is_hydrogen_bond_active):
        self.ensure_data_type.ensure_boolean(parameter=is_hydrogen_bond_active,
                                             parameter_name='is_hydrogen_bond_active')

        if is_hydrogen_bond_active:
            self.hydrogen_bond_partners = dict(subsystem=[[]] * self.xyz_trajectory.shape[1])
        else:
            self.hydrogen_bond_partners = None

    def update_donor_state(self, is_donor_atom, donor_slots):
        self.ensure_data_type.ensure_boolean(parameter=is_donor_atom, parameter_name='is_donor_atom')
        self.ensure_data_type.ensure_integer(parameter=donor_slots, parameter_name='donor_slots')

        if not is_donor_atom and donor_slots > 0:
            raise AtomException('A non-donor atom does not have any donor slots.')

        if is_donor_atom and donor_slots < 1:
            raise AtomException('Donor atom need donor slots.')

        self.is_donor_atom = is_donor_atom
        self.donor_slots = donor_slots
        self.__update_hydrogen_bond_partners__(is_hydrogen_bond_active=is_donor_atom)

    def update_acceptor_state(self, is_acceptor, acceptor_slots):
        self.ensure_data_type.ensure_boolean(parameter=is_acceptor, parameter_name='is_acceptor')
        self.ensure_data_type.ensure_integer(parameter=acceptor_slots, parameter_name='acceptor_slots')

        if not is_acceptor and acceptor_slots > 0:
            raise AtomException('A non-acceptor atom does not have any acceptor slots.')

        if is_acceptor and acceptor_slots < 1:
            raise AtomException('Acceptor atom need acceptor slots.')

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
