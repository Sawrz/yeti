import numpy as np


class AtomWarning(Warning):
    pass


class AtomException(Exception):
    pass


class ResidueException(Exception):
    pass


class EnsureDataTypes(object):
    def __init__(self, exception_class):
        self.exception_class = exception_class

    def __check_type__(self, parameter, parameter_name, data_type):
        if type(parameter) is not data_type:
            msg = 'Wrong data type for parameter "{name}". Desired type is {data_type}'.format(
                name=parameter_name, data_type=data_type.__name__)
            raise self.exception_class(msg)

    def __check_numpy_dimensions__(self, parameter, parameter_name, desired_shape):
        actual_shape = parameter.shape
        msg = 'Wrong shape for parameter "{name}". Desired shape: {des_shape}.'.format(name=parameter_name,
                                                                                       des_shape=desired_shape)

        if len(actual_shape) != len(desired_shape):
            raise self.exception_class(msg)

        for actual_dim, desired_dim in zip(actual_shape, desired_shape):
            if desired_dim is None:
                continue
            else:
                if actual_dim != desired_dim:
                    raise self.exception_class(msg)

    def __check_numpy_data_type__(self, parameter, parameter_name, desired_dtype):
        desired_dtype = np.dtype(desired_dtype)

        if parameter.dtype != desired_dtype:
            msg = 'Wrong dtype for ndarray "{name}". Desired dtype is {data_type}'.format(name=parameter_name,
                                                                                          data_type=desired_dtype)
            raise self.exception_class(msg)

    def ensure_integer(self, parameter, parameter_name):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=int)

    def ensure_float(self, parameter, parameter_name):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=float)

    def ensure_string(self, parameter, parameter_name):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=str)

    def ensure_boolean(self, parameter, parameter_name):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=bool)

    def ensure_tuple(self, parameter, parameter_name):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=tuple)

    def ensure_list(self, parameter, parameter_name):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=list)

    def ensure_dict(self, parameter, parameter_name):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=dict)

    def ensure_numpy_array(self, parameter, parameter_name, shape, desired_dtype=None):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=np.ndarray)
        self.__check_numpy_dimensions__(parameter=parameter, parameter_name=parameter_name, desired_shape=shape)

        if desired_dtype is not None:
            self.__check_numpy_data_type__(parameter=parameter, parameter_name=parameter_name,
                                           desired_dtype=desired_dtype)

    def ensure_atom(self, parameter, parameter_name):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=Atom)

    def ensure_residue(self, parameter, parameter_name):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=Residue)


class Atom(object):
    def __init__(self, structure_file_index, subsystem_index, name, xyz_trajectory):

        self.ensure_data_type = EnsureDataTypes(exception_class=AtomException)
        self.ensure_data_type.ensure_string(parameter=name, parameter_name='name')
        self.ensure_data_type.ensure_integer(parameter=structure_file_index, parameter_name='structure_file_index')
        self.ensure_data_type.ensure_integer(parameter=subsystem_index, parameter_name='subsystem_index')
        self.ensure_data_type.ensure_numpy_array(parameter=xyz_trajectory, parameter_name='xyz_trajectory',
                                                 shape=(None, 3))

        self.name = name
        self.subsystem_index = subsystem_index
        self.structure_file_index = structure_file_index

        self.element = None
        self.residue = None

        self.xyz_trajectory = xyz_trajectory

        # covalent bonds
        self.covalent_bond_partners = ()

        # hydrogen bonds
        self.is_donor_atom = False
        self.donor_slots = 0

        self.is_acceptor = False
        self.acceptor_slots = 0

        self.hydrogen_bond_partners = None

    def set_residue(self, residue):
        self.ensure_data_type.ensure_residue(parameter=residue, parameter_name='residue')

        if self.residue is not None:
            raise AtomWarning('This atom belongs already to a residue. Changing relationship...')

        self.residue = residue

    def __update_covalent_bond__(self, atom):
        # TODO: merge __update_hydrogen_bond_partner__ and __update_covalent_bond__ into one method

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

    def add_system(self, system_name):
        # TODO: Think about overwriting existing data

        self.ensure_data_type.ensure_string(parameter=system_name, parameter_name='system_name')

        if type(self.hydrogen_bond_partners) is dict:
            self.hydrogen_bond_partners[system_name] = [[] for i in range(self.xyz_trajectory.shape[0])]
        else:
            raise AtomException('This atom is neither acceptor nor a donor atom. Update its state first!')

    def __reset_hydrogen_bond_partners__(self, is_hydrogen_bond_active):
        self.ensure_data_type.ensure_boolean(parameter=is_hydrogen_bond_active,
                                             parameter_name='is_hydrogen_bond_active')

        if is_hydrogen_bond_active:
            # TODO: find more efficient way to create list with n frames
            if type(self.hydrogen_bond_partners) is dict:
                for key in self.hydrogen_bond_partners.keys():
                    self.add_system(system_name=key)
            else:
                self.hydrogen_bond_partners = {}
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
        self.__reset_hydrogen_bond_partners__(is_hydrogen_bond_active=is_donor_atom)

    def update_acceptor_state(self, is_acceptor, acceptor_slots):
        self.ensure_data_type.ensure_boolean(parameter=is_acceptor, parameter_name='is_acceptor')
        self.ensure_data_type.ensure_integer(parameter=acceptor_slots, parameter_name='acceptor_slots')

        if not is_acceptor and acceptor_slots > 0:
            raise AtomException('A non-acceptor atom does not have any acceptor slots.')

        if is_acceptor and acceptor_slots < 1:
            raise AtomException('Acceptor atom need acceptor slots.')

        self.is_acceptor = is_acceptor
        self.acceptor_slots = acceptor_slots
        self.__reset_hydrogen_bond_partners__(is_hydrogen_bond_active=is_acceptor)

    def __update_hydrogen_bond_partner__(self, atom, frame, system_name):
        # TODO: merge __update_hydrogen_bond_partner__ and __update_covalent_bond__ into one method

        if type(atom) is not Atom:
            raise AtomException('Wrong data type for parameter "atom". Desired type is atom')

        self.ensure_data_type.ensure_integer(frame, 'frame')
        self.ensure_data_type.ensure_string(system_name, 'system_name')

        if frame < 0:
            raise AtomException('Frame has to be a positive integer.')

        if system_name not in self.hydrogen_bond_partners.keys():
            raise AtomException('Subsystem does not exist. Create it first!')

        if atom.structure_file_index == self.structure_file_index:
            raise AtomException('Atom can not have a covalent bond with itself.')

        if atom in self.hydrogen_bond_partners[system_name][frame]:
            raise AtomWarning('Hydrogen bond already exists. Skipping...')

        self.hydrogen_bond_partners[system_name][frame].append(atom)

    def add_hydrogen_bond_partner(self, frame, atom, system_name):
        if self.hydrogen_bond_partners is None:
            raise AtomException('This atom is neither acceptor nor a donor atom. Update its state first!')
        elif atom.hydrogen_bond_partners is None:
            raise AtomException('Parameter atom is neither acceptor nor a donor atom. Update its state first!')
        else:
            self.__update_hydrogen_bond_partner__(atom=atom, frame=frame, system_name=system_name)
            atom.__update_hydrogen_bond_partner__(atom=self, frame=frame, system_name=system_name)

    def purge_hydrogen_bond_partner_history(self, system_name):
        self.ensure_data_type.ensure_string(system_name, 'system_name')

        if self.is_acceptor or self.is_donor_atom:
            if system_name not in self.hydrogen_bond_partners.keys():
                raise AtomException('Subsystem does not exist. Create it first!')

            del self.hydrogen_bond_partners[system_name]
            self.add_system(system_name=system_name)

        else:
            raise AtomException('The given atom is neither donor nor acceptor. Purging does not make sense!')

    def __str__(self):
        return self.name


class Residue(object):
    def __init__(self, subsystem_index, structure_file_index, name):
        self.ensure_data_type = EnsureDataTypes(exception_class=ResidueException)
        self.ensure_data_type.ensure_integer(parameter=subsystem_index, parameter_name='subsystem_index')
        self.ensure_data_type.ensure_integer(parameter=structure_file_index, parameter_name='structure_file_index')
        self.ensure_data_type.ensure_string(parameter=name, parameter_name='name')

        self.atoms = []
        self.sequence = []

        self.name = name
        self.structure_file_index = structure_file_index
        self.subsystem_index = subsystem_index

        self.number_of_atoms = 0

    def __str__(self):
        return '{name}{index:d}'.format(name=self.name, index=self.subsystem_index)

    # TODO: Think about way to connect residue to atom and atom to residue with one command (maybe in layer above?)
    def add_atom(self, atom):
        self.ensure_data_type.ensure_atom(parameter=atom, parameter_name='atom')

        self.atoms.append(atom)
        self.sequence.append(str(atom))

        self.number_of_atoms += 1

    def finalize(self):
        self.atoms = tuple(self.atoms)
        self.sequence = tuple(self.sequence)

    def definalize(self):
        self.atoms = list(self.atoms)
        self.sequence = list(self.sequence)

    # TODO add method to refresh sequence?
