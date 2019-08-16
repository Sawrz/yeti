import string

import numpy as np

from yeti.get_features.hydrogen_bonds import Triplet
from yeti.systems.building_blocks import Atom


def create_data_type_exception_messages(parameter_name, data_type_name):
    return 'Wrong data type for parameter "{name}". Desired type is {data_type}'.format(name=parameter_name,
                                                                                        data_type=data_type_name)


def create_array_shape_exception_messages(parameter_name, desired_shape):
    return 'Wrong shape for parameter "{name}". Desired shape: {des_shape}.'.format(name=parameter_name,
                                                                                    des_shape=desired_shape)


def create_array_dtype_exception_messages(parameter_name, dtype_name):
    return 'Wrong dtype for ndarray "{name}". Desired dtype is {data_type}'.format(name=parameter_name,
                                                                                   data_type=dtype_name)


def build_unit_cell_angles_and_vectors(number_of_frames):
    angles = []
    vectors = []

    for i in range(number_of_frames):
        angles.append([90, 90, 90])
        vectors.append([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    angles = np.array(angles, dtype=np.float32)
    vectors = np.array(vectors, dtype=np.float32)

    return angles, vectors


def build_atom_triplet():
    # first frame is h-bond
    # second frame is not because of distance
    # third frame is not because of angle

    donor = Atom(structure_file_index=0, subsystem_index=0, name='A',
                 xyz_trajectory=np.array([[0.1, 0.4, 0.3], [0.1, 0.4, 0.3], [0.1, 0.4, 0.3]]))
    donor_atom = Atom(structure_file_index=1, subsystem_index=1, name='B',
                      xyz_trajectory=np.array([[0.1, 0.5, 0.2], [0.1, 0.5, 0.2], [0.5, 0.5, 0.2]]))
    acceptor = Atom(structure_file_index=2, subsystem_index=2, name='C',
                    xyz_trajectory=np.array([[0.1, 0.6, 0.4], [0.1, 0.7, 0.4], [0.1, 0.6, 0.4]]))

    donor.add_covalent_bond(atom=donor_atom)

    donor_atom.update_donor_state(is_donor_atom=True, donor_slots=1)
    #donor_atom.add_system(system_name='test_system')

    acceptor.update_acceptor_state(is_acceptor=True, acceptor_slots=2)
    #acceptor.add_system(system_name='test_system')


    return donor, donor_atom, acceptor


def build_multi_atom_triplets(amount=2):
    all_atoms = []
    names = list(string.ascii_uppercase)

    for triplet_number in range(amount):
        atoms = build_atom_triplet()
        starting_index = triplet_number * len(atoms)

        for atom_number, atom in enumerate(atoms):
            atom.subsystem_index = starting_index + atom_number
            atom.name = names[atom.subsystem_index]
            atom.structure_file_index = atom.subsystem_index + 2

            all_atoms.append(atom)

    return tuple(all_atoms)


def build_triplet():
    donor, donor_atom, acceptor = build_atom_triplet()
    unit_cell_angles, unit_cell_vectors = build_unit_cell_angles_and_vectors(number_of_frames=3)

    return Triplet(donor_atom=donor_atom, acceptor=acceptor, periodic=True, unit_cell_angles=unit_cell_angles,
                   unit_cell_vectors=unit_cell_vectors)
