import numpy as np
from mdtraj.utils.validation import ensure_type

from yeti.systems.building_blocks import EnsureDataTypes


class MetricException(Exception):
    pass


class Metric(object):
    def __init__(self, unit_cell_angles, unit_cell_vectors):
        self.ensure_data_type = EnsureDataTypes(exception_class=MetricException)
        self.ensure_data_type.ensure_numpy_array(parameter=unit_cell_angles, parameter_name='unit_cell_angles',
                                                 shape=(None, 3))
        self.ensure_data_type.ensure_numpy_array(parameter=unit_cell_vectors, parameter_name='unit_cell_vectors',
                                                 shape=(None, 3, 3))

        self.unit_cell_vectors = unit_cell_vectors
        self.unit_cell_angles = unit_cell_angles

    def __prepare_xyz_data__(self, atoms):
        self.ensure_data_type.ensure_tuple(parameter=atoms, parameter_name='atoms')

        xyz = []

        for atom in atoms:
            self.ensure_data_type.ensure_atom(parameter=atom, parameter_name='atom in tuple atoms')
            tmp_xyz = np.expand_dims(atom.xyz_trajectory, axis=1)
            xyz.append(tmp_xyz)

        xyz = np.hstack(xyz)

        # ensure data types are right
        xyz = ensure_type(xyz, dtype=np.float32, ndim=3, name='traj.xyz', shape=(None, None, 3), warn_on_cast=False)

        return xyz

    def __prepare_atom_indices__(self, amount):
        self.ensure_data_type.ensure_integer(parameter=amount, parameter_name='amount')

        if amount == 2:
            name = 'atom_pairs'
            shape = (None, 2)
        elif amount == 3:
            name = 'angle_indices'
            shape = (None, 3)
        elif amount == 4:
            name = 'indices'
            shape = (None, 4)
        else:
            raise MetricException('Invalid amount.')

        indices = np.arange(0, amount)
        indices = np.expand_dims(indices, axis=0)
        indices = ensure_type(indices, dtype=np.int32, ndim=2, name=name, shape=shape, warn_on_cast=False)

        return indices

    def __mdtraj_paramaeter_compatibility_check__(self, xyz, indices, opt, atom_amount):
        self.ensure_data_type.ensure_numpy_array(parameter=indices, parameter_name='indices', shape=(None, atom_amount),
                                                 desired_dtype=np.int32)

        self.ensure_data_type.ensure_numpy_array(parameter=xyz, parameter_name='xyz',
                                                 shape=(None, indices.shape[0] * atom_amount, 3),
                                                 desired_dtype=np.float32)
        self.ensure_data_type.ensure_boolean(parameter=opt, parameter_name='opt')

    def __calculate_no_pbc__(self, xyz, indices, opt):
        pass

    def __calculate_minimal_image_convention__(self, xyz, indices, opt):
        pass

    def calculate(self, atoms, opt, periodic):
        self.ensure_data_type.ensure_boolean(parameter=opt, parameter_name='opt')
        self.ensure_data_type.ensure_boolean(parameter=periodic, parameter_name='periodic')

        xyz = self.__prepare_xyz_data__(atoms)
        indices = self.__prepare_atom_indices__(amount=len(atoms))

        kwargs = dict(xyz=xyz, indices=indices, opt=opt)

        if periodic:
            feature = self.__calculate_minimal_image_convention__(**kwargs)
        else:
            feature = self.__calculate_no_pbc__(**kwargs)

        return feature.flatten()
