import numpy as np
from mdtraj.utils.validation import ensure_type


class MetricException(Exception):
    pass


class Metric(object):
    def __init__(self, periodic, unit_cell_angles, unit_cell_vectors):
        self.unit_cell_vectors = unit_cell_vectors
        self.unit_cell_angles = unit_cell_angles
        self.periodic = periodic

    @staticmethod
    def __get_atom__(name, residue):
        atom_index = np.where(residue.sequence == name)[0][0]

        return residue.atoms[atom_index]

    def __get_atoms__(self, *atom_name_residue_pairs):
        atom_list = []

        for atom_name, residue in atom_name_residue_pairs:
            atom_list.append(self.__get_atom__(name=atom_name, residue=residue))

        return tuple(atom_list)

    @staticmethod
    def __prepare_xyz_data__(*atoms):
        xyz = []

        for atom in atoms:
            tmp_xyz = np.expand_dims(atom.xyz_trajectory, axis=1)
            xyz.append(tmp_xyz)

        xyz = np.hstack(xyz)

        # ensure data types are right
        xyz = ensure_type(xyz, dtype=np.float32, ndim=3, name='traj.xyz', shape=(None, None, 3), warn_on_cast=False)

        return xyz

    @staticmethod
    def __prepare_atom_indices__(amount):
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

    def __calculate_no_pbc__(self, xyz, indices, opt):
        pass

    def __calculate_minimal_image_convention__(self, xyz, indices, opt):
        pass

    def __calculate__(self, atoms, opt, amount):
        xyz = self.__prepare_xyz_data__(*atoms)
        indices = self.__prepare_atom_indices__(amount=amount)

        kwargs = dict(xyz=xyz, indices=indices, opt=opt)

        if self.periodic:
            feature = self.__calculate_minimal_image_convention__(**kwargs)
        else:
            feature = self.__calculate_no_pbc__(**kwargs)

        return feature.flatten()
