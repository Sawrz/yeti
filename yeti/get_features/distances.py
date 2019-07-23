import numpy as np
from mdtraj.geometry._geometry import _dist_mic
from mdtraj.geometry.distance import _distance_mic
from mdtraj.utils.validation import ensure_type

from yeti.get_features.metric import Metric


class Distance(Metric):

    def __calculate_no_pbc__(self, atom_01, atom_02):
        return np.linalg.norm(atom_01.xyz_trajectory - atom_02.xyz_trajectory, axis=1)

    def __calculate_minimal_image_convention__(self, atom_01, atom_02, opt=True):
        # create compatibility parameters for mdtraj
        atom_pairs = np.array([[0, 1]])

        xyz_01 = np.expand_dims(atom_01.xyz_trajectory, axis=1)
        xyz_02 = np.expand_dims(atom_02.xyz_trajectory, axis=1)
        xyz = np.hstack((xyz_01, xyz_02))

        orthogonal = np.allclose(self.unit_cell_angles, 90)

        # ensure data types are right
        xyz = ensure_type(xyz, dtype=np.float32, ndim=3, name='traj.xyz', shape=(None, None, 3), warn_on_cast=False)
        box = ensure_type(self.unit_cell_vectors, dtype=np.float32, ndim=3, name='unitcell_vectors',
                          shape=(len(xyz), 3, 3), warn_on_cast=False)
        pairs = ensure_type(atom_pairs, dtype=np.int32, ndim=2, name='atom_pairs', shape=(None, 2), warn_on_cast=False)

        # calculate distances
        if opt:
            output = np.empty((xyz.shape[0], 1), dtype=np.float32)
            _dist_mic(xyz, pairs, box.transpose(0, 2, 1).copy(), output, orthogonal)
        else:
            output = _distance_mic(xyz, pairs, box.transpose(0, 2, 1), orthogonal)

        return output.flatten()

    def get_distance(self, atom_name_residue_pair_01, atom_name_residue_pair_02):
        name_residue_pairs = (atom_name_residue_pair_01, atom_name_residue_pair_02)

        atom_01, atom_02 = self.__get_atoms__(*name_residue_pairs)

        if self.periodic:
            return self.__calculate_minimal_image_convention__(atom_01=atom_01, atom_02=atom_02)
        else:
            return self.__calculate_no_pbc__(atom_01=atom_01, atom_02=atom_02)
