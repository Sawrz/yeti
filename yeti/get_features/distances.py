import numpy as np
from mdtraj.geometry._geometry import _dist_mic, _dist
from mdtraj.geometry.distance import _distance_mic, _distance
from mdtraj.utils.validation import ensure_type

from yeti.get_features.metric import Metric


class Distance(Metric):

    @staticmethod
    def __calculate_no_pbc__(xyz, indices, opt):
        if opt:
            output = np.empty((xyz.shape[0], indices.shape[0]), dtype=np.float32)
            _dist(xyz, indices, output)
        else:
            output = _distance(xyz, indices)

        return output

    def __calculate_minimal_image_convention__(self, xyz, indices, opt):
        # check box
        box = ensure_type(self.unit_cell_vectors, dtype=np.float32, ndim=3, name='unitcell_vectors',
                          shape=(len(xyz), 3, 3), warn_on_cast=False)
        orthogonal = np.allclose(self.unit_cell_angles, 90)

        # calculate distances
        if opt:
            output = np.empty((xyz.shape[0], indices.shape[0]), dtype=np.float32)
            _dist_mic(xyz, indices, box.transpose(0, 2, 1).copy(), output, orthogonal)
        else:
            output = _distance_mic(xyz, indices, box.transpose(0, 2, 1), orthogonal)

        return output

    def get_distance(self, atom_name_residue_pair_01, atom_name_residue_pair_02, opt=True):
        name_residue_pairs = (atom_name_residue_pair_01, atom_name_residue_pair_02)
        atoms = self.__get_atoms__(*name_residue_pairs)

        return self.__calculate__(atoms=atoms, opt=opt, amount=2)
