import numpy as np
from mdtraj.geometry._geometry import _dist_mic, _dist, _dist_mic_displacement, _dist_displacement
from mdtraj.geometry.distance import _distance_mic, _distance, _displacement_mic, _displacement
from mdtraj.utils.validation import ensure_type

from yeti.get_features.metric import Metric


class Distance(Metric):

    @staticmethod
    def __calculate_no_pbc__(xyz, indices, opt):
        if opt:
            distances = np.empty((xyz.shape[0], indices.shape[0]), dtype=np.float32)
            _dist(xyz, indices, distances)
        else:
            distances = _distance(xyz=xyz, pairs=indices)

        return distances

    def __calculate_minimal_image_convention__(self, xyz, indices, opt):
        # check box
        box = ensure_type(self.unit_cell_vectors, dtype=np.float32, ndim=3, name='unitcell_vectors',
                          shape=(len(xyz), 3, 3), warn_on_cast=False)
        orthogonal = np.allclose(self.unit_cell_angles, 90)

        # calculate distances
        if opt:
            distances = np.empty((xyz.shape[0], indices.shape[0]), dtype=np.float32)

            _dist_mic(xyz, indices, box.transpose(0, 2, 1).copy(), distances, orthogonal)
        else:
            distances = _distance_mic(xyz=xyz, pairs=indices, box_vectors=box.transpose(0, 2, 1).copy(),
                                      orthogonal=orthogonal)

        return distances


class Displacement(Metric):
    def __calculate_no_pbc__(self, xyz, indices, opt):
        if opt:
            displacements = np.empty((xyz.shape[0], indices.shape[0], 3), dtype=np.float32)
            _dist_displacement(xyz, indices, displacements)
        else:
            displacements = _displacement(xyz=xyz, pairs=indices)

        return displacements

    def __calculate_minimal_image_convention__(self, xyz, indices, opt):
        box = ensure_type(self.unit_cell_vectors, dtype=np.float32, ndim=3, name='unitcell_vectors',
                          shape=(len(xyz), 3, 3), warn_on_cast=False)
        orthogonal = np.allclose(self.unit_cell_angles, 90)

        if opt:
            displacements = np.empty((xyz.shape[0], indices.shape[0], 3), dtype=np.float32)
            _dist_mic_displacement(xyz, indices, box.transpose(0, 2, 1).copy(), displacements, orthogonal)
        else:
            displacements = _displacement_mic(xyz=xyz, pairs=indices, box_vectors=box.transpose(0, 2, 1),
                                              orthogonal=orthogonal)

        return displacements

    def get_compatibility_layer(self, xyz, indices, periodic=True, opt=True):
        kwargs = dict(xyz=xyz, indices=indices, opt=opt)

        if periodic:
            return self.__calculate_minimal_image_convention__(**kwargs)
        else:
            return self.__calculate_no_pbc__(**kwargs)
