import numpy as np
from mdtraj.geometry._geometry import _dihedral_mic, _dihedral, _angle_mic, _angle
from mdtraj.utils.validation import ensure_type

from yeti.get_features.distances import Displacement
from yeti.get_features.metric import Metric


class AngleException(Exception):
    pass


class DihedralException(Exception):
    pass


class Angle(Metric):
    def __init__(self, *args, **kwargs):
        super(Angle, self).__init__(*args, **kwargs)
        self.ensure_data_type.exception_class = AngleException

    def __mdtraj_paramaeter_compatibility_check__(self, xyz, indices, opt):
        super(Angle, self).__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt,
                                                                     atom_amount=3)

    def __calculate_angle__(self, xyz, indices, periodic, out):
        """SOURCE: mdTraj
           MODIFICATION: displacement function
        """
        # TODO: Think about restricting amount of atoms only to 3
        self.ensure_data_type.ensure_numpy_array(parameter=out, parameter_name='out',
                                                 shape=(xyz.shape[0], indices.shape[0]), desired_dtype=np.float32)

        ix01 = indices[0, [1, 0]]
        xyz01 = xyz[:, ix01]

        ix21 = indices[0, [1, 2]]
        xyz21 = xyz[:, ix21]

        indices = np.array([[0, 1]], dtype=np.int32)

        displacement = Displacement(periodic=periodic, unit_cell_angles=self.unit_cell_angles,
                                    unit_cell_vectors=self.unit_cell_vectors)

        u_prime = displacement.get_compatibility_layer(xyz=xyz01, indices=indices, periodic=periodic, opt=False)
        v_prime = displacement.get_compatibility_layer(xyz=xyz21, indices=indices, periodic=periodic, opt=False)
        u_norm = np.sqrt((u_prime ** 2).sum(-1))
        v_norm = np.sqrt((v_prime ** 2).sum(-1))

        # adding a new axis makes sure that broasting rules kick in on the third
        # dimension
        u = u_prime / (u_norm[..., np.newaxis])
        v = v_prime / (v_norm[..., np.newaxis])

        return np.arccos((u * v).sum(-1), out=out)

    def __calculate_no_pbc__(self, xyz, indices, opt):
        self.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt)

        angles = np.zeros((xyz.shape[0], indices.shape[0]), dtype=np.float32)

        if opt:
            _angle(xyz, indices, angles)
        else:
            self.__calculate_angle__(xyz=xyz, indices=indices, periodic=False, out=angles)

        return angles

    def __calculate_minimal_image_convention__(self, xyz, indices, opt):
        self.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt)

        angles = np.zeros((xyz.shape[0], indices.shape[0]), dtype=np.float32)

        box = ensure_type(self.unit_cell_vectors, dtype=np.float32, ndim=3, name='unitcell_vectors',
                          shape=(len(xyz), 3, 3), warn_on_cast=True)

        if opt:
            orthogonal = np.allclose(self.unit_cell_angles, 90)
            _angle_mic(xyz, indices, box.transpose(0, 2, 1).copy(), angles, orthogonal)
        else:
            self.__calculate_angle__(xyz=xyz, indices=indices, periodic=True, out=angles)

        return angles


class Dihedral(Metric):
    def __init__(self, *args, **kwargs):
        super(Dihedral, self).__init__(*args, **kwargs)
        self.ensure_data_type.exception_class = DihedralException

    def __mdtraj_paramaeter_compatibility_check__(self, xyz, indices, opt):
        super(Dihedral, self).__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt,
                                                                        atom_amount=4)

    def __calculate_angle__(self, xyz, indices, periodic, out):
        """SOURCE: mdTraj
           MODIFICATION: displacement function
           Compute the dihedral angles of traj for the atom indices in indices.

            Parameters
            ----------
            xyz : np.ndarray, shape=(num_frames, num_atoms, 3), dtype=float
                The XYZ coordinates of a trajectory
            indices : np.ndarray, shape=(num_dihedrals, 4), dtype=int
                Atom indices to compute dihedrals.
            periodic : bool, default=True
                If `periodic` is True and the trajectory contains unitcell
                information, we will treat dihedrals that cross periodic images
                using the minimum image convention.

            Returns
            -------
            dih : np.ndarray, shape=(num_dihedrals), dtype=float
                dih[i,j] gives the dihedral angle at traj[i] correponding to indices[j].

            """
        self.ensure_data_type.ensure_numpy_array(parameter=out, parameter_name='out',
                                                 shape=(xyz.shape[0], indices.shape[0]), desired_dtype=np.float32)

        ix10 = indices[0, [0, 1]]
        xyz10 = xyz[:, ix10]

        ix21 = indices[0, [1, 2]]
        xyz21 = xyz[:, ix21]

        ix32 = indices[0, [2, 3]]
        xyz32 = xyz[:, ix32]

        indices = np.array([[0, 1]], dtype=np.int32)

        displacement = Displacement(periodic=periodic, unit_cell_angles=self.unit_cell_angles,
                                    unit_cell_vectors=self.unit_cell_vectors)
        b1 = displacement.get_compatibility_layer(xyz=xyz10, indices=indices, periodic=periodic, opt=False)
        b2 = displacement.get_compatibility_layer(xyz=xyz21, indices=indices, periodic=periodic, opt=False)
        b3 = displacement.get_compatibility_layer(xyz=xyz32, indices=indices, periodic=periodic, opt=False)

        c1 = np.cross(b2, b3)
        c2 = np.cross(b1, b2)

        p1 = (b1 * c1).sum(-1)
        p1 *= (b2 * b2).sum(-1) ** 0.5
        p2 = (c1 * c2).sum(-1)

        return np.arctan2(p1, p2, out)

    def __calculate_no_pbc__(self, xyz, indices, opt):
        self.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt)

        dihedrals = np.zeros((xyz.shape[0], indices.shape[0]), dtype=np.float32)

        if opt:
            _dihedral(xyz, indices, dihedrals)
        else:
            self.__calculate_angle__(xyz=xyz, indices=indices, out=dihedrals, periodic=False)

        return dihedrals

    def __calculate_minimal_image_convention__(self, xyz, indices, opt):
        self.__mdtraj_paramaeter_compatibility_check__(xyz=xyz, indices=indices, opt=opt)

        dihedrals = np.zeros((xyz.shape[0], indices.shape[0]), dtype=np.float32)

        box = ensure_type(self.unit_cell_vectors, dtype=np.float32, ndim=3, name='unitcell_vectors',
                          shape=(len(xyz), 3, 3), warn_on_cast=True)

        if opt:
            orthogonal = np.allclose(self.unit_cell_angles, 90)
            _dihedral_mic(xyz, indices, box.transpose(0, 2, 1).copy(), dihedrals, orthogonal)
        else:
            self.__calculate_angle__(xyz=xyz, indices=indices, periodic=True, out=dihedrals)

        return dihedrals
