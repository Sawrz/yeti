import numpy as np
from mdtraj.geometry._geometry import _dihedral_mic, _dihedral
from mdtraj.utils.validation import ensure_type

from yeti.get_features.distances import Displacement
from yeti.get_features.metric import Metric


class Dihedral(Metric):
    def __calculate_dihedral__(self, xyz, indices, periodic, out):
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
        ix10 = indices[:, [0, 1]]
        ix21 = indices[:, [1, 2]]
        ix32 = indices[:, [2, 3]]

        displacement = Displacement(periodic=periodic, box_vectors=self.box_vectors,
                                    unit_cell_angles=self.unit_cell_angles,
                                    unit_cell_vectors=self.unit_cell_vectors)
        b1 = displacement.get_displacement_compatibility_layer(xyz, ix10, periodic=periodic, opt=False)
        b2 = displacement.get_displacement_compatibility_layer(xyz, ix21, periodic=periodic, opt=False)
        b3 = displacement.get_displacement_compatibility_layer(xyz, ix32, periodic=periodic, opt=False)

        c1 = np.cross(b2, b3)
        c2 = np.cross(b1, b2)

        p1 = (b1 * c1).sum(-1)
        p1 *= (b2 * b2).sum(-1) ** 0.5
        p2 = (c1 * c2).sum(-1)

        return np.arctan2(p1, p2, out)

    def __calculate_no_pbc__(self, xyz, indices, opt):
        dihedrals = np.zeros((xyz.shape[0], indices.shape[0]), dtype=np.float32)

        if opt:
            _dihedral(xyz, indices, dihedrals)
        else:
            self.__calculate_dihedral__(xyz=xyz, indices=indices, out=dihedrals, periodic=False)

        return dihedrals

    def __calculate_minimal_image_convention__(self, xyz, indices, opt):
        dihedrals = np.zeros((xyz.shape[0], indices.shape[0]), dtype=np.float32)

        box = ensure_type(self.unit_cell_vectors, dtype=np.float32, ndim=3, name='unitcell_vectors',
                          shape=(len(xyz), 3, 3), warn_on_cast=True)

        if opt:
            orthogonal = np.allclose(self.unit_cell_angles, 90)
            _dihedral_mic(xyz, indices, box.transpose(0, 2, 1).copy(), dihedrals, orthogonal)
        else:
            self.__calculate_dihedral__(xyz=xyz, indices=indices, periodic=True, out=dihedrals)

        return dihedrals

    def get_dihedral(self, atom_name_residue_pair_01, atom_name_residue_pair_02, atom_name_residue_pair_03,
                     atom_name_residue_pair_04, opt=False):
        name_residue_pairs = (atom_name_residue_pair_01, atom_name_residue_pair_02, atom_name_residue_pair_03,
                              atom_name_residue_pair_04)
        atoms = self.__get_atoms__(*name_residue_pairs)

        return self.__calculate__(atoms=atoms, opt=opt, amount=4)
