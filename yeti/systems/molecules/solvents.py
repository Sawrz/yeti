from .molecules import ThreeAtomsMolecule, MoleculeException, FourAtomsPlusMolecule
from ..building_blocks import EnsureDataTypes


class Water(ThreeAtomsMolecule):
    # TODO: Get rid off residues or residue parameter
    def __init__(self, residue, molecule_name, *args, **kwargs):
        # TODO: add right exception class
        self.ensure_data_type = EnsureDataTypes(exception_class=MoleculeException)
        self.ensure_data_type.ensure_residue(parameter=residue, parameter_name='residue')

        super(Water, self).__init__(residues=(residue,), molecule_name=molecule_name, *args, **kwargs)

        self.residue = residue
        self._internal_id = self.residue.subsystem_index
        self._structure_file_id = self.residue.structure_file_index

    def get_distance(self):
        atom_positions = self.__get_atom_ids__(atom_names=('HW1', 'HW2'),
                                               residue_ids=2 * (self._internal_id,))
        self.distances = super(Water, self).get_distance(*atom_positions, store_result=False, opt=True)

    def get_angle(self):
        atom_positions = self.__get_atom_ids__(atom_names=('HW1', 'OW', 'HW2'),
                                               residue_ids=3 * (self._internal_id,))
        self.angles = super(Water, self).get_angle(*atom_positions, store_result=False, opt=True)


# TODO: Avoid inheritage of FourAtomsPlusMolecule class
# TODO: think if class really necessary (maybe the methods are more useful in System class)
class Solvent(FourAtomsPlusMolecule):
    def __init__(self, solvent_molecules, solvent_name, *args, **kwargs):
        self.molecules = solvent_molecules
        self.solvent_name = solvent_name

        super(Solvent, self).__init__(residues=tuple(molecule.residue for molecule in self.molecules),
                                      molecule_name=solvent_name, *args, **kwargs)
        del self.molecule_name

    def get_distance(self, atom_name_01, residue_id_01, atom_name_02, residue_id_02):
        atom_positions = self.__get_atom_ids__(atom_names=(atom_name_01, atom_name_02),
                                               residue_ids=(residue_id_01, residue_id_02))

        super(Solvent, self).get_distance(*atom_positions, store_result=True, opt=False)

    def get_angle(self, atom_name_01, residue_id_01, atom_name_02, residue_id_02, atom_name_03, residue_id_03):
        atom_positions = self.__get_atom_ids__(atom_names=(atom_name_01, atom_name_02, atom_name_03),
                                               residue_ids=(residue_id_01, residue_id_02, residue_id_03))

        super(Solvent, self).get_angle(*atom_positions, store_result=True, opt=True)

    def get_dihedral(self, atom_name_01, residue_id_01, atom_name_02, residue_id_02, atom_name_03, residue_id_03,
                     atom_name_04, residue_id_04):
        atom_positions = self.__get_atom_ids__(atom_names=(atom_name_01, atom_name_02, atom_name_03, atom_name_04),
                                               residue_ids=(residue_id_01, residue_id_02, residue_id_03, residue_id_04))

        super(Solvent, self).get_dihedral(*atom_positions, store_result=True, opt=True)
