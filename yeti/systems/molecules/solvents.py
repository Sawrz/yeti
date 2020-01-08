import yeti.dictionaries.molecules.solvents as solvent_dictionaries
from .molecules import ThreeAtomsMolecule, MoleculeException
from ..building_blocks import EnsureDataTypes


class Water(ThreeAtomsMolecule):
    def __init__(self, residue, *args, **kwargs):
        # TODO: add right exception class
        self.ensure_data_type = EnsureDataTypes(exception_class=MoleculeException)
        self.ensure_data_type.ensure_residue(parameter=residue, parameter_name='residue')

        super(Water, self).__init__(residues=(residue,), molecule_name=residue.name, *args, **kwargs)

        self.residue = residue
        self._internal_id = self.residue.subsystem_index
        self._structure_file_id = self.residue.structure_file_index

        self.dictionary = solvent_dictionaries.Water()

    def get_distance(self):
        atom_positions = self.__get_atom_ids__(atom_names=('HW1', 'HW2'),
                                               residue_ids=2 * (self._internal_id,))
        self.distances = super(Water, self).get_distance(*atom_positions, store_result=False, opt=True)

    def get_angle(self):
        atom_positions = self.__get_atom_ids__(atom_names=('HW1', 'OW', 'HW2'),
                                               residue_ids=3 * (self._internal_id,))
        self.angles = super(Water, self).get_angle(*atom_positions, store_result=False, opt=True)



