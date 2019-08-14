import numpy as np

import yeti.dictionaries.molecules.biomolecules as biomolecule_dictionaries
from yeti.systems.molecules.molecules import BioMolecule


class NucleicAcidException(Exception):
    pass


class NucleicAcid(BioMolecule):
    def __init__(self, *args, **kwargs):
        super(NucleicAcid, self).__init__(*args, **kwargs)
        self.ensure_data_type.exception_class = NucleicAcidException

    def get_p_to_p_distance(self, residue_id_01, residue_id_02):
        # TODO: raise Exception if residue doesn't contain P-atom
        atom_positions = self.__get_atom_ids__(atom_name_residue_ids=(('P', residue_id_01), ('P', residue_id_02)))
        self.get_distance(*atom_positions, store_result=True, opt=True)

    # TODO: implement multi-thread
    def get_all_p_to_p_distances(self):
        pass

    def get_dihedral(self, dihedral_name, residue_id):
        atom_names, residue_ids, latex = self.dictionary.dihedral_angles_dictionary[dihedral_name]
        residue_ids = tuple(np.array(residue_ids) + residue_id)

        atom_positions = self.__get_atom_ids__(atom_names=atom_names, residue_ids=residue_ids)
        dihedral_angles = super(NucleicAcid, self).get_dihedral(*atom_positions, store_result=False, opt=True)

        key = '{dihedral_name}_{residue_id:03d}'.format(dihedral_name=dihedral_name, residue_id=residue_id)
        self.dihedral_angles[key] = dihedral_angles

    # TODO: implement multi-thread
    def get_all_dihedral_angles(self):
        pass


class RNA(NucleicAcid):
    def __init__(self, *args, **kwargs):
        super(RNA, self).__init__(*args, **kwargs)
        self.dictionary = biomolecule_dictionaries.RNA


class DNA(NucleicAcid):
    def __init__(self, *args, **kwargs):
        super(RNA, self).__init__(*args, **kwargs)
        self.dictionary = biomolecule_dictionaries.DNA
