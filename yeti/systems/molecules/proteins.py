import yeti.dictionaries.molecules.biomolecules as biomolecule_dictionaries
from yeti.systems.molecules.molecules import BioMolecule, BioMoleculeException


class ProteinException(BioMoleculeException):
    pass


class Protein(BioMolecule):
    def __init__(self, *args, **kwargs):
        super(Protein, self).__init__(*args, **kwargs)
        self.ensure_data_type.exception_class = ProteinException
        self.dictionary = biomolecule_dictionaries.Protein()

    def get_distance(self, distance_name, residue_id_01, residue_id_02):
        self.ensure_data_type.ensure_integer(parameter=residue_id_01, parameter_name='residue_id_01')
        self.ensure_data_type.ensure_integer(parameter=residue_id_02, parameter_name='residue_id_02')
        self.ensure_data_type.ensure_string(parameter=distance_name, parameter_name='distance_name')

        atom_names, latex = self.dictionary.distances_dictionary[distance_name]

        atom_positions = self.__get_atom_ids__(atom_names=(atom_names[0], atom_names[1]),
                                               residue_ids=(residue_id_01, residue_id_02))
        super(Protein, self).get_distance(*atom_positions, store_result=True, opt=True)

    def get_dihedral(self, dihedral_name, residue_id):
        self.ensure_data_type.ensure_string(parameter=dihedral_name, parameter_name='dihedral_name')
        self.ensure_data_type.ensure_integer(parameter=residue_id, parameter_name='residue_id')

        atom_names, residue_ids, latex = self.dictionary.dihedral_angles_dictionary[dihedral_name]
        residue_ids = tuple(res_id + residue_id for res_id in residue_ids)

        atom_positions = self.__get_atom_ids__(atom_names=atom_names, residue_ids=residue_ids)
        print()

        dihedral_angles = super(Protein, self).get_dihedral(*atom_positions, store_result=False, opt=True)

        key = '{dihedral_name}_{residue_id:03d}'.format(dihedral_name=dihedral_name, residue_id=residue_id)
        self.dihedral_angles[key] = dihedral_angles

    def get_all_dihedral_angles(self):
        for residue in self.residues:
            for dihedral_name in self.dictionary.dihedral_angles_dictionary.keys():
                try:
                    self.get_dihedral(dihedral_name=dihedral_name, residue_id=residue.subsystem_index)
                except BioMoleculeException as e:
                    no_atom_exist = e.args[0] == 'Atom does not exist in this residue.'
                    residue_id_too_high = e.args[0] == 'Atom requested residue id is higher than available residues.'

                    if no_atom_exist or residue_id_too_high:
                        msg = '{dihedral_name}_{residue_id:03d} not found'.format(dihedral_name=dihedral_name,
                                                                                  residue_id=residue.subsystem_index)
                        print(msg)
                        continue
                    else:
                        raise e
