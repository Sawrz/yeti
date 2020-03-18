from itertools import combinations

import yeti.dictionaries.molecules.biomolecules as biomolecule_dictionaries
from yeti.systems.molecules.molecules import BioMolecule, BioMoleculeException


class NucleicAcidException(Exception):
    pass


class NucleicAcid(BioMolecule):
    def __init__(self, *args, **kwargs):
        super(NucleicAcid, self).__init__(*args, **kwargs)
        self.ensure_data_type.exception_class = NucleicAcidException
        self.dictionary = biomolecule_dictionaries.NucleicAcid()

    def get_p_to_p_distance(self, residue_id_01, residue_id_02):
        self.ensure_data_type.ensure_integer(parameter=residue_id_01, parameter_name='residue_id_01')
        self.ensure_data_type.ensure_integer(parameter=residue_id_02, parameter_name='residue_id_02')

        # TODO: raise Exception if residue doesn't contain P-atom
        atom_positions = self.__get_atom_ids__(atom_names=('P', 'P'), residue_ids=(residue_id_01, residue_id_02))
        self.get_distance(*atom_positions, store_result=True, opt=True)

    # TODO: implement multi-thread
    def get_all_p_to_p_distances(self):
        for residue_01, residue_02 in combinations(self.residues, 2):
            if residue_01.subsystem_index >= residue_02.subsystem_index:
                continue

            if 'P' in residue_01.sequence and 'P' in residue_02.sequence:
                self.get_p_to_p_distance(residue_id_01=residue_01.subsystem_index,
                                         residue_id_02=residue_02.subsystem_index)

    # TODO: unit test for this function
    def __check_fourth_base__(self, residue, residue_name='TGIF'):
        if residue.name == residue_name:
            return True
        else:
            return False

    def get_dihedral(self, dihedral_name, residue_id):
        self.ensure_data_type.ensure_string(parameter=dihedral_name, parameter_name='dihedral_name')
        self.ensure_data_type.ensure_integer(parameter=residue_id, parameter_name='residue_id')

        residue = self.residues[residue_id]

        if dihedral_name == 'chi' and (residue.name == 'A' or residue.name == 'G'):
            dict_dihedral_name = 'chi_pu'
        elif dihedral_name == 'chi' and (residue.name == 'C' or self.__check_fourth_base__(residue=residue)):
            dict_dihedral_name = 'chi_py'
        else:
            dict_dihedral_name = dihedral_name

        atom_names, residue_ids, latex = self.dictionary.dihedral_angles_dictionary[dict_dihedral_name]
        residue_ids = tuple(res_id + residue_id for res_id in residue_ids)

        atom_positions = self.__get_atom_ids__(atom_names=atom_names, residue_ids=residue_ids)
        dihedral_angles = super(NucleicAcid, self).get_dihedral(*atom_positions, store_result=False, opt=True)

        key = '{dihedral_name}_{residue_id:03d}'.format(dihedral_name=dihedral_name, residue_id=residue_id)
        self.dihedral_angles[key] = dihedral_angles

    # TODO: implement multi-thread
    def get_all_dihedral_angles(self):
        for residue in self.residues:
            for key in self.dictionary.dihedral_angles_dictionary.keys():
                try:
                    # TODO: do chi_pu or chi_py based on base name and not twice
                    if key == 'chi_pu' or key == 'chi_py':
                        dihedral_name = 'chi'
                    else:
                        dihedral_name = key

                    self.get_dihedral(dihedral_name=dihedral_name, residue_id=residue.subsystem_index)
                except BioMoleculeException as e:
                    no_atom_exist = e.args[0] == 'Atom does not exist in this residue.'
                    residue_id_too_high = e.args[0] == 'Atom requested residue id is higher than available residues.'

                    if no_atom_exist or residue_id_too_high:
                        msg = '{dihedral_name}_{residue_id:03d} not found'.format(dihedral_name=key,
                                                                                  residue_id=residue.subsystem_index)
                        print(msg)
                        continue
                    else:
                        raise e


class RNA(NucleicAcid):
    # TODO: Test for Uracil and Guanine
    def __init__(self, *args, **kwargs):
        super(RNA, self).__init__(*args, **kwargs)
        self.dictionary = biomolecule_dictionaries.RNA()

    # TODO: unit test
    def __check_fourth_base__(self, residue):
        return super(RNA, self).__check_fourth_base__(residue=residue, residue_name='U')


class DNA(NucleicAcid):
    # TODO test for Thymine and Guanine
    def __init__(self, *args, **kwargs):
        super(DNA, self).__init__(*args, **kwargs)
        self.dictionary = biomolecule_dictionaries.DNA()

    # TODO: unit test
    def __check_fourth_base__(self, residue):
        return super(DNA, self).__check_fourth_base__(residue=residue, residue_name='T')
