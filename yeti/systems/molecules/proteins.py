import yeti.dictionaries.molecules.biomolecules as biomolecule_dictionaries
from yeti.systems.molecules.molecules import BioMolecule, BioMoleculeException


class ProteinException(Exception):
    pass


class Protein(BioMolecule):
    def __init__(self, *args, **kwargs):
        super(Protein, self).__init__(*args, **kwargs)
        self.ensure_data_type.exception_class = ProteinException
        self.dictionary = biomolecule_dictionaries.Protein()