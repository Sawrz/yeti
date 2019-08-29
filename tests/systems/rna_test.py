from tests.systems.nucleic_acid_test import NucleicAcidInitTest, NucleicAcidGetPTOPDistance, NucleicAcidGetDihedral, \
    NucleicAcidTest


class RNAInitTest(NucleicAcidInitTest):
    def setUp(self) -> None:
        from yeti.systems.molecules.nucleic_acids import RNA
        from yeti.dictionaries.molecules import biomolecules

        super(RNAInitTest, self).setUp()
        self.nucleic_acid = RNA
        self.dictionary = biomolecules.RNA


class RNATest(NucleicAcidTest):
    def setUp(self) -> None:
        from yeti.systems.molecules.nucleic_acids import RNA, NucleicAcidException

        unit_cell_angles, unit_cell_vectors = self.build_unit_cell_angles_and_vectors(number_of_frames=2)
        residues = self.setUpResidues()

        box_information = dict(
            dict(periodic=True, unit_cell_angles=unit_cell_angles, unit_cell_vectors=unit_cell_vectors))

        simulation_information = dict(number_of_frames=3)
        hydrogen_bond_information = dict(distance_cutoff=0.25, angle_cutoff=2.0)

        self.nucleic_acid = RNA(residues=residues, molecule_name='test', box_information=box_information,
                                simulation_information=simulation_information,
                                hydrogen_bond_information=hydrogen_bond_information)
        self.exception = NucleicAcidException


class RNAGetPTOPDistance(RNATest, NucleicAcidGetPTOPDistance):
    pass


class RNAGetDihedral(RNATest, NucleicAcidGetDihedral):
    def setUpResidues(self):
        residue_01, residue_02, residue_03, residue_04 = super(RNAGetDihedral, self).setUpResidues()

        residue_01.name = 'U'
        residue_02.name = 'G'
        residue_03.name = 'U'
        residue_04.name = 'U'

        return residue_01, residue_02, residue_03, residue_04
