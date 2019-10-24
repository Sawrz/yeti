import unittest

from tests.systems.nucleic_acid_test import NucleicAcidTestCase, TestNucleicAcidStandardMethods, \
    TestNucleicAcidDistanceMethods, TestNucleicAcidDihedralAngleMethods, NucleicAcidExceptionsTestCase


class RiboNucleicAcidTestCase(NucleicAcidTestCase):
    def setUp(self) -> None:
        from yeti.systems.molecules.nucleic_acids import RNA

        super(RiboNucleicAcidTestCase, self).setUp()

        self.nucleic_acid = RNA(residues=self.residues, molecule_name=self.molecule_name,
                                box_information=self.box_information,
                                simulation_information=self.simulation_information, periodic=True,
                                hydrogen_bond_information=self.hydrogen_bond_information)


class TestRiboNucleicAcidStandardMethods(RiboNucleicAcidTestCase, TestNucleicAcidStandardMethods):
    def setUp(self) -> None:
        from yeti.systems.molecules.nucleic_acids import RNA
        from yeti.dictionaries.molecules import biomolecules

        super(TestRiboNucleicAcidStandardMethods, self).setUp()

        self.dictionary = biomolecules.RNA
        self.nucleic_acid = RNA(residues=self.residues, molecule_name=self.molecule_name,
                                box_information=self.box_information,
                                simulation_information=self.simulation_information, periodic=True,
                                hydrogen_bond_information=self.hydrogen_bond_information)


class TestRiboNucleicDistanceMethods(RiboNucleicAcidTestCase, TestNucleicAcidDistanceMethods):
    pass


class TestRiboNucleicDihedralAngleMethods(RiboNucleicAcidTestCase, TestNucleicAcidDihedralAngleMethods):
    def setUpResidues(self) -> None:
        super(TestRiboNucleicDihedralAngleMethods, self).setUpResidues()

        self.residues[0].name = 'U'
        self.residues[1].name = 'G'
        self.residues[2].name = 'U'
        self.residues[3].name = 'U'


class RiboNucleicAcidExceptionsTestCase(RiboNucleicAcidTestCase, NucleicAcidExceptionsTestCase):
    pass


class TestDistanceMethodExceptions(RiboNucleicAcidExceptionsTestCase):
    pass


class TestDihedralAngleMethodExceptions(RiboNucleicAcidExceptionsTestCase):
    pass


if __name__ == '__main__':
    unittest.main()
