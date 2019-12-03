import unittest

from tests.systems.nucleic_acid_test import NucleicAcidTestCase, TestNucleicAcidStandardMethods, \
    TestNucleicAcidDistanceMethods, TestNucleicAcidDihedralAngleMethods, NucleicAcidExceptionsTestCase


class DeoxyriboNucleicAcidTestCase(NucleicAcidTestCase):
    def setUp(self) -> None:
        from yeti.systems.molecules.nucleic_acids import DNA

        super(DeoxyriboNucleicAcidTestCase, self).setUp()

        self.molecule = DNA(residues=self.residues, molecule_name=self.molecule_name,
                            box_information=self.box_information, periodic=True)


class TestDeoxyriboNucleicAcidStandardMethods(DeoxyriboNucleicAcidTestCase, TestNucleicAcidStandardMethods):
    def setUp(self) -> None:
        from yeti.systems.molecules.nucleic_acids import DNA
        from yeti.dictionaries.molecules import biomolecules

        super(TestDeoxyriboNucleicAcidStandardMethods, self).setUp()

        self.dictionary = biomolecules.DNA
        self.molecule = DNA(residues=self.residues, molecule_name=self.molecule_name,
                            box_information=self.box_information, periodic=True)


class TestDeoxyriboNucleicDistanceMethods(DeoxyriboNucleicAcidTestCase, TestNucleicAcidDistanceMethods):
    pass


class TestDeoxyriboNucleicDihedralAngleMethods(DeoxyriboNucleicAcidTestCase, TestNucleicAcidDihedralAngleMethods):
    def setUpResidues(self) -> None:
        super(TestDeoxyriboNucleicDihedralAngleMethods, self).setUpResidues()

        self.residues[0].name = 'T'
        self.residues[1].name = 'G'
        self.residues[2].name = 'T'
        self.residues[3].name = 'T'


class DeoxyriboNucleicAcidExceptionsTestCase(DeoxyriboNucleicAcidTestCase, NucleicAcidExceptionsTestCase):
    pass


class TestDistanceMethodExceptions(DeoxyriboNucleicAcidExceptionsTestCase):
    pass


class TestDihedralAngleMethodExceptions(DeoxyriboNucleicAcidExceptionsTestCase):
    pass


if __name__ == '__main__':
    unittest.main()
