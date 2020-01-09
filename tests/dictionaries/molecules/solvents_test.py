from unittest import TestCase


class TestWater(TestCase):
    def setUp(self) -> None:
        from yeti.dictionaries.molecules.solvents import Water
        self.dict = Water()

    def test_init_(self):
        self.assertTupleEqual(self.dict.covalent_bonds, (('O', 'H1'), ('O', 'H2')))
        self.assertDictEqual(self.dict.acceptors_dictionary, {'O': 2})
        self.assertDictEqual(self.dict.donors_dictionary, {'H1': 1, 'H2': 1})
