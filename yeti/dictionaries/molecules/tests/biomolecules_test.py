from unittest import TestCase


class TestBiomolecule(TestCase):
    def test_set_bonds_between_residues(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference = ("A", "B")

        bio_mol = Biomolecule()
        bio_mol.set_bonds_between_residues("A", "B")
        result = bio_mol.bonds_between_residues

        self.assertTupleEqual(reference, result)

    def test_set_bonds_between_residues_exception_false_parameter_type_1(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.set_bonds_between_residues(["A"], "B")

        self.assertTrue("Parameter atom_1 and atom_2 need to be strings." == str(context.exception))

    def test_set_bonds_between_residues_exception_false_parameter_type_2(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.set_bonds_between_residues("A", 1)

        self.assertTrue("Parameter atom_1 and atom_2 need to be strings." == str(context.exception))

    def test_update_abbreviations_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference = {"argenine": "ARG",
                     "lysine": "LYS"}

        bio_mol = Biomolecule()
        bio_mol.update_abbreviation_dictionary(abbreviations=reference)
        result = bio_mol.abbreviation_dictionary

        self.assertDictEqual(reference, result)

    def test_update_abbreviations_dictionary_no_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        abbreviations = ["eggs", "butter"]

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_abbreviation_dictionary(abbreviations=abbreviations)

        self.assertTrue("Parameter abbreviations need to be a dictionary." == str(context.exception))

    def test_update_abbreviations_dictionary_wrong_key_type(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        abbreviations = {"ADE": "A",
                         7: "T",
                         "GUA": "G"}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_abbreviation_dictionary(abbreviations=abbreviations)

        self.assertTrue("Keys of abbreviations need to be strings." == str(context.exception))

    def test_update_abbreviations_dictionary_wrong_value_type(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        abbreviations = {"ADE": "A",
                         "THY": 7,
                         "GUA": "G"}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_abbreviation_dictionary(abbreviations=abbreviations)

        self.assertTrue("Values of abbreviations need to be strings." == str(context.exception))

    def test_update_dihedral_angle_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference = {"alpha": (("O3\'", "P", "O5\'", "C5\'"), (-1, 0, 0, 0), r"$\alpha$"),
                     "beta": (("P", "O5\'", "C5\'", "C4\'"), (0, 0, 0, 0), r"$\beta$")}

        bio_mol = Biomolecule()
        bio_mol.update_dihedral_angle_dictionary(dihedral_angles=reference)
        result = bio_mol.dihedral_angles_dictionary

        self.assertDictEqual(reference, result)

    def test_update_dihedral_angle_dictionary_no_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        dih_angles = [7, 3, 4, 5]

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_dihedral_angle_dictionary(dihedral_angles=dih_angles)

        self.assertTrue("Parameter dihedral_angles need to be a dictionary." == str(context.exception))

    def test_update_dihedral_angle_dictionary_wrong_key_type(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        dih_angles = {1: (("O3\'", "P", "O5\'", "C5\'"), (-1, 0, 0, 0), r"$\alpha$")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_dihedral_angle_dictionary(dihedral_angles=dih_angles)

        self.assertTrue("Keys of dihedral_angles need to be strings." == str(context.exception))

    def test_update_dihedral_angle_dictionary_wrong_value_type(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        dih_angles = {"alpha": [("O3\'", "P", "O5\'", "C5\'"), (-1, 0, 0, 0), r"$\alpha$"]}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_dihedral_angle_dictionary(dihedral_angles=dih_angles)

        self.assertTrue("Values of dihedral_angles need to be tuples." == str(context.exception))

    def test_update_dihedral_angle_dictionary_first_element_no_tuple(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        dih_angles = {"alpha": (["O3\'", "P", "O5\'", "C5\'"], (-1, 0, 0, 0), r"$\alpha$")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_dihedral_angle_dictionary(dihedral_angles=dih_angles)

        self.assertTrue("First Element need to be a tuple." == str(context.exception))

    def test_update_dihedral_angle_dictionary_first_element_not_enough_elements(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        dih_angles = {"alpha": (("O3\'", "P", "O5\'"), (-1, 0, 0, 0), r"$\alpha$")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_dihedral_angle_dictionary(dihedral_angles=dih_angles)

        self.assertTrue("Exactly four strings for first element allowed." == str(context.exception))

    def test_update_dihedral_angle_dictionary_first_element_to_many_elements(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        dih_angles = {"alpha": (("O3\'", "P", "O5\'", "C5\'", "C1\'"), (-1, 0, 0, 0), r"$\alpha$")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_dihedral_angle_dictionary(dihedral_angles=dih_angles)

        self.assertTrue("Exactly four strings for first element allowed." == str(context.exception))

    def test_update_dihedral_angle_dictionary_first_element_no_strings(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        dih_angles = {"alpha": (("O3\'", 6, "O5\'", "C5\'"), (-1, 0, 0, 0), r"$\alpha$")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_dihedral_angle_dictionary(dihedral_angles=dih_angles)

        self.assertTrue("First elements tuple can only contain strings." == str(context.exception))

    def test_update_dihedral_angle_dictionary_second_element_no_tuple(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        dih_angles = {"alpha": (("O3\'", "P", "O5\'", "C5\'"), [-1, 0, 0, 0], r"$\alpha$")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_dihedral_angle_dictionary(dihedral_angles=dih_angles)

        self.assertTrue("Second Element need to be a tuple." == str(context.exception))

    def test_update_dihedral_angle_dictionary_second_element_not_enough_elements(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        dih_angles = {"alpha": (("O3\'", "P", "O5\'", "C5\'"), (-1, 0, 0), r"$\alpha$")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_dihedral_angle_dictionary(dihedral_angles=dih_angles)

        self.assertTrue("Exactly four integers for second element allowed." == str(context.exception))

    def test_update_dihedral_angle_dictionary_second_element_too_much_elements(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        dih_angles = {"alpha": (("O3\'", "P", "O5\'", "C5\'"), (-1, 0, 0, 0, 0), r"$\alpha$")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_dihedral_angle_dictionary(dihedral_angles=dih_angles)

        self.assertTrue("Exactly four integers for second element allowed." == str(context.exception))

    def test_update_dihedral_angle_dictionary_second_element_no_integers(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        dih_angles = {"alpha": (("O3\'", "P", "O5\'", "C5\'"), (-1, 0, "2", 0), r"$\alpha$")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_dihedral_angle_dictionary(dihedral_angles=dih_angles)

        self.assertTrue("Second elements tuple can only contain integers." == str(context.exception))

    def test_update_dihedral_angle_dictionary_third_element_no_string(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        dih_angles = {"alpha": (("O3\'", "P", "O5\'", "C5\'"), (-1, 0, 0, 0), ["stuff"])}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_dihedral_angle_dictionary(dihedral_angles=dih_angles)

        self.assertTrue("Third element need to be a string." == str(context.exception))

    def test_update_distances_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference = {"PToP": (("P", "P"), "P to P"),
                     "PToMG": (("P", "MG"), "P to MG")}

        bio_mol = Biomolecule()
        bio_mol.update_distances_dictionary(distances=reference)
        result = bio_mol.distances_dictionary

        self.assertDictEqual(reference, result)

    def test_update_distances_dictionary_no_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        distances = (("P", "P"), "P to P")

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_distances_dictionary(distances=distances)

        self.assertTrue("Parameter distance need to be a dictionary." == str(context.exception))

    def test_update_distances_dictionary_wrong_key_type(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        distances = {"PToMg": (("P", "Mg"), "P to Mg"),
                     6: (("P", "P"), "P to P")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_distances_dictionary(distances=distances)

        self.assertTrue("Keys of distances need to be strings." == str(context.exception))

    def test_update_distances_dictionary_wrong_value_type(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        distances = {"PToMg": (("P", "Mg"), "P to Mg"),
                     "PToP": [("P", "P"), "P to P"]}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_distances_dictionary(distances=distances)

        self.assertTrue("Values of distances need to be tuples." == str(context.exception))

    def test_update_distances_dictionary_first_element_no_tuple(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        distances = {"PToP": (["P", "P"], "P to P")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_distances_dictionary(distances=distances)

        self.assertTrue("First element need to be a tuple." == str(context.exception))

    def test_update_distances_dictionary_first_element_too_much_entries(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        distances = {"PToP": (("P", "P", "Mg"), "P to P")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_distances_dictionary(distances=distances)

        self.assertTrue("Exactly two strings for first element allowed." == str(context.exception))

    def test_update_distances_dictionary_first_element_not_enough_entries(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        distances = {"PToP": (("P",), "P to P")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_distances_dictionary(distances=distances)

        self.assertTrue("Exactly two strings for first element allowed." == str(context.exception))

    def test_update_distances_dictionary_first_element_contains_no_string(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        distances = {"PToP": (("P", 3), "P to P")}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_distances_dictionary(distances=distances)

        self.assertTrue("First element only contains strings." == str(context.exception))

    def test_update_distances_dictionary_second_element_no_string(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        distances = {"PToP": (("P", "P"), 5)}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_distances_dictionary(distances=distances)

        self.assertTrue("Second element need to be a string." == str(context.exception))

    def test_update_backbone_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference = {"start": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                     "end": (("P", "OP1"), ("P", "O5\'"))}

        bio_mol = Biomolecule()
        bio_mol.update_backbone_bonds_dictionary(backbone_bonds=reference)
        result = bio_mol.backbone_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_update_backbone_bonds_dictionary_no_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = (("P", "OP1"), ("P", "OP2"), ("P", "O5\'"))

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_backbone_bonds_dictionary(backbone_bonds=backbone_bonds)

        self.assertTrue("Parameter backbone_bonds need to be a dictionary." == str(context.exception))

    def test_update_backbone_bonds_dictionary_wrong_key_types(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                          0: (("P", "OP1"), ("P", "O5\'"))}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_backbone_bonds_dictionary(backbone_bonds=backbone_bonds)

        self.assertTrue("Keys of backbone_bonds need to be strings." == str(context.exception))

    def test_update_backbone_bonds_dictionary_wrong_value_types(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                          "end": [("P", "OP1"), ("P", "O5\'")]}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_backbone_bonds_dictionary(backbone_bonds=backbone_bonds)

        self.assertTrue("Values of backbone_bonds need to be tuples." == str(context.exception))

    def test_update_backbone_bonds_dictionary_tuple_elements_no_tuples(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                          "end": (("P", "OP1"), ["P", "O5\'"])}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_backbone_bonds_dictionary(backbone_bonds=backbone_bonds)

        self.assertTrue("Bond type tuple only contains other tuples." == str(context.exception))

    def test_update_backbone_bonds_dictionary_tuple_elements_no_strings(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                          "end": (("P", "OP1"), ("8", 7))}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_backbone_bonds_dictionary(backbone_bonds=backbone_bonds)

        self.assertTrue("Elements of bond tuples need to be strings." == str(context.exception))

    def test_update_backbone_bonds_dictionary_tuple_elements_too_many(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2", "Popeye"), ("P", "O5\'")),
                          "end": (("P", "OP1"), ("P", "O5\'"))}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_backbone_bonds_dictionary(backbone_bonds=backbone_bonds)

        self.assertTrue("Exactly two strings for bond tuple are allowed." == str(context.exception))

    def test_update_backbone_bonds_dictionary_tuple_elements_not_enough(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                          "end": (("P", "OP1"), ("O5\'",))}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_backbone_bonds_dictionary(backbone_bonds=backbone_bonds)

        self.assertTrue("Exactly two strings for bond tuple are allowed." == str(context.exception))

    def test_update_termini_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference = {"start": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                     "end": (("P", "OP1"), ("P", "O5\'"))}

        bio_mol = Biomolecule()
        bio_mol.update_termini_bonds_dictionary(termini_bonds=reference)
        result = bio_mol.backbone_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_update_termini_bonds_dictionary_no_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = (("P", "OP1"), ("P", "OP2"), ("P", "O5\'"))

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_termini_bonds_dictionary(termini_bonds=backbone_bonds)

        self.assertTrue("Parameter termini_bonds need to be a dictionary." == str(context.exception))

    def test_update_termini_bonds_dictionary_wrong_key_types(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                          0: (("P", "OP1"), ("P", "O5\'"))}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_termini_bonds_dictionary(termini_bonds=backbone_bonds)

        self.assertTrue("Keys of termini_bonds need to be strings." == str(context.exception))

    def test_update_termini_bonds_dictionary_wrong_value_types(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                          "end": [("P", "OP1"), ("P", "O5\'")]}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_termini_bonds_dictionary(termini_bonds=backbone_bonds)

        self.assertTrue("Values of termini_bonds need to be tuples." == str(context.exception))

    def test_update_termini_bonds_dictionary_tuple_elements_no_tuples(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                          "end": (("P", "OP1"), ["P", "O5\'"])}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_termini_bonds_dictionary(termini_bonds=backbone_bonds)

        self.assertTrue("Bond type tuple only contains other tuples." == str(context.exception))

    def test_update_termini_bonds_dictionary_tuple_elements_no_strings(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                          "end": (("P", "OP1"), ("8", 7))}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_termini_bonds_dictionary(termini_bonds=backbone_bonds)

        self.assertTrue("Elements of bond tuples need to be strings." == str(context.exception))

    def test_update_termini_bonds_dictionary_tuple_elements_too_many(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2", "Popeye"), ("P", "O5\'")),
                          "end": (("P", "OP1"), ("P", "O5\'"))}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_termini_bonds_dictionary(termini_bonds=backbone_bonds)

        self.assertTrue("Exactly two strings for bond tuple are allowed." == str(context.exception))

    def test_update_termini_bonds_dictionary_tuple_elements_not_enough(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                          "end": (("P", "OP1"), ("O5\'",))}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_termini_bonds_dictionary(termini_bonds=backbone_bonds)

        self.assertTrue("Exactly two strings for bond tuple are allowed." == str(context.exception))

    def test_update_base_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference = {"A": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                     "B": (("P", "OP1"), ("P", "O5\'"))}

        bio_mol = Biomolecule()
        bio_mol.update_side_chain_bonds_dictionary(new_side_chain=reference)
        result = bio_mol.side_chain_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_update_base_bonds_dictionary_no_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        base_bonds = (("P", "OP1"), ("P", "OP2"), ("P", "O5\'"))

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_side_chain_bonds_dictionary(new_side_chain=base_bonds)

        self.assertTrue("Parameter new_bases need to be a dictionary." == str(context.exception))

    def test_update_base_bonds_dictionary_wrong_key_type(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        base_bonds = {3: (("P", "OP1"), ("P", "OP2"), ("P", "O5\'"))}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_side_chain_bonds_dictionary(new_side_chain=base_bonds)

        self.assertTrue("Keys of new_side_chain need to be strings." == str(context.exception))

    def test_update_base_bonds_dictionary_wrong_value_type(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        base_bonds = {"A": [("P", "OP1"), ("P", "OP2"), ("P", "O5\'")]}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_side_chain_bonds_dictionary(new_side_chain=base_bonds)

        self.assertTrue("Values of new_side_chain need to be tuples." == str(context.exception))

    def test_update_base_bonds_dictionary_tuple_elements_no_tuples(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        base_bonds = {"A": (("P", "OP1"), ["P", "OP2"], ("P", "O5\'")),
                      "B": (("P", "OP1"), ("P", "O5\'"))}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_side_chain_bonds_dictionary(new_side_chain=base_bonds)

        self.assertTrue("Bond type tuple only contains other tuples." == str(context.exception))

    def test_update_base_bonds_dictionary_tuple_elements_no_strings(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        base_bonds = {"A": (("P", "OP1"), ("P", "OP2"), (8, "O5\'")),
                      "B": (("P", "OP1"), ("P", "O5\'"))}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_side_chain_bonds_dictionary(new_side_chain=base_bonds)

        self.assertTrue("Elements of bond tuples need to be strings." == str(context.exception))

    def test_update_base_bonds_dictionary_tuple_elements_too_many(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        base_bonds = {"A": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'")),
                      "B": (("P", "OP1", "Olivia"), ("P", "O5\'"))}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_side_chain_bonds_dictionary(new_side_chain=base_bonds)

        self.assertTrue("Exactly two strings for bond tuple are allowed." == str(context.exception))

    def test_update_base_bonds_dictionary_tuple_elements_not_enough(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        base_bonds = {"A": (("OP1",), ("P", "OP2"), ("P", "O5\'")),
                      "B": (("P", "OP1"), ("P", "O5\'"))}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_side_chain_bonds_dictionary(new_side_chain=base_bonds)

        self.assertTrue("Exactly two strings for bond tuple are allowed." == str(context.exception))

    def test_update_hydrogen_bond_dictionary_donor(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference_atoms = {"adenine": {"H61": 1,
                                       "H62": 1}}

        bio_mol = Biomolecule()
        bio_mol.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=reference_atoms, update_donors=True)
        result = bio_mol.donors_dictionary

        self.assertDictEqual(reference_atoms, result)

    def test_update_hydrogen_bond_dictionary_acceptor(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference_atoms = {"adenine": {"H61": 1,
                                       "H62": 1}}

        bio_mol = Biomolecule()
        bio_mol.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=reference_atoms, update_donors=False)
        result = bio_mol.acceptors_dictionary

        self.assertDictEqual(reference_atoms, result)

    def test_update_hydrogen_bond_dictionary_wrong_input_type_for_update_donors(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        new_atoms = {"adenine": {"H61": 1,
                                 "H62": 1}}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=new_atoms, update_donors=7)

        self.assertTrue("Parameter update_donors need to be a boolean." == str(context.exception))

    def test_update_hydrogen_bond_dictionary_no_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        new_atoms = 7

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=new_atoms)

        self.assertTrue("Parameter hydrogen_bond_atoms need to be a dictionary." == str(context.exception))

    def test_update_hydrogen_bond_dictionary_wrong_key_type(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        new_atoms = {"adenine": {"H61": 1,
                                 "H62": 1},
                     42: {"H61": 1,
                          "H62": 1,
                          "H63": 1}
                     }

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=new_atoms)

        self.assertTrue("Keys of hydrogen_bond_atoms need to be a strings." == str(context.exception))

    def test_update_hydrogen_bond_dictionary_wrong_value_type(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        new_atoms = {"adenine": {"H61": 1,
                                 "H62": 1},
                     "super_adenine": ["potatoes, eggs, milk"]
                     }

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=new_atoms)

        self.assertTrue("Values of hydrogen_bond_atoms need to be a dictionaries." == str(context.exception))

    def test_update_hydrogen_bond_dictionary_atoms_dictionary_no_integers(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        new_atoms = {"adenine": {"H61": 1,
                                 "H62": 1},
                     "super_adenine": {"H61": "1",
                                       "H62": 1}
                     }

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=new_atoms)

        self.assertTrue("Values of the value dictionary need to be integers." == str(context.exception))


class TestNucleicAcid(TestCase):
    def test_update_base_pairs_dictionary_new_pair_type(self):
        from yeti.dictionaries.molecules.biomolecules import NucleicAcid

        new_base_combination = {"a_with_b": (("N3", "N1"), ("O2", "N2"))}
        new_pair_type = {"watson-trick": new_base_combination}

        watson_crick = {"cytosine_guanine": (("N4", "O6"), ("N3", "N1"), ("O2", "N2"))}

        reference = {"watson-crick": watson_crick,
                     "watson-trick": new_base_combination}

        nuc_acid = NucleicAcid()
        nuc_acid.update_base_pairs_dictionary(new_base_pairs=new_pair_type)
        result = nuc_acid.base_pairs_dictionary

        self.assertDictEqual(reference, result)

    def test_update_base_pairs_dictionary_new_watson_crick(self):
        from yeti.dictionaries.molecules.biomolecules import NucleicAcid

        new_base_combination = {"a_with_b": (("N3", "N1"), ("O2", "N2"))}
        new_pair_type = {"watson-crick": new_base_combination}

        watson_crick = {"cytosine_guanine": (("N4", "O6"), ("N3", "N1"), ("O2", "N2")),
                        "a_with_b": (("N3", "N1"), ("O2", "N2"))}

        reference = {"watson-crick": watson_crick}

        nuc_acid = NucleicAcid()
        nuc_acid.update_base_pairs_dictionary(new_base_pairs=new_pair_type)
        result = nuc_acid.base_pairs_dictionary

        self.assertDictEqual(reference, result)

    def test_update_base_pairs_dictionary_no_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import NucleicAcid, BiomoleculesException

        new_pair_type = (("N3", "N1"), ("O2", "N2"))

        nucleic_acid = NucleicAcid()

        with self.assertRaises(BiomoleculesException) as context:
            nucleic_acid.update_base_pairs_dictionary(new_base_pairs=new_pair_type)

        self.assertTrue("Parameter new_base_pairs need to be a dictionary." == str(context.exception))

    def test_update_base_pairs_dictionary_wrong_value_type(self):
        from yeti.dictionaries.molecules.biomolecules import NucleicAcid, BiomoleculesException

        new_pair_type = {7: {"a_with_b": (("N3", "N1"), ("O2", "N2"))}}

        nucleic_acid = NucleicAcid()

        with self.assertRaises(BiomoleculesException) as context:
            nucleic_acid.update_base_pairs_dictionary(new_base_pairs=new_pair_type)

        self.assertTrue("Keys of new_base_pairs need to be a strings." == str(context.exception))

    def test_update_base_pairs_dictionary_base_pair_dict_no_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import NucleicAcid, BiomoleculesException

        new_pair_type = {"watson-crick": (("N3", "N1"), ("O2", "N2"))}

        nucleic_acid = NucleicAcid()

        with self.assertRaises(BiomoleculesException) as context:
            nucleic_acid.update_base_pairs_dictionary(new_base_pairs=new_pair_type)

        self.assertTrue("Value of new_base_pairs need to be a dictionary." == str(context.exception))

    def test_update_base_pairs_dictionary_wrong_base_pair_key_type(self):
        from yeti.dictionaries.molecules.biomolecules import NucleicAcid, BiomoleculesException

        new_pair_type = {"watson-crick": {0: (("N3", "N1"), ("O2", "N2"))}}

        nucleic_acid = NucleicAcid()

        with self.assertRaises(BiomoleculesException) as context:
            nucleic_acid.update_base_pairs_dictionary(new_base_pairs=new_pair_type)

        self.assertTrue("Keys of the base pair dictionary need to be a strings." == str(context.exception))

    def test_update_base_pairs_dictionary_base_pair_value_no_tuple(self):
        from yeti.dictionaries.molecules.biomolecules import NucleicAcid, BiomoleculesException

        new_pair_type = {"watson-crick": {"a_with_b": [("N3", "N1"), ("O2", "N2")]}}

        nucleic_acid = NucleicAcid()

        with self.assertRaises(BiomoleculesException) as context:
            nucleic_acid.update_base_pairs_dictionary(new_base_pairs=new_pair_type)

        self.assertTrue("Value of the base pair dictionary need to be a tuple." == str(context.exception))

    def test_update_base_pairs_dictionary_base_pair_bond_no_tuples(self):
        from yeti.dictionaries.molecules.biomolecules import NucleicAcid, BiomoleculesException

        new_pair_type = {"watson-crick": {"a_with_b": (("N3", "N1"), ["O2", "N2"])}}

        nucleic_acid = NucleicAcid()

        with self.assertRaises(BiomoleculesException) as context:
            nucleic_acid.update_base_pairs_dictionary(new_base_pairs=new_pair_type)

        self.assertTrue("Base pair bonds need to be tuples." == str(context.exception))

    def test_update_base_pairs_dictionary_base_pair_bonds_too_many_atoms(self):
        from yeti.dictionaries.molecules.biomolecules import NucleicAcid, BiomoleculesException

        new_pair_type = {"watson-crick": {"a_with_b": (("N3", "N1", "N42"), ("O2", "N2"))}}

        nucleic_acid = NucleicAcid()

        with self.assertRaises(BiomoleculesException) as context:
            nucleic_acid.update_base_pairs_dictionary(new_base_pairs=new_pair_type)

        self.assertTrue("Base pair bonds contains only two strings." == str(context.exception))

    def test_update_base_pairs_dictionary_base_pair_bonds_not_enough_atoms(self):
        from yeti.dictionaries.molecules.biomolecules import NucleicAcid, BiomoleculesException

        new_pair_type = {"watson-crick": {"a_with_b": (("N3", "N1"), ("O2",))}}

        nucleic_acid = NucleicAcid()

        with self.assertRaises(BiomoleculesException) as context:
            nucleic_acid.update_base_pairs_dictionary(new_base_pairs=new_pair_type)

        self.assertTrue("Base pair bonds contains only two strings." == str(context.exception))

    def test_update_base_pairs_dictionary_base_pair_bonds_atoms_no_strings(self):
        from yeti.dictionaries.molecules.biomolecules import NucleicAcid, BiomoleculesException

        new_pair_type = {"watson-crick": {"a_with_b": (("N3", 4), ("O2", "N2"))}}

        nucleic_acid = NucleicAcid()

        with self.assertRaises(BiomoleculesException) as context:
            nucleic_acid.update_base_pairs_dictionary(new_base_pairs=new_pair_type)

        self.assertTrue("Base pair bonds contains only two strings." == str(context.exception))


class TestRNA(TestCase):

    def test_abbreviations_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        reference = {"adenine": "A",
                     "cytosine": "C",
                     "guanine": "G",
                     "uracil": "U"}

        rna = RNA()
        result = rna.abbreviation_dictionary

        self.assertDictEqual(reference, result)

    def test_dihedral_angles_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        reference = {"alpha": (("O3\'", "P", "O5\'", "C5\'"), (-1, 0, 0, 0), r"$\alpha$"),
                     "beta": (("P", "O5\'", "C5\'", "C4\'"), (0, 0, 0, 0), r"$\beta$"),
                     "gamma": (("O5\'", "C5\'", "C4\'", "C3\'"), (0, 0, 0, 0), r"$\gamma$"),
                     "delta": (("C5\'", "C4\'", "C3\'", "O3\'"), (0, 0, 0, 0), r"$\delta$"),
                     "epsilon": (("C4\'", "C3\'", "O3\'", "P"), (0, 0, 0, 1), r"$\epsilon$"),
                     "zeta": (("C3\'", "O3\'", "P", "O5\'"), (0, 0, 1, 1), r"$\zeta$"),
                     "tau0": (("C4\'", "O4\'", "C1\'", "C2\'"), (0, 0, 0, 0), r"$\tau_0$"),
                     "tau1": (("O4\'", "C1\'", "C2\'", "C3\'"), (0, 0, 0, 0), r"$\tau_1$"),
                     "tau2": (("C1\'", "C2\'", "C3\'", "C4\'"), (0, 0, 0, 0), r"$\tau_2$"),
                     "tau3": (("C2\'", "C3\'", "C4\'", "O4\'"), (0, 0, 0, 0), r"$\tau_3$"),
                     "tau4": (("C3\'", "C4\'", "O4\'", "C1\'"), (0, 0, 0, 0), r"$\tau_4$"),
                     "chi_py": (("O4\'", "C1\'", "N1", "C2"), (0, 0, 0, 0), r"$\chi_pyrimidine$"),
                     "chi_pu": (("O4\'", "C1\'", "N9", "C4"), (0, 0, 0, 0), r"$\chi_purine$")
                     }

        rna = RNA()
        result = rna.dihedral_angles_dictionary

        self.assertDictEqual(reference, result)

    def test_distances_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        reference = {"P_P": (("P", "P"), "P to P"),
                     }

        rna = RNA()
        result = rna.distances_dictionary

        self.assertDictEqual(reference, result)

    def test_bonds_between_residues(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        reference = ("O3\'", "P")

        rna = RNA()
        result = rna.bonds_between_residues

        self.assertTupleEqual(reference, result)

    def test_backbone_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        reference = {"residual": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'"), ("O5\'", "C5\'"),
                                  ("C5\'", "H5\'1"), ("C5\'", "H5\'2"), ("C5\'", "C4\'"), ("C4\'", "H4\'"),
                                  ("C4\'", "O4\'"), ("C4\'", "C3\'"), ("O4\'", "C1\'"), ("C1\'", "H1\'"),
                                  ("C1\'", "C2\'"), ("C3\'", "H3\'"), ("C3\'", "C2\'"), ("C3\'", "O3\'"),
                                  ("C2\'", "H2\'1"), ("C2\'", "O2\'"), ("O2\'", "HO\'2"), ("O5\'", "H5T"),
                                  ("O3'", "HO3\'"))}

        rna = RNA()
        result = rna.backbone_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_termini_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        reference = {"p_capping": (("O5\'", "C5\'"), ("C5\'", "H5\'1"), ("C5\'", "H5\'2"), ("C5\'", "C4\'"),
                                   ("C4\'", "H4\'"), ("C4\'", "O4\'"), ("C4\'", "C3\'"), ("O4\'", "C1\'"),
                                   ("C1\'", "H1\'"), ("C1\'", "C2\'"), ("C3\'", "H3\'"), ("C3\'", "C2\'"),
                                   ("C3\'", "O3\'"), ("C2\'", "H2\'1"), ("C2\'", "O2\'"), ("O2\'", "HO\'2"),
                                   ("O5\'", "H5T"), ("O3'", "HO3\'"))}

        rna = RNA()
        result = rna.termini_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_base_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        reference = {"adenine": (("C1\'", "N9"), ("N9", "C8"), ("N9", "C4"), ("C8", "H8"), ("C8", "N7"),
                                 ("N7", "C5"), ("C5", "C6"), ("C5", "C4"), ("C6", "N6"), ("C6", "N1"),
                                 ("N6", "H61"), ("N6", "H62"), ("N1", "C2"), ("C2", "H2"), ("C2", "N3"),
                                 ("N3", "C4")),
                     "cytosine": (("C1\'", "N1"), ("N1", "C6"), ("N1", "C2"), ("C6", "H6"),
                                  ("C6", "C5"), ("C5", "H5"), ("C5", "C4"), ("C4", "N4"), ("C4", "N3"),
                                  ("N4", "H41"), ("N4", "H42"), ("N3", "C2"), ("C2", "O2")),
                     "guanine": (("C1\'", "N9"), ("N9", "C8"), ("N9", "C4"), ("C8", "H8"), ("C8", "N7"),
                                 ("N7", "C5"), ("C5", "C6"), ("C5", "C4"), ("C6", "O6"), ("C6", "N1"),
                                 ("N1", "H1"), ("N1", "C2"), ("C2", "N2"), ("C2", "N3"), ("N2", "H21"),
                                 ("N2", "H22"), ("N3", "C4")),
                     "uracil": (("C1\'", "N1"), ("N1", "C6"), ("N1", "C2"), ("C6", "H6"), ("C6", "C5"), ("C5", "H5"),
                                ("C5", "C4"), ("C4", "O4"), ("C4", "N3"), ("N3", "H3"), ("N3", "C2"), ("C2", "O2"))
                     }

        rna = RNA()
        result = rna.side_chain_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_donors_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        reference = {"adenine": {"H61": 1,
                                 "H62": 1
                                 },
                     "cytosine": {"H41": 1,
                                  "H42": 1
                                  },
                     "guanine": {"H1": 1,
                                 "H21": 1,
                                 "H22": 1
                                 },
                     "uracil": {"H3": 1
                                },
                     "backbone": {"HO\'2": 1}
                     }

        rna = RNA()
        result = rna.donors_dictionary

        self.assertDictEqual(reference, result)

    def test_acceptors_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        reference = {"adenine": {"N1": 1,
                                 "N3": 1,
                                 "N6": 1,
                                 "N7": 1
                                 },
                     "cytosine": {"O2": 2,
                                  "N3": 1,
                                  "N4": 1
                                  },
                     "guanine": {"O6": 2,
                                 "N3": 1,
                                 "N7": 1
                                 },
                     "uracil": {"O2": 2,
                                "O4": 2
                                },
                     "backbone": {"OP1": 2,
                                  "OP2": 2,
                                  "O2\'": 2,
                                  "O3\'": 2,
                                  "O4\'": 2,
                                  "O5\'": 2}
                     }

        rna = RNA()
        result = rna.acceptors_dictionary

        self.assertDictEqual(reference, result)

    def test_base_pairs_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        watson_crick_reference = {"adenine_uracil": (("N6", "O4"), ("N1", "N3")),
                                  "cytosine_guanine": (("N4", "O6"), ("N3", "N1"), ("O2", "N2"))}

        reference = {"watson-crick": watson_crick_reference}

        rna = RNA()
        result = rna.base_pairs_dictionary

        self.assertDictEqual(reference, result)


class TestDNA(TestCase):

    def test_abbreviations_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        reference = {"adenine": "A",
                     "cytosine": "C",
                     "guanine": "G",
                     "thymine": "T"}

        dna = DNA()
        result = dna.abbreviation_dictionary

        self.assertDictEqual(reference, result)

    def test_dihedral_angles_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        reference = {"alpha": (("O3\'", "P", "O5\'", "C5\'"), (-1, 0, 0, 0), r"$\alpha$"),
                     "beta": (("P", "O5\'", "C5\'", "C4\'"), (0, 0, 0, 0), r"$\beta$"),
                     "gamma": (("O5\'", "C5\'", "C4\'", "C3\'"), (0, 0, 0, 0), r"$\gamma$"),
                     "delta": (("C5\'", "C4\'", "C3\'", "O3\'"), (0, 0, 0, 0), r"$\delta$"),
                     "epsilon": (("C4\'", "C3\'", "O3\'", "P"), (0, 0, 0, 1), r"$\epsilon$"),
                     "zeta": (("C3\'", "O3\'", "P", "O5\'"), (0, 0, 1, 1), r"$\zeta$"),
                     "tau0": (("C4\'", "O4\'", "C1\'", "C2\'"), (0, 0, 0, 0), r"$\tau_0$"),
                     "tau1": (("O4\'", "C1\'", "C2\'", "C3\'"), (0, 0, 0, 0), r"$\tau_1$"),
                     "tau2": (("C1\'", "C2\'", "C3\'", "C4\'"), (0, 0, 0, 0), r"$\tau_2$"),
                     "tau3": (("C2\'", "C3\'", "C4\'", "O4\'"), (0, 0, 0, 0), r"$\tau_3$"),
                     "tau4": (("C3\'", "C4\'", "O4\'", "C1\'"), (0, 0, 0, 0), r"$\tau_4$"),
                     "chi_py": (("O4\'", "C1\'", "N1", "C2"), (0, 0, 0, 0), r"$\chi_pyrimidine$"),
                     "chi_pu": (("O4\'", "C1\'", "N9", "C4"), (0, 0, 0, 0), r"$\chi_purine$")
                     }

        dna = DNA()
        result = dna.dihedral_angles_dictionary

        self.assertDictEqual(reference, result)

    def test_distances_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        reference = {"P_P": (("P", "P"), "P to P")}

        dna = DNA()
        result = dna.distances_dictionary

        self.assertDictEqual(reference, result)

    def test_bonds_between_residues(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        reference = ("O3\'", "P")

        dna = DNA()
        result = dna.bonds_between_residues

        self.assertTupleEqual(reference, result)

    def test_backbone_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        reference = {"residual": (("P", "OP1"), ("P", "OP2"), ("P", "O5\'"), ("O5\'", "C5\'"), ("C5\'", "H5\'1"),
                                  ("C5\'", "H5\'2"), ("C5\'", "C4\'"), ("C4\'", "H4\'"), ("C4\'", "O4\'"),
                                  ("C4\'", "C3\'"), ("O4\'", "C1\'"), ("C1\'", "H1\'"), ("C1\'", "C2\'"),
                                  ("C3\'", "H3\'"), ("C3\'", "C2\'"), ("C3\'", "O3\'"), ("C2\'", "H2\'1"),
                                  ("C2\'", "H2\'2"))}

        dna = DNA()
        result = dna.backbone_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_termini_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        reference = {}

        dna = DNA()
        result = dna.termini_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_base_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        reference = {"adenine": (("C1\'", "N9"), ("N9", "C8"), ("N9", "C4"), ("C8", "H8"), ("C8", "N7"), ("N7", "C5"),
                                 ("C5", "C6"), ("C5", "C4"), ("C6", "N6"), ("C6", "N1"), ("N6", "H61"), ("N6", "H62"),
                                 ("N1", "C2"), ("C2", "H2"), ("C2", "N3"), ("N3", "C4")),
                     "cytosine": (("C1\'", "N1"), ("N1", "C6"), ("N1", "C2"), ("C6", "H6"), ("C6", "C5"), ("C5", "H5"),
                                  ("C5", "C4"), ("C4", "N4"), ("C4", "N3"), ("N4", "H41"), ("N4", "H42"), ("N3", "C2"),
                                  ("C2", "O2")),
                     "guanine": (("C1\'", "N9"), ("N9", "C8"), ("N9", "C4"), ("C8", "H8"), ("C8", "N7"), ("N7", "C5"),
                                 ("C5", "C6"), ("C5", "C4"), ("C6", "O6"), ("C6", "N1"), ("N1", "H1"), ("N1", "C2"),
                                 ("C2", "N2"), ("C2", "N3"), ("N2", "H21"), ("N2", "H22"), ("N3", "C4")),
                     "thymine": (("C1\'", "N1"), ("N1", "C6"), ("N1", "C2"), ("C6", "H6"), ("C6", "C5"), ("C5", "C5M"),
                                 ("C5M", "H51"), ("C5M", "H52"), ("C5M", "H53"), ("C5", "C4"), ("C4", "O4"),
                                 ("C4", "N3"), ("N3", "H3"), ("N3", "C2"), ("C2", "O2"))
                     }

        dna = DNA()
        result = dna.side_chain_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_donors_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        reference = {"adenine": {"H61": 1,
                                 "H62": 1
                                 },
                     "cytosine": {"H41": 1,
                                  "H42": 1
                                  },
                     "guanine": {"H1": 1,
                                 "H21": 1,
                                 "H22": 1
                                 },
                     "thymine": {"H3": 1
                                 },
                     "backbone": {}
                     }

        dna = DNA()
        result = dna.donors_dictionary

        self.assertDictEqual(reference, result)

    def test_acceptors_dictionary(self):
        # Note: Order of elements is important
        from yeti.dictionaries.molecules.biomolecules import DNA

        reference = {"adenine": {"N1": 1,
                                 "N3": 1,
                                 "N6": 1,
                                 "N7": 1
                                 },
                     "cytosine": {"O2": 2,
                                  "N3": 1,
                                  "N4": 1
                                  },
                     "guanine": {"O6": 2,
                                 "N3": 1,
                                 "N7": 1
                                 },
                     "thymine": {"O2": 2,
                                 "O4": 2
                                 },
                     "backbone": {"OP1": 2,
                                  "OP2": 2,
                                  "O3\'": 2,
                                  "O4\'": 2,
                                  "O5\'": 2}}

        dna = DNA()
        result = dna.acceptors_dictionary

        self.assertDictEqual(reference, result)

    def test_base_pairs_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        watson_crick_reference = {"adenine_thymine": (("N6", "O4"), ("N1", "N3")),
                                  "cytosine_guanine": (("N4", "O6"), ("N3", "N1"), ("O2", "N2"))}

        reference = {"watson-crick": watson_crick_reference}

        dna = DNA()
        result = dna.base_pairs_dictionary

        self.assertDictEqual(reference, result)


class TestProtein(TestCase):
    def test_abbreviations_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Protein

        reference = {"glycine": "Gly",
                     "alanine": "Ala",
                     "valine": "Val",
                     "leucine": "Leu",
                     "isoleucine": "Ile",
                     "serine": "Ser",
                     "threonine": "Thr",
                     "proline": "Pro",
                     "aspartic acid": "Asp",
                     "glutamic acid": "Glu",
                     "asparagine": "Asn",
                     "glutamine": "Gln",
                     "methionine": "Met",
                     "cysteine": "Cys",
                     "lysine": "Lys",
                     "arginine": "Arg",
                     "histidine": "His",
                     "phenylalanine": "Phe",
                     "tyrosine": "Tyr",
                     "tryptophan": "Trp",
                     "acetyl group": "Ace",
                     "methylamine": "Nme"}

        protein = Protein()
        result = protein.abbreviation_dictionary

        self.assertDictEqual(reference, result)

    def test_dihedral_angles_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Protein

        reference = {"psi": (("N", "CA", "C", "N"), (0, 0, 0, 1), r'$\psi$'),
                     "phi": (("C", "N", "CA", "C"), (-1, 0, 0, 0), r'$\phi$'),
                     "omega": (("CA", "C", "N", "CA"), (0, 0, 1, 1), r'$\omega$'),
                     "chi1": (("N", "CA", "CB", "CG"), (0, 0, 0, 0), r'$\chi_1$'),
                     "chi2": (("CA", "CB", "CG", "CD"), (0, 0, 0, 0), r'$\chi_2$'),
                     "chi3": (("CB", "CG", "CD", "CE"), (0, 0, 0, 0), r'$\chi_3$'),
                     "chi4": (("CG", "CD", "CE", "NZ"), (0, 0, 0, 0), r'$\chi_4$')}

        protein = Protein()
        result = protein.dihedral_angles_dictionary

        self.assertDictEqual(reference, result)

    def test_distances_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Protein

        reference = {"CA_CA": (("CA", "CA"), r'$C_{\alpha}$ to $C_{\alpha}$')}

        protein = Protein()
        result = protein.distances_dictionary

        self.assertDictEqual(reference, result)

    def test_bonds_between_residues(self):
        from yeti.dictionaries.molecules.biomolecules import Protein

        reference = ("C", "N")

        protein = Protein()
        result = protein.bonds_between_residues

        self.assertTupleEqual(reference, result)

    def test_backbone_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Protein

        reference = {"residual": (("N", "H"), ("N", "CA"), ("CA", "HA1"), ("CA", "C"), ("C", "O"))}

        protein = Protein()
        result = protein.backbone_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_termini_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Protein

        reference = {"acetyl group": (("C", "O"), ("C", "CH3"), ("CH3", "HH31"), ("CH3", "HH32"), ("CH3", "HH33")),
                     "methylamine": (("N", "H"), ("N", "CH3"), ("CH3", "HH31"), ("CH3", "HH32"), ("CH3", "HH33"))}

        protein = Protein()
        result = protein.termini_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_amino_acid_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Protein

        reference = {"glycine": (("CA", "HA2"),),
                     "alanine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "HB3")),
                     "valine": (("CA", "CB"), ("CB", "HB1"), ("CB", "CG1"), ("CG1", "HG11"), ("CG1", "HG12"),
                                ("CG1", "HG13"), ("CB", "CG2"), ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "HG23")),
                     "leucine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"), ("CG", "CD1"),
                                 ("CD1", "HD11"), ("CD1", "HD12"), ("CD1", "HD13"), ("CG", "CD2"), ("CD2", "HD21"),
                                 ("CD2", "HD22"), ("CD2", "HD23")),
                     "isoleucine": (("CA", "CB"), ("CB", "HB1"), ("CB", "CG1"), ("CG1", "HG11"), ("CG1", "HG12"),
                                    ("CG1", "HG13"), ("CB", "CG2"), ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "CD"),
                                    ("CD", "HD1"), ("CD", "HD2"), ("CD", "HD3")),
                     "serine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "OG"), ("OG", "HG1")),
                     "threonine": (("CA", "CB"), ("CB", "HB1"), ("CB", "OG1"), ("OG1", "HG11"), ("CB", "CG2"),
                                   ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "HG23")),
                     "proline": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"), ("CG", "HG2"),
                                 ("CG", "CD"), ("CD", "HD1"), ("CD", "HD2"), ("CD", "N")),
                     "aspartic acid": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "OD1"),
                                       ("CG", "OD2")),
                     "glutamic acid": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"),
                                       ("CG", "HG2"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")),
                     "asparagine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "OD1"),
                                    ("CG", "ND2"), ("ND2", "HD21"), ("ND2", "HD22")),
                     "glutamine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"),
                                   ("CG", "HG2"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2"), ("NE2", "HE21"),
                                   ("NE2", "HE22")),
                     "methionine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"),
                                    ("CG", "HG2"), ("CG", "SD"), ("SD", "CE"), ("CE", "HE1"), ("CE", "HE2"),
                                    ("CE", "HE3")),
                     "cysteine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "SG"), ("SG", "HG")),
                     "lysine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"), ("CG", "HG2"),
                                ("CG", "CD"), ("CD", "HD1"), ("CD", "HD2"), ("CD", "CE"), ("CE", "HE1"), ("CE", "HE2"),
                                ("CE", "NZ"), ("NZ", "HZ1"), ("NZ", "HZ2"), ("NZ", "HZ3")),
                     "arginine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"),
                                  ("CG", "HG2"), ("CG", "CD"), ("CD", "HD1"), ("CD", "HD2"), ("CD", "NE"),
                                  ("NE", "HE1"), ("NE", "CZ"), ("CZ", "NH1"), ("NH1", "HH11"), ("NH1", "HH12"),
                                  ("CZ", "NH2"), ("NH2", "HH21"), ("NH2", "HH22")),
                     "histidine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "CD1"),
                                   ("CD1", "HD11"), ("CG", "ND2"), ("ND2", "HD21"), ("CD1", "NE1"), ("NE1", "HE11"),
                                   ("ND2", "CE2"), ("CE2", "HE21"), ("NE1", "CE2")),
                     "phenylalanine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "CD1"),
                                       ("CD1", "HD11"), ("CG", "CD2"), ("CD2", "HD21"), ("CD1", "CE1"), ("CE1", "HE11"),
                                       ("CD2", "CE2"), ("CE2", "HE21"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "HZ1")),
                     "tyrosine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "CD1"),
                                  ("CD1", "HD11"), ("CG", "CD2"), ("CD2", "HD21"), ("CD1", "CE1"), ("CE1", "HE11"),
                                  ("CD2", "CE2"), ("CE2", "HE21"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH"),
                                  ("OH", "HH1")),
                     "tryptophan": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "CD1"),
                                    ("CG", "CD2"), ("CD2", "HD21"), ("CD2", "NE2"), ("NE2", "HE21"), ("NE2", "CZ2"),
                                    ("CZ2", "CH2"), ("CH2", "HH21"), ("CH2", "CTH2"), ("CTH2", "HTH21"), ("CD1", "CZ2"),
                                    ("CD1", "CE3"), ("CE3", "HE31"), ("CE3", "CZ3"), ("CZ3", "HZ31"), ("CZ3", "CTH2"))}

        protein = Protein()
        result = protein.side_chain_bonds_dictionary

        for amino_acid in reference.keys():
            print(amino_acid)
            tmp_reference = reference[amino_acid]
            tmp_result = result[amino_acid]

            self.assertTupleEqual(tmp_reference, tmp_result)

        self.assertDictEqual(reference, result)

    def test_acceptors_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Protein

        # TODO: Fill dictionary
        reference = {}

        protein = Protein()
        result = protein.acceptors_dictionary

        self.assertDictEqual(reference, result)

    def test_donors_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Protein

        # TODO: Fill dictionary
        reference = {}

        protein = Protein()
        result = protein.donors_dictionary

        self.assertDictEqual(reference, result)
