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

    def test_update_derivations_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference = {"argenine": "ARG",
                     "lysine": "LYS"}

        bio_mol = Biomolecule()
        bio_mol.update_derivations_dictionary(derivations=reference)
        result = bio_mol.derivations_dictionary

        self.assertDictEqual(reference, result)

    def test_update_derivations_dictionary_false_parameter_type(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        derivations = ["eggs", "butter"]

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_derivations_dictionary(derivations=derivations)

        self.assertTrue("Parameter derivations need to be a dictionary." == str(context.exception))

    def test_update_derivations_dictionary_false_parameter_type(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        derivations = {"ADE": "A",
                       "THY": "T",
                       "GUA": 7}

        bio_mol = Biomolecule()

        with self.assertRaises(BiomoleculesException) as context:
            bio_mol.update_derivations_dictionary(derivations=derivations)

        self.assertTrue("Values of derivations need to be strings." == str(context.exception))

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

    def test_update_dihedral_angle_dictionary_value_no_tuple(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule, BiomoleculesException

        dih_angles = {"alpha": [["O3\'", "P", "O5\'", "C5\'"], (-1, 0, 0, 0), r"$\alpha$"]}

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

        print(str(context.exception))

        self.assertTrue("Third element need to be a string." == str(context.exception))

    def test_update_distances_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference = {"PToP": (["P", "P"], "P to P"),
                     "PToMG": (["P", "MG"], "P to MG")}

        bio_mol = Biomolecule()
        bio_mol.update_distances_dictionary(distances=reference)
        result = bio_mol.distances_dictionary

        self.assertDictEqual(reference, result)

    def test_update_backbone_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference = {"start": [["P", "OP1"], ["P", "OP2"], ["P", "O5\'"]],
                     "end": [["P", "OP1"], ["P", "O5\'"]]}

        bio_mol = Biomolecule()
        bio_mol.update_backbone_bonds_dictionary(backbone_bonds=reference)
        result = bio_mol.backbone_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_update_base_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference = {"A": [["P", "OP1"], ["P", "OP2"], ["P", "O5\'"]],
                     "B": [["P", "OP1"], ["P", "O5\'"]]}

        bio_mol = Biomolecule()
        bio_mol.update_base_bonds_dictionary(new_bases=reference)
        result = bio_mol.base_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_update_hydrogen_bond_dictionary_donor(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference_atoms = {"adenine": ("H61", "H62")}

        reference_slots = {"adenine": {"H61": 1,
                                       "H62": 1}}

        reference = {"atoms": reference_atoms,
                     "slots": reference_slots}

        bio_mol = Biomolecule()
        bio_mol.update_hydrogen_bond_dictionary(atoms=reference_atoms, slots=reference_slots, update_donors=True)
        result = bio_mol.donors_dictionary

        self.assertDictEqual(reference, result)

    def test_update_hydrogen_bond_dictionary_aceptor(self):
        from yeti.dictionaries.molecules.biomolecules import Biomolecule

        reference_atoms = {"adenine": ("H61", "H62")}

        reference_slots = {"adenine": {"H61": 1,
                                       "H62": 1}}

        reference = {"atoms": reference_atoms,
                     "slots": reference_slots}

        bio_mol = Biomolecule()
        bio_mol.update_hydrogen_bond_dictionary(atoms=reference_atoms, slots=reference_slots, update_donors=False)
        result = bio_mol.acceptors_dictionary

        self.assertDictEqual(reference, result)


class TestNucleicAcid(TestCase):
    def test_update_base_pairs_dictionary_new_pair_type(self):
        from yeti.dictionaries.molecules.biomolecules import NucleicAcid

        new_base_combination = {"a_with_b": (["N3", "N1"], ["O2", "N2"])}
        new_pair_type = {"watson-trick": new_base_combination}

        watson_crick = {"cytosine_guanine": (["N4", "O6"], ["N3", "N1"], ["O2", "N2"])}

        reference = {"watson-crick": watson_crick,
                     "watson-trick": new_base_combination}

        nuc_acid = NucleicAcid()
        nuc_acid.update_base_pairs_dictionary(new_base_pairs=new_pair_type)
        result = nuc_acid.base_pairs_dictionary

        self.assertDictEqual(reference, result)

    def test_update_base_pairs_dictionary_new_watson_crick(self):
        from yeti.dictionaries.molecules.biomolecules import NucleicAcid

        new_base_combination = {"a_with_b": (["N3", "N1"], ["O2", "N2"])}
        new_pair_type = {"watson-crick": new_base_combination}

        watson_crick = {"cytosine_guanine": (["N4", "O6"], ["N3", "N1"], ["O2", "N2"]),
                        "a_with_b": (["N3", "N1"], ["O2", "N2"])}

        reference = {"watson-crick": watson_crick}

        nuc_acid = NucleicAcid()
        nuc_acid.update_base_pairs_dictionary(new_base_pairs=new_pair_type)
        result = nuc_acid.base_pairs_dictionary

        self.assertDictEqual(reference, result)


class TestRNA(TestCase):

    def test_derivations_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        reference = {"adenine": "A",
                     "cytosine": "C",
                     "guanine": "G",
                     "uracil": "U"}

        rna = RNA()
        result = rna.derivations_dictionary

        self.assertDictEqual(reference, result)

    def test_dihedral_angles_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        reference = {"alpha": (["O3\'", "P", "O5\'", "C5\'"], [-1, 0, 0, 0], r"$\alpha$"),
                     "beta": (["P", "O5\'", "C5\'", "C4\'"], [0, 0, 0, 0], r"$\beta$"),
                     "gamma": (["O5\'", "C5\'", "C4\'", "C3\'"], [0, 0, 0, 0], r"$\gamma$"),
                     "delta": (["C5\'", "C4\'", "C3\'", "O3\'"], [0, 0, 0, 0], r"$\delta$"),
                     "epsilon": (["C4\'", "C3\'", "O3\'", "P"], [0, 0, 0, 1], r"$\epsilon$"),
                     "zeta": (["C3\'", "O3\'", "P", "O5\'"], [0, 0, 1, 1], r"$\zeta$"),
                     "tau0": (["C4\'", "O4\'", "C1\'", "C2\'"], [0, 0, 0, 0], r"$\tau_0$"),
                     "tau1": (["O4\'", "C1\'", "C2\'", "C3\'"], [0, 0, 0, 0], r"$\tau_1$"),
                     "tau2": (["C1\'", "C2\'", "C3\'", "C4\'"], [0, 0, 0, 0], r"$\tau_2$"),
                     "tau3": (["C2\'", "C3\'", "C4\'", "O4\'"], [0, 0, 0, 0], r"$\tau_3$"),
                     "tau4": (["C3\'", "C4\'", "O4\'", "C1\'"], [0, 0, 0, 0], r"$\tau_4$"),
                     "chi_py": (["O4\'", "C1\'", "N1", "C2"], [0, 0, 0, 0], r"$\chi_pyrimidine$"),
                     "chi_pu": (["O4\'", "C1\'", "N9", "C4"], [0, 0, 0, 0], r"$\chi_purine$")
                     }

        rna = RNA()
        result = rna.dihedral_angles_dictionary

        self.assertDictEqual(reference, result)

    def test_distances_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        reference = {"PToP": (["P", "P"], "P to P"),
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

        reference = {"start": (["O5\'", "C5\'"], ["C5\'", "H5\'1"], ["C5\'", "H5\'2"], ["C5\'", "C4\'"],
                               ["C4\'", "H4\'"], ["C4\'", "O4\'"], ["C4\'", "C3\'"], ["O4\'", "C1\'"], ["C1\'", "H1\'"],
                               ["C1\'", "C2\'"], ["C3\'", "H3\'"], ["C3\'", "C2\'"], ["C3\'", "O3\'"],
                               ["C2\'", "H2\'1"], ["C2\'", "O2\'"], ["O2\'", "HO\'2"], ["O5\'", "H5T"],
                               ["O3'", "HO3\'"]),
                     "residual": (["P", "OP1"], ["P", "OP2"], ["P", "O5\'"], ["O5\'", "C5\'"],
                                  ["C5\'", "H5\'1"], ["C5\'", "H5\'2"], ["C5\'", "C4\'"], ["C4\'", "H4\'"],
                                  ["C4\'", "O4\'"], ["C4\'", "C3\'"], ["O4\'", "C1\'"], ["C1\'", "H1\'"],
                                  ["C1\'", "C2\'"], ["C3\'", "H3\'"], ["C3\'", "C2\'"], ["C3\'", "O3\'"],
                                  ["C2\'", "H2\'1"], ["C2\'", "O2\'"], ["O2\'", "HO\'2"], ["O5\'", "H5T"],
                                  ["O3'", "HO3\'"])}

        rna = RNA()
        result = rna.backbone_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_base_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        reference = {"adenine": (["C1\'", "N9"], ["N9", "C8"], ["N9", "C4"], ["C8", "H8"], ["C8", "N7"],
                                 ["N7", "C5"], ["C5", "C6"], ["C5", "C4"], ["C6", "N6"], ["C6", "N1"],
                                 ["N6", "H61"], ["N6", "H62"], ["N1", "C2"], ["C2", "H2"], ["C2", "N3"],
                                 ["N3", "C4"]),
                     "cytosine": (["C1\'", "N1"], ["N1", "C6"], ["N1", "C2"], ["C6", "H6"],
                                  ["C6", "C5"], ["C5", "H5"], ["C5", "C4"], ["C4", "N4"], ["C4", "N3"],
                                  ["N4", "H41"], ["N4", "H42"], ["N3", "C2"], ["C2", "O2"]),
                     "guanine": (["C1\'", "N9"], ["N9", "C8"], ["N9", "C4"], ["C8", "H8"], ["C8", "N7"],
                                 ["N7", "C5"], ["C5", "C6"], ["C5", "C4"], ["C6", "O6"], ["C6", "N1"],
                                 ["N1", "H1"], ["N1", "C2"], ["C2", "N2"], ["C2", "N3"], ["N2", "H21"],
                                 ["N2", "H22"], ["N3", "C4"]),
                     "uracil": (["C1\'", "N1"], ["N1", "C6"], ["N1", "C2"], ["C6", "H6"], ["C6", "C5"], ["C5", "H5"],
                                ["C5", "C4"], ["C4", "O4"], ["C4", "N3"], ["N3", "H3"], ["N3", "C2"], ["C2", "O2"])
                     }

        rna = RNA()
        result = rna.base_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_donors_dictionary(self):
        # Note: Order of elements is important
        from yeti.dictionaries.molecules.biomolecules import RNA

        ref_atoms_dict = {"adenine": ("H61", "H62"),
                          "cytosine": ("H41", "H42"),
                          "guanine": ("H1", "H21", "H22"),
                          "uracil": ("H3",),
                          "backbone": ("HO\'2",)
                          }

        ref_adenine_slots = {"H61": 1,
                             "H62": 1
                             }

        ref_cytosine_slots = {"H41": 1,
                              "H42": 1
                              }

        ref_guanine_slots = {"H1": 1,
                             "H21": 1,
                             "H22": 1
                             }

        ref_uracil_slots = {"H3": 1
                            }

        ref_backbone_slots = {"HO\'2": 1}

        ref_slots_dict = {"adenine": ref_adenine_slots,
                          "cytosine": ref_cytosine_slots,
                          "guanine": ref_guanine_slots,
                          "uracil": ref_uracil_slots,
                          "backbone": ref_backbone_slots
                          }

        reference = {"atoms": ref_atoms_dict,
                     "slots": ref_slots_dict}

        rna = RNA()
        result = rna.donors_dictionary

        self.assertDictEqual(reference, result)

    def test_acceptors_dictionary(self):
        # Note: Order of elements is important
        from yeti.dictionaries.molecules.biomolecules import RNA

        ref_atoms_dict = {"adenine": ("N1", "N3", "N6", "N7"),
                          "cytosine": ("N3", "N4", "O2"),
                          "guanine": ("N3", "N7", "O6"),
                          "uracil": ("O2", "O4"),
                          "backbone": ("O2\'", "O3\'", "O4\'", "O5\'", "OP1", "OP2")
                          }

        ref_adenine_slots = {"N1": 1,
                             "N3": 1,
                             "N6": 1,
                             "N7": 1
                             }

        ref_cytosine_slots = {"O2": 2,
                              "N3": 1,
                              "N4": 1
                              }

        ref_guanine_slots = {"O6": 2,
                             "N3": 1,
                             "N7": 1
                             }

        ref_uracil_slots = {"O2": 2,
                            "O4": 2
                            }

        ref_backbone_slots = {"OP1": 2,
                              "OP2": 2,
                              "O2\'": 2,
                              "O3\'": 2,
                              "O4\'": 2,
                              "O5\'": 2}

        ref_slots_dict = {"adenine": ref_adenine_slots,
                          "cytosine": ref_cytosine_slots,
                          "guanine": ref_guanine_slots,
                          "uracil": ref_uracil_slots,
                          "backbone": ref_backbone_slots
                          }

        reference = {"atoms": ref_atoms_dict,
                     "slots": ref_slots_dict}

        rna = RNA()
        result = rna.acceptors_dictionary

        self.assertDictEqual(reference, result)

    def test_base_pairs_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import RNA

        watson_crick_reference = {"adenine_uracil": (["N6", "O4"], ["N1", "N3"]),
                                  "cytosine_guanine": (["N4", "O6"], ["N3", "N1"], ["O2", "N2"])}

        reference = {"watson-crick": watson_crick_reference}

        rna = RNA()
        result = rna.base_pairs_dictionary

        self.assertDictEqual(reference, result)


class TestDNA(TestCase):

    def test_derivations_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        reference = {"adenine": "A",
                     "cytosine": "C",
                     "guanine": "G",
                     "thymine": "T"}

        dna = DNA()
        result = dna.derivations_dictionary

        self.assertDictEqual(reference, result)

    def test_dihedral_angles_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        reference = {"alpha": (["O3\'", "P", "O5\'", "C5\'"], [-1, 0, 0, 0], r"$\alpha$"),
                     "beta": (["P", "O5\'", "C5\'", "C4\'"], [0, 0, 0, 0], r"$\beta$"),
                     "gamma": (["O5\'", "C5\'", "C4\'", "C3\'"], [0, 0, 0, 0], r"$\gamma$"),
                     "delta": (["C5\'", "C4\'", "C3\'", "O3\'"], [0, 0, 0, 0], r"$\delta$"),
                     "epsilon": (["C4\'", "C3\'", "O3\'", "P"], [0, 0, 0, 1], r"$\epsilon$"),
                     "zeta": (["C3\'", "O3\'", "P", "O5\'"], [0, 0, 1, 1], r"$\zeta$"),
                     "tau0": (["C4\'", "O4\'", "C1\'", "C2\'"], [0, 0, 0, 0], r"$\tau_0$"),
                     "tau1": (["O4\'", "C1\'", "C2\'", "C3\'"], [0, 0, 0, 0], r"$\tau_1$"),
                     "tau2": (["C1\'", "C2\'", "C3\'", "C4\'"], [0, 0, 0, 0], r"$\tau_2$"),
                     "tau3": (["C2\'", "C3\'", "C4\'", "O4\'"], [0, 0, 0, 0], r"$\tau_3$"),
                     "tau4": (["C3\'", "C4\'", "O4\'", "C1\'"], [0, 0, 0, 0], r"$\tau_4$"),
                     "chi_py": (["O4\'", "C1\'", "N1", "C2"], [0, 0, 0, 0], r"$\chi_pyrimidine$"),
                     "chi_pu": (["O4\'", "C1\'", "N9", "C4"], [0, 0, 0, 0], r"$\chi_purine$")
                     }

        dna = DNA()
        result = dna.dihedral_angles_dictionary

        self.assertDictEqual(reference, result)

    def test_distances_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        reference = {"PToP": (["P", "P"], "P to P"),
                     }

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

        reference = {"residual": (["P", "OP1"], ["P", "OP2"], ["P", "O5\'"], ["O5\'", "C5\'"], ["C5\'", "H5\'1"],
                                  ["C5\'", "H5\'2"], ["C5\'", "C4\'"], ["C4\'", "H4\'"], ["C4\'", "O4\'"],
                                  ["C4\'", "C3\'"], ["O4\'", "C1\'"], ["C1\'", "H1\'"], ["C1\'", "C2\'"],
                                  ["C3\'", "H3\'"], ["C3\'", "C2\'"], ["C3\'", "O3\'"], ["C2\'", "H2\'1"],
                                  ["C2\'", "H2\'2"])}

        dna = DNA()
        result = dna.backbone_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_base_bonds_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        reference = {"adenine": (["C1\'", "N9"], ["N9", "C8"], ["N9", "C4"], ["C8", "H8"], ["C8", "N7"], ["N7", "C5"],
                                 ["C5", "C6"], ["C5", "C4"], ["C6", "N6"], ["C6", "N1"], ["N6", "H61"], ["N6", "H62"],
                                 ["N1", "C2"], ["C2", "H2"], ["C2", "N3"], ["N3", "C4"]),
                     "cytosine": (["C1\'", "N1"], ["N1", "C6"], ["N1", "C2"], ["C6", "H6"], ["C6", "C5"], ["C5", "H5"],
                                  ["C5", "C4"], ["C4", "N4"], ["C4", "N3"], ["N4", "H41"], ["N4", "H42"], ["N3", "C2"],
                                  ["C2", "O2"]),
                     "guanine": (["C1\'", "N9"], ["N9", "C8"], ["N9", "C4"], ["C8", "H8"], ["C8", "N7"], ["N7", "C5"],
                                 ["C5", "C6"], ["C5", "C4"], ["C6", "O6"], ["C6", "N1"], ["N1", "H1"], ["N1", "C2"],
                                 ["C2", "N2"], ["C2", "N3"], ["N2", "H21"], ["N2", "H22"], ["N3", "C4"]),
                     "thymine": (["C1\'", "N1"], ["N1", "C6"], ["N1", "C2"], ["C6", "H6"], ["C6", "C5"], ["C5", "C5M"],
                                 ["C5M", "H51"], ["C5M", "H52"], ["C5M", "H53"], ["C5", "C4"], ["C4", "O4"],
                                 ["C4", "N3"], ["N3", "H3"], ["N3", "C2"], ["C2", "O2"])
                     }

        dna = DNA()
        result = dna.base_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_donors_dictionary(self):
        # Note: Order of elements is important
        from yeti.dictionaries.molecules.biomolecules import DNA

        ref_atoms_dict = {"adenine": ("H61", "H62"),
                          "cytosine": ("H41", "H42"),
                          "guanine": ("H1", "H21", "H22"),
                          "thymine": ("H3",),
                          "backbone": ()
                          }

        ref_adenine_slots = {"H61": 1,
                             "H62": 1
                             }

        ref_cytosine_slots = {"H41": 1,
                              "H42": 1
                              }

        ref_guanine_slots = {"H1": 1,
                             "H21": 1,
                             "H22": 1
                             }

        ref_thymine_slots = {"H3": 1
                             }

        ref_backbone_slots = {}

        ref_slots_dict = {"adenine": ref_adenine_slots,
                          "cytosine": ref_cytosine_slots,
                          "guanine": ref_guanine_slots,
                          "thymine": ref_thymine_slots,
                          "backbone": ref_backbone_slots
                          }

        reference = {"atoms": ref_atoms_dict,
                     "slots": ref_slots_dict}

        dna = DNA()
        result = dna.donors_dictionary

        self.assertDictEqual(reference, result)

    def test_acceptors_dictionary(self):
        # Note: Order of elements is important
        from yeti.dictionaries.molecules.biomolecules import DNA

        ref_atoms_dict = {"adenine": ("N1", "N3", "N6", "N7"),
                          "cytosine": ("N3", "N4", "O2"),
                          "guanine": ("N3", "N7", "O6"),
                          "thymine": ("O2", "O4"),
                          "backbone": ("O3\'", "O4\'", "O5\'", "OP1", "OP2")
                          }

        ref_adenine_slots = {"N1": 1,
                             "N3": 1,
                             "N6": 1,
                             "N7": 1
                             }

        ref_cytosine_slots = {"O2": 2,
                              "N3": 1,
                              "N4": 1
                              }

        ref_guanine_slots = {"O6": 2,
                             "N3": 1,
                             "N7": 1
                             }

        ref_thymine_slots = {"O2": 2,
                             "O4": 2
                             }

        ref_backbone_slots = {"OP1": 2,
                              "OP2": 2,
                              "O3\'": 2,
                              "O4\'": 2,
                              "O5\'": 2}

        ref_slots_dict = {"adenine": ref_adenine_slots,
                          "cytosine": ref_cytosine_slots,
                          "guanine": ref_guanine_slots,
                          "thymine": ref_thymine_slots,
                          "backbone": ref_backbone_slots}

        reference = {"atoms": ref_atoms_dict,
                     "slots": ref_slots_dict}

        dna = DNA()
        result = dna.acceptors_dictionary

        self.assertDictEqual(reference, result)

    def test_base_pairs_dictionary(self):
        from yeti.dictionaries.molecules.biomolecules import DNA

        watson_crick_reference = {"adenine_thymine": (["N6", "O4"], ["N1", "N3"]),
                                  "cytosine_guanine": (["N4", "O6"], ["N3", "N1"], ["O2", "N2"])}

        reference = {"watson-crick": watson_crick_reference}

        dna = DNA()
        result = dna.base_pairs_dictionary

        self.assertDictEqual(reference, result)
