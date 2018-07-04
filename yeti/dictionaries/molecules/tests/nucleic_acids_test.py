from unittest import TestCase


class TestRNA(TestCase):
    def test_get_dihedral_angles_dictionary(self):
        from yeti.dictionaries.molecules.nucleic_acids import RNA

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

    def test_get_distances_dictionary(self):
        from yeti.dictionaries.molecules.nucleic_acids import RNA

        reference = {"PToP": (["P", "P"], "P to P"),
                     }

        rna = RNA()
        result = rna.distances_dict

        self.assertDictEqual(reference, result)

    def test_get_residue_bonds_dictionary(self):
        from yeti.dictionaries.molecules.nucleic_acids import RNA

        reference = {"adenine": (["C1\'", "N9"], ["N9", "C8"], ["N9", "C4"], ["C8", "H8"], ["C8", "N7"],
                                 ["N7", "C5"], ["C5", "C6"], ["C5", "C4"], ["C6", "N6"], ["C6", "N1"],
                                 ["N6", "H61"], ["N6", "H62"], ["N1", "C2"], ["C2", "H2"], ["C2", "N3"],
                                 ["N3", "C4"], ["P", "OP1"], ["P", "OP2"], ["P", "O5\'"], ["O5\'", "C5\'"],
                                 ["C5\'", "H5\'1"], ["C5\'", "H5\'2"], ["C5\'", "C4\'"], ["C4\'", "H4\'"],
                                 ["C4\'", "O4\'"], ["C4\'", "C3\'"], ["O4\'", "C1\'"], ["C1\'", "H1\'"],
                                 ["C1\'", "C2\'"], ["C3\'", "H3\'"], ["C3\'", "C2\'"], ["C3\'", "O3\'"],
                                 ["C2\'", "H2\'1"], ["C2\'", "O2\'"], ["O2\'", "HO\'2"], ["O5\'", "H5T"],
                                 ["O3'", "HO3\'"]),
                     "cytosine": (["C1\'", "N1"], ["N1", "C6"], ["N1", "C2"], ["C6", "H6"],
                                  ["C6", "C5"], ["C5", "H5"], ["C5", "C4"], ["C4", "N4"], ["C4", "N3"],
                                  ["N4", "H41"], ["N4", "H42"], ["N3", "C2"], ["C2", "O2"], ["P", "OP1"],
                                  ["P", "OP2"], ["P", "O5\'"], ["O5\'", "C5\'"], ["C5\'", "H5\'1"],
                                  ["C5\'", "H5\'2"], ["C5\'", "C4\'"], ["C4\'", "H4\'"], ["C4\'", "O4\'"],
                                  ["C4\'", "C3\'"], ["O4\'", "C1\'"], ["C1\'", "H1\'"], ["C1\'", "C2\'"],
                                  ["C3\'", "H3\'"], ["C3\'", "C2\'"], ["C3\'", "O3\'"], ["C2\'", "H2\'1"],
                                  ["C2\'", "O2\'"], ["O2\'", "HO\'2"], ["O5\'", "H5T"], ["O3'", "HO3\'"]),
                     "guanine": (["C1\'", "N9"], ["N9", "C8"], ["N9", "C4"], ["C8", "H8"], ["C8", "N7"],
                                 ["N7", "C5"], ["C5", "C6"], ["C5", "C4"], ["C6", "O6"], ["C6", "N1"],
                                 ["N1", "H1"], ["N1", "C2"], ["C2", "N2"], ["C2", "N3"], ["N2", "H21"],
                                 ["N2", "H22"], ["N3", "C4"], ["P", "OP1"], ["P", "OP2"], ["P", "O5\'"],
                                 ["O5\'", "C5\'"], ["C5\'", "H5\'1"], ["C5\'", "H5\'2"], ["C5\'", "C4\'"],
                                 ["C4\'", "H4\'"], ["C4\'", "O4\'"], ["C4\'", "C3\'"], ["O4\'", "C1\'"],
                                 ["C1\'", "H1\'"], ["C1\'", "C2\'"], ["C3\'", "H3\'"], ["C3\'", "C2\'"],
                                 ["C3\'", "O3\'"], ["C2\'", "H2\'1"], ["C2\'", "O2\'"], ["O2\'", "HO\'2"],
                                 ["O5\'", "H5T"], ["O3'", "HO3\'"]),
                     "uracil": (["C1\'", "N1"], ["N1", "C6"], ["N1", "C2"], ["C6", "H6"], ["C6", "C5"], ["C5", "H5"],
                                ["C5", "C4"], ["C4", "O4"], ["C4", "N3"], ["N3", "H3"], ["N3", "C2"], ["C2", "O2"],
                                ["P", "OP1"], ["P", "OP2"], ["P", "O5\'"], ["O5\'", "C5\'"], ["C5\'", "H5\'1"],
                                ["C5\'", "H5\'2"], ["C5\'", "C4\'"], ["C4\'", "H4\'"], ["C4\'", "O4\'"],
                                ["C4\'", "C3\'"], ["O4\'", "C1\'"], ["C1\'", "H1\'"], ["C1\'", "C2\'"],
                                ["C3\'", "H3\'"], ["C3\'", "C2\'"], ["C3\'", "O3\'"], ["C2\'", "H2\'1"],
                                ["C2\'", "O2\'"], ["O2\'", "HO\'2"], ["O5\'", "H5T"], ["O3'", "HO3\'"])
                     }

        rna = RNA()
        result = rna.residue_bonds_dictionary

        self.assertDictEqual(reference, result)

    def test_get_donors_dictionary(self):
        from yeti.dictionaries.molecules.nucleic_acids import RNA

        ref_atoms_dict = {"adenine": ("H61", "H62", "HO\'2"),
                          "cytosine": ("H41", "H42", "HO\'2"),
                          "guanine": ("H1", "H21", "H22", "HO\'2"),
                          "uracil": ("H3", "HO\'2")
                          }

        ref_adenine_slots = {"H61": 1,
                             "H62": 1,
                             "HO\'2": 1
                             }

        ref_cytosine_slots = {"H41": 1,
                              "H42": 1,
                              "HO\'2": 1
                              }

        ref_guanine_slots = {"H1": 1,
                             "H21": 1,
                             "H22": 1,
                             "HO\'2": 1,
                             }

        ref_uracil_slots = {"H3": 1,
                            "HO\'2": 1
                            }

        ref_slots_dict = {"adenine": ref_adenine_slots,
                          "cytosine": ref_cytosine_slots,
                          "guanine": ref_guanine_slots,
                          "uracil": ref_uracil_slots
                          }

        reference = {"atoms": ref_atoms_dict,
                     "slots": ref_slots_dict}

        rna = RNA()
        result = rna.donors_dictionary

        self.assertDictEqual(reference, result)

    def test_get_acceptors_dictionary(self):
        from yeti.dictionaries.molecules.nucleic_acids import RNA

        ref_atoms_dict = {"adenine": ("N1", "N3", "N6", "N7", "O2\'", "O3\'", "O4\'", "O5\'", "OP1", "OP2"),
                          "cytosine": ("N3", "N4", "O2", "O2\'", "O3\'", "O4\'", "O5\'", "OP1", "OP2"),
                          "guanine": ("N3", "N7", "O6", "O2\'", "O3\'", "O4\'", "O5\'", "OP1", "OP2"),
                          "uracil": ("O2", "O4", "O2\'", "O3\'", "O4\'", "O5\'", "OP1", "OP2")
                          }

        ref_adenine_slots = {"N1": 1,
                             "N3": 1,
                             "N6": 1,
                             "N7": 1,
                             "OP1": 2,
                             "OP2": 2,
                             "O2\'": 2,
                             "O3\'": 2,
                             "O4\'": 2,
                             "O5\'": 2
                             }

        ref_cytosine_slots = {"O2": 2,
                              "N3": 1,
                              "N4": 1,
                              "OP1": 2,
                              "OP2": 2,
                              "O2\'": 2,
                              "O3\'": 2,
                              "O4\'": 2,
                              "O5\'": 2
                              }

        ref_guanine_slots = {"O6": 2,
                             "N3": 1,
                             "N7": 1,
                             "OP1": 2,
                             "OP2": 2,
                             "O2\'": 2,
                             "O3\'": 2,
                             "O4\'": 2,
                             "O5\'": 2
                             }

        ref_uracil_slots = {"O2": 2,
                            "O4": 2,
                            "OP1": 2,
                            "OP2": 2,
                            "O2\'": 2,
                            "O3\'": 2,
                            "O4\'": 2,
                            "O5\'": 2
                            }

        ref_slots_dict = {"adenine": ref_adenine_slots,
                          "cytosine": ref_cytosine_slots,
                          "guanine": ref_guanine_slots,
                          "uracil": ref_uracil_slots
                          }

        reference = {"atoms": ref_atoms_dict,
                     "slots": ref_slots_dict}

        rna = RNA()
        result = rna.donors_dictionary

        self.assertDictEqual(reference, result)

    def test_get_base_pairs_dictionary(self):
        from yeti.dictionaries.molecules.nucleic_acids import RNA

        watson_crick_reference = {"adenine_uracil": (["N6", "O4"], ["N1", "N3"]),
                                  "cytosine_guanine": (["N4", "O6"], ["N3", "N1"], ["O2", "N2"])}

        reference = {"watson-crick": watson_crick_reference}

        rna = RNA()
        result = rna.base_pairs_dictionary

        self.assertDictEqual(reference, result)
