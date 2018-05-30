# Reference for conventions is: Molecular Modeling and Simulation - An Interdisciplinary Guide - 2nd Edition by
# Prof. Tamar Schlick


class NucleicAcid(object):
    def __init__(self):
        self.dihedral_angles_dictionary = self.__get_dihedral_angles_dictionary()
        self.distance_dict = self.__get_distances_dictionary()
        self.bonds_between_residues = ["O3\'", "P"]

    @staticmethod
    def __get_dihedral_angles_dictionary():
        """
        Returns a dictionary of all dihedral angles in Nucleic Acids. The key is the name of the dihedral
        angle, according to conventions. The value is a list with 3 entries.
        First entry: List of the atom names (according to conventions) belonging to the dihedral.
        Second entry: List of relative residue numbers to form a dihedral angle.
        Third entry: Latex code for the dihedral name for plots.

        :rtype: dict
        """
        dihedral_angles_dictionary = {"alpha": (["O3\'", "P", "O5\'", "C5\'"], [-1, 0, 0, 0], r"$\alpha$"),
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

        return dihedral_angles_dictionary

    @staticmethod
    def __get_distances_dictionary():
        """
        Returns a dictionary of all typical distances in Nucleic Acids. The key is the internal naming for the
        distance. The value is a list with 2 entries.
        First entry: List of the atom names (according to conventions) belonging to the distance.
        Second entry: String of the distance name for plots.

        :rtype: dict
        """

        distance_dict = {"PToP": (["P", "P"], "P to P"),
                         }

        return distance_dict

    def __get_residue_bonds_dictionary(self):
        # TODO
        self.base_bond_dictionary = {"adenine": (["C1\'", "N9"], ["N9", "C8"], ["N9", "C4"], ["C8", "H8"], ["C8", "N7"],
                                                 ["N7", "C5"], ["C5", "C6"], ["C5", "C4"], ["C6", "N6"], ["C6", "N1"],
                                                 ["N6", "H61"], ["N6", "H62"], ["N1", "C2"], ["C2", "H2"], ["C2", "N3"],
                                                 ["N3", "C4"]),
                                     "cytosine": (["C1\'", "N1"], ["N1", "C6"], ["N1", "C2"], ["C6", "H6"],
                                                  ["C6", "C5"], ["C5", "H5"], ["C5", "C4"], ["C4", "N4"], ["C4", "N3"],
                                                  ["N4", "H41"], ["N4", "H42"], ["N3", "C2"], ["C2", "O2"]),
                                     "guanine": (["C1\'", "N9"], ["N9", "C8"], ["N9", "C4"], ["C8", "H8"], ["C8", "N7"],
                                                 ["N7", "C5"], ["C5", "C6"], ["C5", "C4"], ["C6", "O6"], ["C6", "N1"],
                                                 ["N1", "H1"], ["N1", "C2"], ["C2", "N2"], ["C2", "N3"], ["N2", "H21"],
                                                 ["N2", "H22"], ["N3", "C4"])
                                     }

    def get_donors_dict(self):
        # TODO
        self.base_bond_donors_dict = {"adenine": (["N6", "H61"], ["N6", "H62"]),
                                      "cytosine": (["N4", "H41"], ["N4", "H42"]),
                                      "guanine": (["N1", "H1"], ["N2", "H21"], ["N2", "H22"])
                                      }

    def get_acceptors_dict(self):
        # TODO
        self.base_bond_acceptors = {"adenine": (),
                                    "cytosine": (["C2", "O2"],),
                                    "guanine": (["C6", "O6"],)
                                    }

        self.base_atom_acceptors_dict = {"adenine": ("N1", "N3", "N6", "N7"),
                                         "cytosine": ("N3", "N4"),
                                         "guanine": ("N3", "N7")
                                         }

        adenine_slots = {"N1": 1,
                         "N3": 1,
                         "N6": 1,
                         "N7": 1
                         }

        cytosine_slots = {"O2": 2,
                          "N3": 1,
                          "N4": 1
                          }

        guanine_slots = {"O6": 2,
                         "N3": 1,
                         "N7": 1
                         }

        self.acceptor_slots = {"adenine": adenine_slots,
                               "cytosine": cytosine_slots,
                               "guanine": guanine_slots
                               }

    def get_base_pairs_dict(self):
        # TODO
        self.watson_crick_base_pair_hbond_partners = {"CG": (["N4", "O6"], ["N3", "N1"], ["O2", "N2"]),
                                                      }

        self.watson_crick_base_pair_donors = {"C": "N4",
                                              "G": ("N1", "N2")
                                              }

        self.watson_crick_base_pair_acceptors = {"C": ("N3", "O2"),
                                                 "G": "O6"
                                                 }


class RNA(NucleicAcid):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.residue_bonds_dictionary = self.__get_residue_bonds_dictionary()

    def __get_residue_bonds_dictionary(self):

        super(RNA, self).__get_residue_bonds_dictionary()

        backbone_bonds = [["P", "OP1"], ["P", "OP2"], ["P", "O5\'"], ["O5\'", "C5\'"], ["C5\'", "H5\'1"],
                          ["C5\'", "H5\'2"], ["C5\'", "C4\'"], ["C4\'", "H4\'"],
                          ["C4\'", "O4\'"], ["C4\'", "C3\'"], ["O4\'", "C1\'"], ["C1\'", "H1\'"], ["C1\'", "C2\'"],
                          ["C3\'", "H3\'"], ["C3\'", "C2\'"],
                          ["C3\'", "O3\'"], ["C2\'", "H2\'1"], ["C2\'", "O2\'"], ["O2\'", "HO\'2"], ["O5\'", "H5T"],
                          ["O3'", "HO3\'"]]

        tmp = {"uracil": (["C1\'", "N1"], ["N1", "C6"], ["N1", "C2"], ["C6", "H6"], ["C6", "C5"], ["C5", "H5"],
                          ["C5", "C4"], ["C4", "O4"], ["C4", "N3"], ["N3", "H3"], ["N3", "C2"], ["C2", "O2"])
               }

        residue_dictionary.update(tmp)

        for key in residue_dictionary:
            for i in range(len(backbone_bonds)):
                residue_dictionary[key] += (backbone_bonds[i],)

        return residue_dictionary


nuc_acid = NucleicAcid()
rna = RNA()

print(nuc_acid.residue_bonds_dictionary)
print(rna.residue_bonds_dictionary)
