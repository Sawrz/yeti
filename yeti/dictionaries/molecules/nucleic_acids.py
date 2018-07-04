# Reference for conventions is: Molecular Modeling and Simulation - An Interdisciplinary Guide - 2nd Edition by
# Prof. Tamar Schlick


class NucleicAcid(object):
    def __init__(self):
        """
        The basic building class for standard RNA and DNA.
        """

        self.dihedral_angles_dictionary = self.__get_dihedral_angles_dictionary()
        self.distances_dict = self.__get_distances_dictionary()
        self.bonds_between_residues = ["O3\'", "P"]

        self.residue_bonds_dictionary = self.__get_residue_bonds_dictionary()
        self.donors_dictionary = self.__get_donors_dictionary()
        self.acceptors_dictionary = self.__get_acceptors_dictionary()

        self.base_pairs_dictionary = self.__get_base_pairs_dictionary()

    def __get_dihedral_angles_dictionary(self):
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

    def __get_distances_dictionary(self):
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
        """
        Creates an attribute. That attribute is a dictionary, which contains a list of all bonds within the standard
        bases adenine, cytosine and guanine.
        """

        residue_bonds_dictionary = {"adenine": (["C1\'", "N9"], ["N9", "C8"], ["N9", "C4"], ["C8", "H8"],
                                                ["C8", "N7"], ["N7", "C5"], ["C5", "C6"], ["C5", "C4"],
                                                ["C6", "N6"], ["C6", "N1"], ["N6", "H61"], ["N6", "H62"],
                                                ["N1", "C2"], ["C2", "H2"], ["C2", "N3"], ["N3", "C4"]),
                                    "cytosine": (["C1\'", "N1"], ["N1", "C6"], ["N1", "C2"], ["C6", "H6"],
                                                 ["C6", "C5"], ["C5", "H5"], ["C5", "C4"], ["C4", "N4"],
                                                 ["C4", "N3"], ["N4", "H41"], ["N4", "H42"], ["N3", "C2"],
                                                 ["C2", "O2"]),
                                    "guanine": (["C1\'", "N9"], ["N9", "C8"], ["N9", "C4"], ["C8", "H8"],
                                                ["C8", "N7"], ["N7", "C5"], ["C5", "C6"], ["C5", "C4"],
                                                ["C6", "O6"], ["C6", "N1"], ["N1", "H1"], ["N1", "C2"],
                                                ["C2", "N2"], ["C2", "N3"], ["N2", "H21"], ["N2", "H22"],
                                                ["N3", "C4"])
                                    }

        return residue_bonds_dictionary

    def __get_donors_dictionary(self):
        """
        Creates an attribute. That attribute is a dictionary, which contains further dictionaries for donor atoms,
        bonds and slots for the standard bases adenine, cytosine and guanine.
        """

        donor_base_atoms_dict = {"adenine": ("H61", "H62"),
                                 "cytosine": ("H41", "H42"),
                                 "guanine": ("H1", "H21", "H22")
                                 }

        donor_slots_dict = {"adenine": {"H61": 1,
                                        "H62": 1},
                            "cytosine": {"H41": 1,
                                         "H42": 1},
                            "guanine": {"H1": 1,
                                        "H21": 1,
                                        "H22": 1}
                            }

        donors_dictionary = {"atoms": donor_base_atoms_dict,
                             "slots": donor_slots_dict}

        return donors_dictionary

    def __get_acceptors_dictionary(self):
        """
        Creates an attribute. That attribute is a dictionary, which contains further dictionaries for acceptor atoms,
        bonds and slots for the standard bases adenine, cytosine and guanine.
        """

        acceptor_base_atoms_dict = {"adenine": ("N1", "N3", "N6", "N7"),
                                    "cytosine": ("N3", "N4", "O2"),
                                    "guanine": ("N3", "N7", "O6")
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

        acceptor_slots_dict = {"adenine": adenine_slots,
                               "cytosine": cytosine_slots,
                               "guanine": guanine_slots
                               }

        acceptors_dictionary = {"atoms": acceptor_base_atoms_dict,
                                "slots": acceptor_slots_dict}

        return acceptors_dictionary

    def __get_base_pairs_dictionary(self):
        watson_crick_dictionary = {"cytosine_guanine": (["N4", "O6"], ["N3", "N1"], ["O2", "N2"])}

        base_pairs_dictionary = {"watson-crick": watson_crick_dictionary}

        return base_pairs_dictionary


class RNA(NucleicAcid):
    def __init__(self):
        super(RNA, self).__init__()

        # run internal methods
        self.__update_residue_bonds_dictionary()
        self.__update_donors_dictionary()
        self.__update_acceptors_dictionary()
        self.__update_base_pairs_dictionary()

    def __update_residue_bonds_dictionary(self):

        backbone_bonds = [["P", "OP1"], ["P", "OP2"], ["P", "O5\'"], ["O5\'", "C5\'"], ["C5\'", "H5\'1"],
                          ["C5\'", "H5\'2"], ["C5\'", "C4\'"], ["C4\'", "H4\'"],
                          ["C4\'", "O4\'"], ["C4\'", "C3\'"], ["O4\'", "C1\'"], ["C1\'", "H1\'"], ["C1\'", "C2\'"],
                          ["C3\'", "H3\'"], ["C3\'", "C2\'"],
                          ["C3\'", "O3\'"], ["C2\'", "H2\'1"], ["C2\'", "O2\'"], ["O2\'", "HO\'2"], ["O5\'", "H5T"],
                          ["O3'", "HO3\'"]]

        tmp = {"uracil": (["C1\'", "N1"], ["N1", "C6"], ["N1", "C2"], ["C6", "H6"], ["C6", "C5"], ["C5", "H5"],
                          ["C5", "C4"], ["C4", "O4"], ["C4", "N3"], ["N3", "H3"], ["N3", "C2"], ["C2", "O2"])
               }

        self.residue_bonds_dictionary.update(tmp)

        # add backbone bonds
        for key in self.residue_bonds_dictionary:
            for i in range(len(backbone_bonds)):
                self.residue_bonds_dictionary[key] += (backbone_bonds[i],)

    def __update_donors_dictionary(self):
        tmp_atoms = {"uracil": ("H3",)}
        tmp_slots = {"uracil": {"H3": 1}}

        self.donors_dictionary["atoms"].update(tmp_atoms)
        self.donors_dictionary["slots"].update(tmp_slots)

        # add backbone atoms
        backbone_atoms = ("HO\'2",)
        backbone_slots = {"HO\'2": 1}

        for key in self.donors_dictionary["atoms"]:
            self.donors_dictionary["atoms"][key] += backbone_atoms
            self.donors_dictionary["slots"][key].update(backbone_slots)

    def __update_acceptors_dictionary(self):
        tmp_atoms = {"uracil": ("O2", "O4")}
        tmp_slots = {"uracil": {"O2": 2,
                                "O4": 2}}

        self.acceptors_dictionary["atoms"].update(tmp_atoms)
        self.acceptors_dictionary["slots"].update(tmp_slots)

        # add backbone atoms
        backbone_atoms = ("O2\'", "O3\'", "O4\'", "O5\'", "OP1", "OP2")
        backbone_slots = {"OP1": 2,
                          "OP2": 2,
                          "O2\'": 2,
                          "O3\'": 2,
                          "O4\'": 2,
                          "O5\'": 2}

        for key in self.acceptors_dictionary["atoms"]:
            self.acceptors_dictionary["atoms"][key] += backbone_atoms
            self.acceptors_dictionary["slots"][key].update(backbone_slots)

    def __update_base_pairs_dictionary(self):
        pass
