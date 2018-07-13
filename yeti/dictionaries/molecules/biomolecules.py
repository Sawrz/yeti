# Reference for conventions is: Molecular Modeling and Simulation - An Interdisciplinary Guide - 2nd Edition by
# Prof. Tamar Schlick


class BiomoleculesException(Exception):
    pass


# TODO: keys need to be strings
class Biomolecule(object):
    def __init__(self):
        """
        Constructor class for all biomolecules in dictionary.
        """

        # ABBREVIATION
        self.abbreviation_dictionary = {}

        # MEASURES
        self.dihedral_angles_dictionary = {}
        self.distances_dictionary = {}

        # COVALENT BONDS
        self.bonds_between_residues = (None, None)
        self.backbone_bonds_dictionary = {}
        self.base_bonds_dictionary = {}

        # HYDROGEN BONDS
        self.donors_dictionary = {}
        self.acceptors_dictionary = {}

    def set_bonds_between_residues(self, atom_1, atom_2):
        """
        Set the bond which connects two residues with each other.

        :param atom_1: Name of first atom in bond.
        :type atom_1: str
        :param atom_2: Name of second atom in bond.
        :type atom_2: str
        """

        if type(atom_1) is not str or type(atom_2) is not str:
            raise BiomoleculesException("Parameter atom_1 and atom_2 need to be strings.")

        self.bonds_between_residues = (atom_1, atom_2)

    def update_abbreviation_dictionary(self, abbreviations):
        """
        Update the dictionary containing all abbreviations.

        :param abbreviations: Dictionary with all abbreviations.
        :type abbreviations: dict

        Example input:
        abbreviations = {"adenine": "A"}
        """

        if type(abbreviations) is not dict:
            raise BiomoleculesException("Parameter abbreviations need to be a dictionary.")

        if not all(type(key) is str for key in abbreviations.keys()):
            raise BiomoleculesException("Keys of abbreviations need to be strings.")

        if not all(type(value) is str for value in abbreviations.values()):
            raise BiomoleculesException("Values of abbreviations need to be strings.")

        self.abbreviation_dictionary.update(abbreviations)

    def update_dihedral_angle_dictionary(self, dihedral_angles):
        """
        Update dictionary of dihedral angles.

        :param dihedral_angles: Dictionary of dihedral angles containing atom names, residue dependence and Latex
                                code for the angle name.
        :type dihedral_angles: dict of (tuple of str, tuple of int, str)

        Example input:
        dihedral_angles = {"alpha": (("O3\'", "P", "O5\'", "C5\'"), (-1, 0, 0, 0), r"$\alpha$")}
        """

        if type(dihedral_angles) is not dict:
            raise BiomoleculesException("Parameter dihedral_angles need to be a dictionary.")

        if not all(type(key) is str for key in dihedral_angles.keys()):
            raise BiomoleculesException("Keys of dihedral_angles need to be strings.")

        if not all(type(value) is tuple for value in dihedral_angles.values()):
            raise BiomoleculesException("Values of dihedral_angles need to be tuples.")

        if not all(type(value[0]) is tuple for value in dihedral_angles.values()):
            raise BiomoleculesException("First Element need to be a tuple.")

        if not all(len(value[0]) == 4 for value in dihedral_angles.values()):
            raise BiomoleculesException("Exactly four strings for first element allowed.")

        if not all(type(atom_name) is str for value in dihedral_angles.values() for atom_name in value[0]):
            raise BiomoleculesException("First elements tuple can only contain strings.")

        if not all(type(value[1]) is tuple for value in dihedral_angles.values()):
            raise BiomoleculesException("Second Element need to be a tuple.")

        if not all(len(value[1]) == 4 for value in dihedral_angles.values()):
            raise BiomoleculesException("Exactly four integers for second element allowed.")

        if not all(type(atom_name) is int for value in dihedral_angles.values() for atom_name in value[1]):
            raise BiomoleculesException("Second elements tuple can only contain integers.")

        if not all(type(value[2]) is str for value in dihedral_angles.values()):
            raise BiomoleculesException("Third element need to be a string.")

        self.dihedral_angles_dictionary.update(dihedral_angles)

    def update_distances_dictionary(self, distances):
        """
        Update dictionary of distances.

        :param distances: Dictionary of distances containing atom names and Latex code for the distance name.
        :type distances: dict of (tuple of str, str)

        Example input:
        distances = {"PToP": (("P", "P"), "P to P")}
        """

        if type(distances) is not dict:
            raise BiomoleculesException("Parameter distance need to be a dictionary.")

        if not all(type(key) is str for key in distances.keys()):
            raise BiomoleculesException("Keys of distances need to be strings.")

        if not all(type(value) is tuple for value in distances.values()):
            raise BiomoleculesException("Values of distances need to be tuples.")

        if not all(type(value[0]) is tuple for value in distances.values()):
            raise BiomoleculesException("First element need to be a tuple.")

        if not all(len(value[0]) == 2 for value in distances.values()):
            raise BiomoleculesException("Exactly two strings for first element allowed.")

        if not all(type(atom_name) is str for value in distances.values() for atom_name in value[0]):
            raise BiomoleculesException("First element only contains strings.")

        if not all(type(value[1]) is str for value in distances.values()):
            raise BiomoleculesException("Second element need to be a string.")

        self.distances_dictionary.update(distances)

    def update_backbone_bonds_dictionary(self, backbone_bonds):
        if type(backbone_bonds) is not dict:
            raise BiomoleculesException("Parameter backbone_bonds need to be a dictionary.")

        if not all(type(key) is str for key in backbone_bonds.keys()):
            raise BiomoleculesException("Keys of backbone_bonds need to be strings.")

        if not all(type(value) is tuple for value in backbone_bonds.values()):
            raise BiomoleculesException("Values of backbone_bonds need to be tuples.")

        if not all(type(bond) is tuple for value in backbone_bonds.values() for bond in value):
            raise BiomoleculesException("Bond type tuple only contains other tuples.")

        if not all(type(atom) is str for value in backbone_bonds.values() for bond in value for atom in bond):
            raise BiomoleculesException("Elements of bond tuples need to be strings.")

        if not all(len(bond) == 2 for value in backbone_bonds.values() for bond in value):
            raise BiomoleculesException("Exactly two strings for bond tuple are allowed.")

        if not all(type(atom_name) is str for value in backbone_bonds.values() for atom_name in value[0]):
            raise BiomoleculesException("First element only contains strings.")

        self.backbone_bonds_dictionary.update(backbone_bonds)

    def update_base_bonds_dictionary(self, new_bases):
        if type(new_bases) is not dict:
            raise BiomoleculesException("Parameter new_bases need to be a dictionary.")

        if not all(type(value) is tuple for value in new_bases.values()):
            raise BiomoleculesException("Values of new_bases need to be tuples.")

        if not all(type(bond) is tuple for value in new_bases.values() for bond in value):
            raise BiomoleculesException("Bond type tuple only contains other tuples.")

        if not all(type(atom) is str for value in new_bases.values() for bond in value for atom in bond):
            raise BiomoleculesException("Elements of bond tuples need to be strings.")

        if not all(len(bond) == 2 for value in new_bases.values() for bond in value):
            raise BiomoleculesException("Exactly two strings for bond tuple are allowed.")

        if not all(type(atom_name) is str for value in new_bases.values() for atom_name in value[0]):
            raise BiomoleculesException("First element only contains strings.")

        self.base_bonds_dictionary.update(new_bases)

    def update_hydrogen_bond_dictionary(self, atoms, update_donors=True):
        if type(atoms) is not dict:
            raise BiomoleculesException("Parameter atoms need to be a dictionary.")

        if not all(type(value) is dict for value in atoms.values()):
            raise BiomoleculesException("Value of atoms need to be a dictionary.")

        if not all(type(slot) is int for value_dict in atoms.values() for slot in value_dict.values()):
            raise BiomoleculesException("Values of the value dictionary need to be integers.")

        if update_donors:
            dictionary = self.donors_dictionary
        else:
            dictionary = self.acceptors_dictionary

        dictionary.update(atoms)


class NucleicAcid(Biomolecule):
    def __init__(self):
        """
        Abstract class for RNA and DNA.
        """

        super(NucleicAcid, self).__init__()

        # abbreviations
        abbreviations = {"adenine": "A",
                         "guanine": "G",
                         "cytosine": "C"}

        self.update_abbreviation_dictionary(abbreviations=abbreviations)

        # MEASURES
        # Dihedral Angles
        dihedral_angles_dictionary = {"alpha": (("O3\'", "P", "O5\'", "C5\'"), (-1, 0, 0, 0), r"$\alpha$"),
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

        self.update_dihedral_angle_dictionary(dihedral_angles=dihedral_angles_dictionary)

        # Distances
        distance_dict = {"PToP": (("P", "P"), "P to P")}

        self.update_distances_dictionary(distances=distance_dict)

        # COVALENT BONDS
        self.set_bonds_between_residues("O3\'", "P")

        base_bonds_dictionary = {"adenine": (("C1\'", "N9"), ("N9", "C8"), ("N9", "C4"), ("C8", "H8"),
                                             ("C8", "N7"), ("N7", "C5"), ("C5", "C6"), ("C5", "C4"),
                                             ("C6", "N6"), ("C6", "N1"), ("N6", "H61"), ("N6", "H62"),
                                             ("N1", "C2"), ("C2", "H2"), ("C2", "N3"), ("N3", "C4")),
                                 "cytosine": (("C1\'", "N1"), ("N1", "C6"), ("N1", "C2"), ("C6", "H6"),
                                              ("C6", "C5"), ("C5", "H5"), ("C5", "C4"), ("C4", "N4"),
                                              ("C4", "N3"), ("N4", "H41"), ("N4", "H42"), ("N3", "C2"),
                                              ("C2", "O2")),
                                 "guanine": (("C1\'", "N9"), ("N9", "C8"), ("N9", "C4"), ("C8", "H8"),
                                             ("C8", "N7"), ("N7", "C5"), ("C5", "C6"), ("C5", "C4"),
                                             ("C6", "O6"), ("C6", "N1"), ("N1", "H1"), ("N1", "C2"),
                                             ("C2", "N2"), ("C2", "N3"), ("N2", "H21"), ("N2", "H22"),
                                             ("N3", "C4"))
                                 }

        self.update_base_bonds_dictionary(new_bases=base_bonds_dictionary)

        # HYDROGEN BONDS
        # Donors
        donors_dict = {"adenine": {"H61": 1,
                                   "H62": 1},
                       "cytosine": {"H41": 1,
                                    "H42": 1},
                       "guanine": {"H1": 1,
                                   "H21": 1,
                                   "H22": 1}
                       }

        self.update_hydrogen_bond_dictionary(atoms=donors_dict, update_donors=True)

        # Acceptors
        acceptors_dict = {"adenine": {"N1": 1,
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
                                      }
                          }

        self.update_hydrogen_bond_dictionary(atoms=acceptors_dict, update_donors=False)

        # BASE PAIRS
        watson_crick_dictionary = {"cytosine_guanine": (("N4", "O6"), ("N3", "N1"), ("O2", "N2"))}
        self.base_pairs_dictionary = {"watson-crick": watson_crick_dictionary}

    def update_base_pairs_dictionary(self, new_base_pairs):
        if type(new_base_pairs) is not dict:
            raise BiomoleculesException("Parameter new_base_pairs need to be a dictionary.")

        if not all(type(value) is dict for value in new_base_pairs.values()):
            raise BiomoleculesException("Value of new_base_pairs need to be a dictionary.")

        if not all(type(base_pairs) is tuple for value in new_base_pairs.values() for base_pairs in value.values()):
            raise BiomoleculesException("Value of the base pair dictionary need to be a tuple.")

        if not all(type(bond) is tuple for value in new_base_pairs.values() for base_pairs in value.values() for bond in
                   base_pairs):
            raise BiomoleculesException("Base pair bonds need to be tuples.")

        if not all(len(bond) == 2 for value in new_base_pairs.values() for base_pairs in value.values() for bond in
                   base_pairs):
            raise BiomoleculesException("Base pair bonds contains only two strings.")

        if not all(
                type(atom_name) is str for value in new_base_pairs.values() for base_pairs in value.values() for bond in
                base_pairs for atom_name in bond):
            raise BiomoleculesException("Base pair bonds contains only two strings.")

        base_pair_types = tuple(self.base_pairs_dictionary.keys())
        new_base_pair_types = tuple(new_base_pairs.keys())

        for new_type in new_base_pair_types:
            if new_type in base_pair_types:
                self.base_pairs_dictionary[new_type].update(new_base_pairs[new_type])
            else:
                self.base_pairs_dictionary[new_type] = new_base_pairs[new_type]


class RNA(NucleicAcid):

    def __init__(self):
        super(RNA, self).__init__()

        # abbreviations
        abbreviations = {"uracil": "U"}

        self.update_abbreviation_dictionary(abbreviations=abbreviations)

        # COVALENT BONDS
        # Backbone Bonds
        full_backbone_bond_list = [("P", "OP1"), ("P", "OP2"), ("P", "O5\'"), ("O5\'", "C5\'"), ("C5\'", "H5\'1"),
                                   ("C5\'", "H5\'2"), ("C5\'", "C4\'"), ("C4\'", "H4\'"), ("C4\'", "O4\'"),
                                   ("C4\'", "C3\'"), ("O4\'", "C1\'"), ("C1\'", "H1\'"), ("C1\'", "C2\'"),
                                   ("C3\'", "H3\'"), ("C3\'", "C2\'"), ("C3\'", "O3\'"), ("C2\'", "H2\'1"),
                                   ("C2\'", "O2\'"), ("O2\'", "HO\'2"), ("O5\'", "H5T"), ("O3'", "HO3\'")]

        start_backbone_bond_list = full_backbone_bond_list.copy()
        for bond in full_backbone_bond_list:
            if "P" in bond:
                start_backbone_bond_list.remove(bond)

        backbone_bonds = {"start": tuple(start_backbone_bond_list),
                          "residual": tuple(full_backbone_bond_list)
                          }

        self.update_backbone_bonds_dictionary(backbone_bonds=backbone_bonds)

        # Base Bonds
        base = {"uracil": (("C1\'", "N1"), ("N1", "C6"), ("N1", "C2"), ("C6", "H6"), ("C6", "C5"), ("C5", "H5"),
                           ("C5", "C4"), ("C4", "O4"), ("C4", "N3"), ("N3", "H3"), ("N3", "C2"), ("C2", "O2"))}

        self.update_base_bonds_dictionary(new_bases=base)

        # HYDROGEN BONDS
        # Donors
        donors = {"uracil": {"H3": 1},
                  "backbone": {"HO\'2": 1}}

        self.update_hydrogen_bond_dictionary(atoms=donors, update_donors=True)

        # Acceptors
        acceptors = {"uracil": {"O2": 2,
                                "O4": 2},
                     "backbone": {"OP1": 2,
                                  "OP2": 2,
                                  "O2\'": 2,
                                  "O3\'": 2,
                                  "O4\'": 2,
                                  "O5\'": 2}}

        self.update_hydrogen_bond_dictionary(atoms=acceptors, update_donors=False)

        # BASE PAIRS
        new_base_pairs = {"watson-crick": {"adenine_uracil": (("N6", "O4"), ("N1", "N3"))}}

        self.update_base_pairs_dictionary(new_base_pairs=new_base_pairs)


class DNA(NucleicAcid):

    def __init__(self):
        super(DNA, self).__init__()

        # abbreviations
        abbreviations = {"thymine": "T"}

        self.update_abbreviation_dictionary(abbreviations=abbreviations)

        # COVALENT BONDS
        # Backbone Bonds
        full_backbone_bond_list = [("P", "OP1"), ("P", "OP2"), ("P", "O5\'"), ("O5\'", "C5\'"), ("C5\'", "H5\'1"),
                                   ("C5\'", "H5\'2"), ("C5\'", "C4\'"), ("C4\'", "H4\'"), ("C4\'", "O4\'"),
                                   ("C4\'", "C3\'"), ("O4\'", "C1\'"), ("C1\'", "H1\'"), ("C1\'", "C2\'"),
                                   ("C3\'", "H3\'"), ("C3\'", "C2\'"), ("C3\'", "O3\'"), ("C2\'", "H2\'1"),
                                   ("C2\'", "H2\'2")]

        backbone_bonds = {"residual": tuple(full_backbone_bond_list)
                          }

        self.update_backbone_bonds_dictionary(backbone_bonds=backbone_bonds)

        # Base Bonds
        base = {"thymine": (("C1\'", "N1"), ("N1", "C6"), ("N1", "C2"), ("C6", "H6"), ("C6", "C5"), ("C5", "C5M"),
                            ("C5M", "H51"), ("C5M", "H52"), ("C5M", "H53"), ("C5", "C4"), ("C4", "O4"), ("C4", "N3"),
                            ("N3", "H3"), ("N3", "C2"), ("C2", "O2"))}

        self.update_base_bonds_dictionary(new_bases=base)

        # HYDROGEN BONDS
        # Donors
        donors = {"thymine": {"H3": 1},
                  "backbone": {}}

        self.update_hydrogen_bond_dictionary(atoms=donors, update_donors=True)

        # Acceptors

        acceptors = {"thymine": {"O2": 2,
                                 "O4": 2},
                     "backbone": {"OP1": 2,
                                  "OP2": 2,
                                  "O3\'": 2,
                                  "O4\'": 2,
                                  "O5\'": 2}}

        self.update_hydrogen_bond_dictionary(atoms=acceptors, update_donors=False)

        # BASE PAIRS
        new_base_pairs = {"watson-crick": {"adenine_thymine": (("N6", "O4"), ("N1", "N3"))}}

        self.update_base_pairs_dictionary(new_base_pairs=new_base_pairs)


class Protein(Biomolecule):
    def __init__(self):
        super(Protein, self).__init__()

        self.bonds_between_residues = ["C", "N"]

        self.__get_abbreviations_dictionary()

    def __get_abbreviations_dictionary(self):
        self.abbreviations = {"glycine": "Gly",
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
                              "tryptophan": "Trp"}

    def __get_dihedral_angles_dictionary(self):
        # TODO: Get a clear idea about rotameric structures etc.
        # TODO: Important non-dehedral angles?

        backbone_dihedral_angles_dictionary = {"psi": (["N", "C_alpha", "C", "N"], [0, 0, 0, 1], r"$\psi$"),
                                               "phi": (["C_alpha", "C", "N", "C_alpha"], [-1, -1, 0, 0], r"$\phi$"),
                                               "omega": (["C", "N", "C_alpha", "C"], [-1, 0, 0, 0], r"$\omega$")
                                               }

        rotameric_structures_exclude = ["glycine", "alanine"]
        rotameric_structures_dictionary = {"xi_1": ([], [0, 0, 0, 1], r"$\xi_1$"),
                                           }

    def __get_distances_dictionary(self):
        # TODO: Are there important distances? C_alpha - C_alpha
        pass

    def __get_residue_bonds_dictionary(self):
        # TODO: Get a clear idea about naming conventions
        backbone_bonds = [["N", "H"], ["N", "C_alpha"], ["C_alpha", "H_alpha_1"], ["C_alpha", "C"], ["C", "O"]]

        amino_acids = {"glycine": [["C_alpha", "H_alpha_2"]],
                       "alanine": [["C_alpha", "C_beta"], ["C_beta", "H_beta_1"], ["C_beta", "H_beta_2"],
                                   ["C_beta", "H_beta_3"]],
                       "valine": [["C_alpha", "C_beta"], ["C_beta", "H_beta_1"], ["C_beta", "C_gamma"],
                                  ["C_gamma", "H_gamma_1"], ["C_gamma", "H_gamma_2"], ["C_gamma", "H_gamma_3"],
                                  ["C_beta", "C_delta"], ["C_delta", "H_gamma_1"], ["C_delta", "H_gamma_2"],
                                  ["C_delta", "H_gamma_3"]],
                       "leucine": [["C_alpha", "C_beta"], ["C_beta", "H_beta_1"], ["C_beta", "H_beta_2"],
                                   ["C_beta", "C_gamma"], ["C_gamma", "H_gamma_1"], ["C_gamma", "C_delta"],
                                   ["C_delta", "H_gamma_1"], ["C_delta", "H_gamma_2"], ["C_delta", "H_gamma_3"],
                                   ["C_gamma", "C_epsilon"], ["C_epsilon", "H_epsilon_1"], ["C_epsilon", "H_epsilon_2"],
                                   ["C_epsilon", "H_epsilon_3"]],
                       "isoleucine": [[]],
                       "serine": [[]],
                       "threonine": [[]],
                       "proline": [[]],
                       "aspartic acid": [[]],
                       "glutamic acid": [[]],
                       "asparagine": [[]],
                       "glutamine": [[]],
                       "methionine": [[]],
                       "cysteine": [[]],
                       "lysine": [[]],
                       "arginine": [[]],
                       "histidine": [[]],
                       "phenylalanine": [[]],
                       "tyrosine": [[]],
                       "tryptophan": [[]]
                       }

        protein_beginning = {"standard": [["H_start", "N"]]}

        protein_end = {"standard": [["C", "O_end"], ["O_end", "H"]]}

    def __get_donors_dictionary(self):
        # TODO: What about Donors?
        pass

    def __get_acceptors_dictionary(self):
        # TODO: What about Acceptors?
        pass
