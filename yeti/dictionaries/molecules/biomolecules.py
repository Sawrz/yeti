# Reference for conventions is: Molecular Modeling and Simulation - An Interdisciplinary Guide - 2nd Edition by
# Prof. Tamar Schlick


class BiomoleculesException(Exception):
    pass


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
        self.termini_bonds_dictionary = {}
        self.side_chain_bonds_dictionary = {}

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
        """
        Update dictionary which contains information about backbone bonds.

        :param backbone_bonds: Dictionary containing backbone bonds assigned to a name.
        :type backbone_bonds: dict of (tuple of (tuple of str))

        Example input:
        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2"))}
        """

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

        self.backbone_bonds_dictionary.update(backbone_bonds)

    def update_termini_bonds_dictionary(self, termini_bonds):
        """
        Update dictionary which contains information about termini bonds.

        :param termini_bonds: Dictionary containing backbone bonds assigned to a name.
        :type termini_bonds: dict of (tuple of (tuple of str))

        Example input:
        backbone_bonds = {"start": (("P", "OP1"), ("P", "OP2"))}
        """

        if type(termini_bonds) is not dict:
            raise BiomoleculesException("Parameter termini_bonds need to be a dictionary.")

        if not all(type(key) is str for key in termini_bonds.keys()):
            raise BiomoleculesException("Keys of termini_bonds need to be strings.")

        if not all(type(value) is tuple for value in termini_bonds.values()):
            raise BiomoleculesException("Values of termini_bonds need to be tuples.")

        if not all(type(bond) is tuple for value in termini_bonds.values() for bond in value):
            raise BiomoleculesException("Bond type tuple only contains other tuples.")

        if not all(type(atom) is str for value in termini_bonds.values() for bond in value for atom in bond):
            raise BiomoleculesException("Elements of bond tuples need to be strings.")

        if not all(len(bond) == 2 for value in termini_bonds.values() for bond in value):
            raise BiomoleculesException("Exactly two strings for bond tuple are allowed.")

        self.termini_bonds_dictionary.update(termini_bonds)

    def update_side_chain_bonds_dictionary(self, new_side_chain):
        """
        Update dictionary which contains information about side chain bonds.

        :param new_side_chain: Dictionary containing base bonds assigned to a name.
        :type new_side_chain: dict of (tuple of (tuple of str))

        Example input:
        new_bases = {"fantasy_base": (("C1\'", "N1"), ("N1", "C6"))}
        """

        if type(new_side_chain) is not dict:
            raise BiomoleculesException("Parameter new_bases need to be a dictionary.")

        if not all(type(key) is str for key in new_side_chain.keys()):
            raise BiomoleculesException("Keys of new_side_chain need to be strings.")

        if not all(type(value) is tuple for value in new_side_chain.values()):
            raise BiomoleculesException("Values of new_side_chain need to be tuples.")

        if not all(type(bond) is tuple for value in new_side_chain.values() for bond in value):
            raise BiomoleculesException("Bond type tuple only contains other tuples.")

        if not all(type(atom) is str for value in new_side_chain.values() for bond in value for atom in bond):
            raise BiomoleculesException("Elements of bond tuples need to be strings.")

        if not all(len(bond) == 2 for value in new_side_chain.values() for bond in value):
            raise BiomoleculesException("Exactly two strings for bond tuple are allowed.")

        self.side_chain_bonds_dictionary.update(new_side_chain)

    def update_hydrogen_bond_dictionary(self, hydrogen_bond_atoms, update_donors=True):
        """
        Update the dictionary containing all information about hydrogen bond candidates for attachments and backbone.

        :param hydrogen_bond_atoms: Dictionary containing attachment raltaion of atoms and the number of slots. Slots
                                    is the number of hydrogen bonds an atom can have at the same time.
        :type hydrogen_bond_atoms: dict of (dict of int)
        :param update_donors: Update the donor or acceptor dictionary?
        :type update_donors: bool

        Example input:
        hydrogen_bond_atoms = {"uracil": {"H3": 1},
                               "backbone": {"HO\'2": 1}}
        """

        if type(update_donors) is not bool:
            raise BiomoleculesException("Parameter update_donors need to be a boolean.")

        if type(hydrogen_bond_atoms) is not dict:
            raise BiomoleculesException("Parameter hydrogen_bond_atoms need to be a dictionary.")

        if not all(type(key) is str for key in hydrogen_bond_atoms.keys()):
            raise BiomoleculesException("Keys of hydrogen_bond_atoms need to be a strings.")

        if not all(type(value) is dict for value in hydrogen_bond_atoms.values()):
            raise BiomoleculesException("Values of hydrogen_bond_atoms need to be a dictionaries.")

        if not all(type(slot) is int for value_dict in hydrogen_bond_atoms.values() for slot in value_dict.values()):
            raise BiomoleculesException("Values of the value dictionary need to be integers.")

        if update_donors:
            dictionary = self.donors_dictionary
        else:
            dictionary = self.acceptors_dictionary

        dictionary.update(hydrogen_bond_atoms)


class NucleicAcid(Biomolecule):
    def __init__(self):
        """
        Constructor class for RNA and DNA.
        """

        super(NucleicAcid, self).__init__()

        # ABBREVIATION
        abbreviations = {"A": "adenine", "G": "guanine", "C": "cytosine"}

        self.update_abbreviation_dictionary(abbreviations=abbreviations)

        # MEASURES
        # Dihedral Angles
        dihedral_angles_dictionary = {
            "alpha": (("O3\'", "P", "O5\'", "C5\'"), (-1, 0, 0, 0), r"$\alpha$"),
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
        distance_dict = {"P_P": (("P", "P"), "P to P")}

        self.update_distances_dictionary(distances=distance_dict)

        # COVALENT BONDS
        self.set_bonds_between_residues("O3\'", "P")

        base_bonds_dictionary = {
            "adenine": (("C1\'", "N9"), ("N9", "C8"), ("N9", "C4"), ("C8", "H8"), ("C8", "N7"), ("N7", "C5"),
                        ("C5", "C6"), ("C5", "C4"), ("C6", "N6"), ("C6", "N1"), ("N6", "H61"), ("N6", "H62"),
                        ("N1", "C2"), ("C2", "H2"), ("C2", "N3"), ("N3", "C4")),
            "cytosine":
            (("C1\'", "N1"), ("N1", "C6"), ("N1", "C2"), ("C6", "H6"), ("C6", "C5"), ("C5", "H5"), ("C5", "C4"),
             ("C4", "N4"), ("C4", "N3"), ("N4", "H41"), ("N4", "H42"), ("N3", "C2"), ("C2", "O2")),
            "guanine": (("C1\'", "N9"), ("N9", "C8"), ("N9", "C4"), ("C8", "H8"), ("C8", "N7"), ("N7", "C5"),
                        ("C5", "C6"), ("C5", "C4"), ("C6", "O6"), ("C6", "N1"), ("N1", "H1"), ("N1", "C2"),
                        ("C2", "N2"), ("C2", "N3"), ("N2", "H21"), ("N2", "H22"), ("N3", "C4"))
        }

        self.update_side_chain_bonds_dictionary(new_side_chain=base_bonds_dictionary)

        # HYDROGEN BONDS
        # Donors
        donors_dict = {
            "adenine": {
                "H61": 1,
                "H62": 1
            },
            "cytosine": {
                "H41": 1,
                "H42": 1
            },
            "guanine": {
                "H1": 1,
                "H21": 1,
                "H22": 1
            }
        }

        self.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=donors_dict, update_donors=True)

        # Acceptors
        acceptors_dict = {
            "adenine": {
                "N1": 1,
                "N3": 1,
                "N6": 1,
                "N7": 1
            },
            "cytosine": {
                "O2": 2,
                "N3": 1,
                "N4": 1
            },
            "guanine": {
                "O6": 2,
                "N3": 1,
                "N7": 1
            }
        }

        self.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=acceptors_dict, update_donors=False)

        # BASE PAIRS
        watson_crick_dictionary = {"cytosine_guanine": (("N4", "O6"), ("N3", "N1"), ("O2", "N2"))}
        self.base_pairs_dictionary = {"watson-crick": watson_crick_dictionary}

    def update_base_pairs_dictionary(self, new_base_pairs):
        """
        Update dictionary containing information about base pairs.

        :param new_base_pairs: Dictionary with base pair information
        :type new_base_pairs: dict of (dict of (tuple of (tuple of str)))

        Example input:
        new_base_pairs =  {"watson-crick": {"cytosine_guanine": (("N4", "O6"), ("N3", "N1"), ("O2", "N2"))}}
        """

        if type(new_base_pairs) is not dict:
            raise BiomoleculesException("Parameter new_base_pairs need to be a dictionary.")

        if not all(type(key) is str for key in new_base_pairs.keys()):
            raise BiomoleculesException("Keys of new_base_pairs need to be a strings.")

        if not all(type(value) is dict for value in new_base_pairs.values()):
            raise BiomoleculesException("Value of new_base_pairs need to be a dictionary.")

        if not all(type(key) is str for value in new_base_pairs.values() for key in value.keys()):
            raise BiomoleculesException("Keys of the base pair dictionary need to be a strings.")

        if not all(type(base_pairs) is tuple for value in new_base_pairs.values() for base_pairs in value.values()):
            raise BiomoleculesException("Value of the base pair dictionary need to be a tuple.")

        if not all(
                type(bond) is tuple for value in new_base_pairs.values() for base_pairs in value.values()
                for bond in base_pairs):
            raise BiomoleculesException("Base pair bonds need to be tuples.")

        if not all(
                len(bond) == 2 for value in new_base_pairs.values() for base_pairs in value.values()
                for bond in base_pairs):
            raise BiomoleculesException("Base pair bonds contains only two strings.")

        if not all(
                type(atom_name) is str for value in new_base_pairs.values() for base_pairs in value.values()
                for bond in base_pairs for atom_name in bond):
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
        """
        Contains all information to build and evaluate DNA.
        """

        super(RNA, self).__init__()

        # ABBREVIATION
        abbreviations = {"U": "uracil"}

        self.update_abbreviation_dictionary(abbreviations=abbreviations)

        # COVALENT BONDS
        # Backbone Bonds
        backbone_bonds = {}
        termini_bonds = {}

        full_backbone_bond_list = [("P", "OP1"), ("P", "OP2"), ("P", "O5\'"), ("O5\'", "C5\'"), ("C5\'", "H5\'1"),
                                   ("C5\'", "H5\'2"), ("C5\'", "C4\'"), ("C4\'", "H4\'"), ("C4\'", "O4\'"),
                                   ("C4\'", "C3\'"), ("O4\'", "C1\'"), ("C1\'", "H1\'"), ("C1\'", "C2\'"),
                                   ("C3\'", "H3\'"), ("C3\'", "C2\'"), ("C3\'", "O3\'"), ("C2\'", "H2\'1"),
                                   ("C2\'", "O2\'"), ("O2\'", "HO\'2")]

        backbone_bonds["residual"] = tuple(full_backbone_bond_list)

        # Backbone Bonds with P-capping
        p_capping_bond_list = full_backbone_bond_list.copy()
        for bond in full_backbone_bond_list:
            if "P" in bond:
                p_capping_bond_list.remove(bond)

        backbone_bonds["p_capped"] = tuple(p_capping_bond_list)

        # Update Backbone Dictionary
        self.update_backbone_bonds_dictionary(backbone_bonds=backbone_bonds)

        # Termini Bonds
        # P-capping bond
        termini_bonds["p_capped"] = (("O5\'", "H5T"), )

        # last residue bond
        termini_bonds['last_residue'] = (("O3'", "HO3\'"), )

        self.update_termini_bonds_dictionary(termini_bonds=termini_bonds)

        # Base Bonds
        base = {
            "uracil": (("C1\'", "N1"), ("N1", "C6"), ("N1", "C2"), ("C6", "H6"), ("C6", "C5"), ("C5", "H5"),
                       ("C5", "C4"), ("C4", "O4"), ("C4", "N3"), ("N3", "H3"), ("N3", "C2"), ("C2", "O2"))
        }

        self.update_side_chain_bonds_dictionary(new_side_chain=base)

        # HYDROGEN BONDS
        # Donors
        donors = {"uracil": {"H3": 1}, "backbone": {"HO\'2": 1}}

        self.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=donors, update_donors=True)

        # Acceptors
        acceptors = {
            "uracil": {
                "O2": 2,
                "O4": 2
            },
            "backbone": {
                "OP1": 2,
                "OP2": 2,
                "O2\'": 2,
                "O3\'": 2,
                "O4\'": 2,
                "O5\'": 2
            },
            "backbone_p_capped": {
                "O2\'": 2,
                "O3\'": 2,
                "O4\'": 2,
                "O5\'": 2
            }
        }

        self.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=acceptors, update_donors=False)

        # BASE PAIRS
        new_base_pairs = {"watson-crick": {"adenine_uracil": (("N6", "O4"), ("N1", "N3"))}}

        self.update_base_pairs_dictionary(new_base_pairs=new_base_pairs)


class DNA(NucleicAcid):
    def __init__(self):
        """
        Contains all information to build and evaluate DNA.
        """

        super(DNA, self).__init__()

        # ABBREVIATION
        abbreviations = {"T": "thymine"}

        self.update_abbreviation_dictionary(abbreviations=abbreviations)

        # COVALENT BONDS
        # Backbone Bonds
        full_backbone_bond_list = [("P", "OP1"), ("P", "OP2"), ("P", "O5\'"), ("O5\'", "C5\'"), ("C5\'", "H5\'1"),
                                   ("C5\'", "H5\'2"), ("C5\'", "C4\'"), ("C4\'", "H4\'"), ("C4\'", "O4\'"),
                                   ("C4\'", "C3\'"), ("O4\'", "C1\'"), ("C1\'", "H1\'"), ("C1\'", "C2\'"),
                                   ("C3\'", "H3\'"), ("C3\'", "C2\'"), ("C3\'", "O3\'"), ("C2\'", "H2\'1"),
                                   ("C2\'", "H2\'2")]

        backbone_bonds = {"residual": tuple(full_backbone_bond_list)}

        self.update_backbone_bonds_dictionary(backbone_bonds=backbone_bonds)

        # Base Bonds
        base = {
            "thymine": (("C1\'", "N1"), ("N1", "C6"), ("N1", "C2"), ("C6", "H6"), ("C6", "C5"), ("C5", "C5M"),
                        ("C5M", "H51"), ("C5M", "H52"), ("C5M", "H53"), ("C5", "C4"), ("C4", "O4"), ("C4", "N3"),
                        ("N3", "H3"), ("N3", "C2"), ("C2", "O2"))
        }

        self.update_side_chain_bonds_dictionary(new_side_chain=base)

        # HYDROGEN BONDS
        # Donors
        donors = {"thymine": {"H3": 1}, "backbone": {}}

        self.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=donors, update_donors=True)

        # Acceptors

        acceptors = {"thymine": {"O2": 2, "O4": 2}, "backbone": {"OP1": 2, "OP2": 2, "O3\'": 2, "O4\'": 2, "O5\'": 2}}

        self.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=acceptors, update_donors=False)

        # BASE PAIRS
        new_base_pairs = {"watson-crick": {"adenine_thymine": (("N6", "O4"), ("N1", "N3"))}}

        self.update_base_pairs_dictionary(new_base_pairs=new_base_pairs)


class Protein(Biomolecule):
    def __init__(self):
        super(Protein, self).__init__()

        # ABBREVIATION
        abbreviations = {
            "GLY": "glycine",
            "ALA": "alanine",
            "VAL": "valine",
            "LEU": "leucine",
            "ILE": "isoleucine",
            "SER": "serine",
            "THR": "threonine",
            "PRO": "proline",
            "ASP": "aspartic acid",
            "GLU": "glutamic acid",
            "ASN": "asparagine",
            "GLN": "glutamine",
            "MET": "methionine",
            "CYS": "cysteine",
            "LYS": "lysine",
            "ARG": "arginine",
            "HIS": "histidine",
            "PHE": "phenylalanine",
            "TYR": "tyrosine",
            "TRP": "tryptophan",
            "ACE": "acetyl group",
            "NME": "methylamine"
        }

        self.update_abbreviation_dictionary(abbreviations=abbreviations)

        # MEASURES
        # Angles
        dihedral_angles_dictionary = {
            "psi": (("N", "CA", "C", "N"), (0, 0, 0, 1), r"$\psi$"),
            "phi": (("C", "N", "CA", "C"), (-1, 0, 0, 0), r"$\phi$"),
            "omega": (("CA", "C", "N", "CA"), (0, 0, 1, 1), r"$\omega$"),
            "chi1": (("N", "CA", "CB", "CG"), (0, 0, 0, 0), r"$\chi_1$"),
            "chi2": (("CA", "CB", "CG", "CD"), (0, 0, 0, 0), r"$\chi_2$"),
            "chi3": (("CB", "CG", "CD", "CE"), (0, 0, 0, 0), r"$\chi_3$"),
            "chi4": (("CG", "CD", "CE", "NZ"), (0, 0, 0, 0), r"$\chi_4$")
        }

        self.update_dihedral_angle_dictionary(dihedral_angles=dihedral_angles_dictionary)

        # Distances
        distance_dict = {"CA_CA": (("CA", "CA"), r"$C_{\alpha}$ to $C_{\alpha}$"),
                         "CA_CB": (("CA", "CB"), r"$C_{\alpha}$ to $C_{\beta}$"),
                         "CA_O": (("CA", "O"), r"$C_{\alpha}$ to $O$")}

        self.update_distances_dictionary(distances=distance_dict)

        # COVALENT BONDS
        # Backbone Bonds
        self.set_bonds_between_residues("C", "N")

        backbone_bonds = {"residual": (("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "C"), ("C", "O"))}

        self.update_backbone_bonds_dictionary(backbone_bonds=backbone_bonds)

        # Termini Bonds
        termini_bonds = {
            "acetyl group": (("C", "O"), ("C", "CH3"), ("CH3", "H1"), ("CH3", "H2"), ("CH3", "H3")),
            "methylamine": (("N", "H"), ("N", "C"), ("C", "H1"), ("C", "H2"), ("C", "H3"))
        }

        self.update_termini_bonds_dictionary(termini_bonds=termini_bonds)

        # Base Bonds
        amino_acids = {
            "glycine": (("CA", "HA2"), ),
            "alanine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "HB3")),
            "valine": (("CA", "CB"), ("CB", "HB1"), ("CB", "CG1"), ("CG1", "HG11"), ("CG1", "HG12"), ("CG1", "HG13"),
                       ("CB", "CG2"), ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "HG23")),
            "leucine":
            (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"), ("CG", "CD1"), ("CD1", "HD11"),
             ("CD1", "HD12"), ("CD1", "HD13"), ("CG", "CD2"), ("CD2", "HD21"), ("CD2", "HD22"), ("CD2", "HD23")),
            "isoleucine": (("CA", "CB"), ("CB", "HB1"), ("CB", "CG1"), ("CG1", "HG11"), ("CG1", "HG12"),
                           ("CG1", "HG13"), ("CB", "CG2"), ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "CD"),
                           ("CD", "HD1"), ("CD", "HD2"), ("CD", "HD3")),
            "serine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "OG"), ("OG", "HG1")),
            "threonine": (("CA", "CB"), ("CB", "HB1"), ("CB", "OG1"), ("OG1", "HG11"), ("CB", "CG2"), ("CG2", "HG21"),
                          ("CG2", "HG22"), ("CG2", "HG23")),
            "proline": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"), ("CG", "HG2"),
                        ("CG", "CD"), ("CD", "HD1"), ("CD", "HD2"), ("CD", "N")),
            "aspartic acid": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
            "glutamic acid": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"), ("CG", "HG2"),
                              ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")),
            "asparagine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2"),
                           ("ND2", "HD21"), ("ND2", "HD22")),
            "glutamine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"), ("CG", "HG2"),
                          ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2"), ("NE2", "HE21"), ("NE2", "HE22")),
            "methionine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"), ("CG", "HG2"),
                           ("CG", "SD"), ("SD", "CE"), ("CE", "HE1"), ("CE", "HE2"), ("CE", "HE3")),
            "cysteine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "SG"), ("SG", "HG1")),
            "lysine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"), ("CG", "HG2"),
                       ("CG", "CD"), ("CD", "HD1"), ("CD", "HD2"), ("CD", "CE"), ("CE", "HE1"), ("CE", "HE2"),
                       ("CE", "NZ"), ("NZ", "HZ1"), ("NZ", "HZ2"), ("NZ", "HZ3")),
            "arginine":
            (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "HG1"), ("CG", "HG2"), ("CG", "CD"),
             ("CD", "HD1"), ("CD", "HD2"), ("CD", "NE"), ("NE", "HE1"), ("NE", "CZ"), ("CZ", "NH1"), ("NH1", "HH11"),
             ("NH1", "HH12"), ("CZ", "NH2"), ("NH2", "HH21"), ("NH2", "HH22")),
            "histidine":
            (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "CD1"), ("CD1", "HD11"), ("CG", "ND2"),
             ("ND2", "HD21"), ("CD1", "NE1"), ("NE1", "HE11"), ("ND2", "CE2"), ("CE2", "HE21"), ("NE1", "CE2")),
            "phenylalanine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "CD1"), ("CD1", "HD11"),
                              ("CG", "CD2"), ("CD2", "HD21"), ("CD1", "CE1"), ("CE1", "HE11"), ("CD2", "CE2"),
                              ("CE2", "HE21"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "HZ1")),
            "tyrosine": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "CD1"), ("CD1", "HD11"),
                         ("CG", "CD2"), ("CD2", "HD21"), ("CD1", "CE1"), ("CE1", "HE11"), ("CD2", "CE2"),
                         ("CE2", "HE21"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH"), ("OH", "HH1")),
            "tryptophan": (("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "CG"), ("CG", "CD1"), ("CD1", "CE1"),
                           ("CE1", "HE11"), ("CE1", "CZ1"), ("CZ1", "HZ11"), ("CG", "CD2"), ("CD2", "HD21"),
                           ("CD2", "NE2"), ("NE2", "HE21"), ("NE2", "CZ2"), ("CZ2", "CH2"), ("CH2", "HH21"),
                           ("CH2", "CTH2"), ("CTH2", "HTH21"), ("CD1", "CZ2"), ("CZ1", "CTH2"))
        }

        self.update_side_chain_bonds_dictionary(new_side_chain=amino_acids)

        # HYDROGEN BONDS
        # Donors
        # TODO: Protein Donors
        donors_dict = {}

        self.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=donors_dict, update_donors=True)

        # Acceptors
        # TODO: Protein Acceptors
        acceptors_dict = {}

        self.update_hydrogen_bond_dictionary(hydrogen_bond_atoms=acceptors_dict, update_donors=False)
