# -*- coding: utf-8 -*-
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

class HydrogenBondCriterion:

    @staticmethod
    def is_acceptor(rdkit_atom):
        symbol = rdkit_atom.GetSymbol()
        atomic_num = rdkit_atom.GetAtomicNum()

        acceptor_atoms = {7, 8, 9, 16, 17, 35, 53}

        if atomic_num in acceptor_atoms:
            hybridization = rdkit_atom.GetHybridization()
            formal_charge = rdkit_atom.GetFormalCharge()

            if formal_charge < 0:
                return True

            if hybridization in [Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]:
                if symbol in ['O', 'N']:
                    return True

            if symbol == 'O' and formal_charge == 0:
                return True
            if symbol == 'N' and formal_charge == 0 and not rdkit_atom.GetIsAromatic():
                return True

        return False

    @staticmethod
    def is_donor(rdkit_atom, mol=None):
        symbol = rdkit_atom.GetSymbol()

        donor_atoms = {'O', 'N', 'F', 'S'}

        if symbol in donor_atoms:
            if mol is not None:
                for neighbor in rdkit_atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'H':
                        return True
            return True

        return False


HBondCriterion = HydrogenBondCriterion()

SYBYL_TYPE_MAP = {
    1: 'H', 6: 'C.3', 7: 'N.3', 8: 'O.3', 9: 'F', 15: 'P.3', 16: 'S.3', 17: 'Cl', 35: 'Br', 53: 'I'
}

VDW_RADII = {
    1: 1.20,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    15: 1.80,
    16: 1.80,
    17: 1.75,
    35: 1.85,
    53: 1.98
}


class atom_feat(object):
    def __init__(self, Atom, hb_criterion=HBondCriterion, mol=None):
        self.coordinates = Atom.rdkit_coor
        self.symbol = Atom.rdkit_atom.GetSymbol()
        self.hybridization = Atom.rdkit_atom.GetHybridization().__str__()

        self.chirality = self._get_chirality(Atom.rdkit_atom)
        self.is_chiral = Atom.rdkit_atom.HasProp('_ChiralityPossible') or self.chirality != 'CHI_UNSPECIFIED'

        self.explicitvalence = Atom.rdkit_atom.GetExplicitValence()
        self.implicitvalence = Atom.rdkit_atom.GetImplicitValence()
        self.totalnumHs = Atom.rdkit_atom.GetTotalNumHs()

        self.formalcharge = Atom.rdkit_atom.GetFormalCharge()
        self.radical_electrons = Atom.rdkit_atom.GetNumRadicalElectrons()
        self.is_aromatic = Atom.rdkit_atom.GetIsAromatic()

        self.is_acceptor = hb_criterion.is_acceptor(Atom.rdkit_atom)
        self.is_donor = hb_criterion.is_donor(Atom.rdkit_atom, mol)

        self.is_spiro = self._is_spiro_atom(Atom.rdkit_atom, mol)
        self.is_cyclic = Atom.rdkit_atom.IsInRing()
        self.is_metal = Atom.rdkit_atom.GetAtomicNum() in [3, 11, 12, 19, 20, 26, 29, 30, 47, 50, 79]

        self.atomic_weight = Atom.rdkit_atom.GetMass()
        self.atomic_number = Atom.rdkit_atom.GetAtomicNum()

        self.vdw_radius = VDW_RADII.get(Atom.rdkit_atom.GetAtomicNum(), 1.5)

        self.sybyl_type = SYBYL_TYPE_MAP.get(
            Atom.rdkit_atom.GetAtomicNum(),
            Atom.rdkit_atom.GetSymbol()
        )

        self.degree = Atom.rdkit_atom.GetDegree()

    def _get_chirality(self, rdkit_atom):
        chiral_tag = rdkit_atom.GetChiralTag()
        chiral_map = {
            Chem.ChiralType.CHI_UNSPECIFIED: 'CHI_UNSPECIFIED',
            Chem.ChiralType.CHI_TETRAHEDRAL_CW: 'CHI_TETRAHEDRAL_CW',
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 'CHI_TETRAHEDRAL_CCW',
            Chem.ChiralType.CHI_OTHER: 'CHI_OTHER'
        }
        return chiral_map.get(chiral_tag, 'CHI_UNSPECIFIED')

    def _is_spiro_atom(self, rdkit_atom, mol):
        if mol is None:
            return False

        atom_idx = rdkit_atom.GetIdx()
        ring_info = mol.GetRingInfo()
        atom_rings = []

        for ring in ring_info.AtomRings():
            if atom_idx in ring:
                atom_rings.append(set(ring))

        if len(atom_rings) >= 2:
            common_atoms = set.intersection(*atom_rings)
            if len(common_atoms) == 1 and atom_idx in common_atoms:
                return True

        return False


class Atom(object):
    def __init__(self, rdkit_atom, hb_criterion=HBondCriterion, mol=None):
        self.rdkit_atom = rdkit_atom
        self.hb_criterion = hb_criterion
        self.mol = mol

        self.idx = rdkit_atom.GetIdx()

        if mol is not None and mol.GetNumConformers() > 0:
            self.rdkit_coor = np.array([i for i in mol.GetConformer().GetAtomPosition(self.idx)])
        else:
            self.rdkit_coor = np.array([0.0, 0.0, 0.0])

        self.index = self.idx

    @property
    def feature(self):
        return atom_feat(self, hb_criterion=self.hb_criterion, mol=self.mol)

    @property
    def get_bonds(self):
        self.bonds = {}
        if self.mol is not None:
            for bond in self.rdkit_atom.GetBonds():
                bond_key = (self.idx, bond.GetOtherAtomIdx(self.idx))
                self.bonds.setdefault(bond_key, Bond(bond, self.mol))
        return self.bonds

    @property
    def get_adjHs(self):
        Hs = []
        if self.mol is not None:
            for bond in self.rdkit_atom.GetBonds():
                adj_atom = bond.GetOtherAtom(self.rdkit_atom)
                if adj_atom.GetSymbol() == 'H':
                    Hs.append(bond.GetOtherAtomIdx(self.idx))
        return Hs


class Bond(object):
    def __init__(self, rdkit_bond, mol=None):
        self.end_atom_idx = rdkit_bond.GetEndAtomIdx()
        self.begin_atom_idx = rdkit_bond.GetBeginAtomIdx()

        if mol is not None and mol.GetNumConformers() > 0:
            conf = mol.GetConformer()
            self.end_atom_coor = [i for i in conf.GetAtomPosition(self.end_atom_idx)]
            self.begin_atom_coor = [i for i in conf.GetAtomPosition(self.begin_atom_idx)]
        else:
            self.end_atom_coor = [0.0, 0.0, 0.0]
            self.begin_atom_coor = [0.0, 0.0, 0.0]

        bond_type_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        self.bond_type = rdkit_bond.GetBondType().__str__()
        try:
            self.type_number = bond_type_list.index(self.bond_type) + 1
        except:
            print('bond type is out of range {}'.format(self.bond_type))
            self.type_number = 5

        self.length = np.linalg.norm(np.array(self.end_atom_coor) - np.array(self.begin_atom_coor))
        self.is_ring = rdkit_bond.IsInRing()
        self.is_conjugated = rdkit_bond.GetIsConjugated()