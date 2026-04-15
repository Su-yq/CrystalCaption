import numpy as np
from rdkit import Chem


class HydrogenBondCriterion:

    def __init__(self):
        self.donor_types = {
            '+nitrogen': False,
            'metal bound N': True,
            'imine N': True,
            'aromatic (6-ring) N': True,
            'amide or thioamide N': True,
            'planar N': True,
            'pyramidal N': True,
            'ammonium N (NH4+, RNH3+ etc.)': True,
            'unclassified N': True,
            '-nitrogen': False,

            '+oxygen': False,
            'metal bound O': True,
            'water O': True,
            'hydroxyl O': True,
            'unclassified O': True,
            '-oxygen': False,

            '+sulphur': False,
            'metal bound S': True,
            'non metal bound S': True,
            '-sulphur': False,

            '+carbon': False,
            'sp C': False,
            'sp2 C': False,
            'sp3 C': False,
            'aromatic C': False,
            'carboncationic C': False,
            '-carbon': False,
        }

        self.acceptor_types = {
            '+nitrogen': False,
            'metal bound N': True,
            'terminal N (cyano, etc.)': True,
            'aromatic (6-ring) N': True,
            'other 2-coordinate N': True,
            '3-coordinate N': True,
            'unclassified N': True,
            '-nitrogen': False,

            '+oxygen': False,
            'metal bound O': True,
            'carboxylate O': True,
            'other terminal O (C=O, S=O ...)': True,
            'water O': True,
            'hydroxyl O': True,
            'bridging O (ether, etc.)': True,
            'unclassified O': True,
            '-oxygen': False,

            '+sulphur': False,
            'metal bound S': True,
            'terminal S': True,
            'unclassified S': True,
            '-sulphur': False,

            '+fluorine': False,
            'metal bound F': True,
            'fluoride ion (F-)': True,
            'unclassified F': False,
            '-fluorine': False,
        }

    def _classify_atom(self, rdkit_atom, mol=None):

        atom_types = []
        symbol = rdkit_atom.GetSymbol()
        atomic_num = rdkit_atom.GetAtomicNum()
        formal_charge = rdkit_atom.GetFormalCharge()
        hybridization = rdkit_atom.GetHybridization()
        is_aromatic = rdkit_atom.GetIsAromatic()

        if atomic_num == 6:
            if formal_charge > 0:
                atom_types.append('+carbon')
                atom_types.append('carboncationic C')
            elif formal_charge < 0:
                atom_types.append('-carbon')
            else:
                if hybridization == Chem.HybridizationType.SP:
                    atom_types.append('sp C')
                elif hybridization == Chem.HybridizationType.SP2:
                    if is_aromatic:
                        atom_types.append('aromatic C')
                    atom_types.append('sp2 C')
                elif hybridization == Chem.HybridizationType.SP3:
                    atom_types.append('sp3 C')

        elif atomic_num == 7:
            if formal_charge > 0:
                atom_types.append('+nitrogen')
                if self._is_ammonium_nitrogen(rdkit_atom, mol):
                    atom_types.append('ammonium N (NH4+, RNH3+ etc.)')
            elif formal_charge < 0:
                atom_types.append('-nitrogen')
            else:
                if is_aromatic:
                    atom_types.append('aromatic (6-ring) N')
                atom_types.append('unclassified N')

        elif atomic_num == 8:
            if formal_charge > 0:
                atom_types.append('+oxygen')
            elif formal_charge < 0:
                atom_types.append('-oxygen')
                if self._is_carboxylate_oxygen(rdkit_atom, mol):
                    atom_types.append('carboxylate O')
            else:
                if self._is_hydroxyl_oxygen(rdkit_atom, mol):
                    atom_types.append('hydroxyl O')
                elif self._is_water_oxygen(rdkit_atom, mol):
                    atom_types.append('water O')
                else:
                    atom_types.append('unclassified O')

        return atom_types

    def _is_ammonium_nitrogen(self, rdkit_atom, mol):
        if rdkit_atom.GetFormalCharge() > 0:
            degree = rdkit_atom.GetDegree()
            total_H = rdkit_atom.GetTotalNumHs()
            return degree + total_H >= 4
        return False

    def _is_carboxylate_oxygen(self, rdkit_atom, mol):
        if rdkit_atom.GetFormalCharge() < 0:
            for neighbor in rdkit_atom.GetNeighbors():
                if neighbor.GetSymbol() == 'C' and neighbor.GetHybridization() == Chem.HybridizationType.SP2:
                    return True
        return False

    def _is_hydroxyl_oxygen(self, rdkit_atom, mol):
        for neighbor in rdkit_atom.GetNeighbors():
            if neighbor.GetSymbol() == 'H':
                return True
        return False

    def _is_water_oxygen(self, rdkit_atom, mol):
        h_count = 0
        for neighbor in rdkit_atom.GetNeighbors():
            if neighbor.GetSymbol() == 'H':
                h_count += 1
        return h_count == 2

    def is_donor(self, rdkit_atom, mol=None):
        atom_types = self._classify_atom(rdkit_atom, mol)

        for atom_type in atom_types:
            if atom_type in self.donor_types:
                return self.donor_types[atom_type]

        if mol is not None:
            for neighbor in rdkit_atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    return True
        return False

    def is_acceptor(self, rdkit_atom, mol=None):
        atom_types = self._classify_atom(rdkit_atom, mol)

        for atom_type in atom_types:
            if atom_type in self.acceptor_types:
                return self.acceptor_types[atom_type]

        symbol = rdkit_atom.GetSymbol()
        acceptor_elements = {'O', 'N', 'F', 'Cl', 'Br', 'I', 'S'}
        return symbol in acceptor_elements